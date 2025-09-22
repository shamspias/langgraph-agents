"""
Hungry Services Agent for food search and recommendations
"""
from typing import Dict, Any, Type, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from core.base_agent import BaseAgent
from core.state import FoodSearchState, InputState, OutputState
from .tools import FoodSearchTools


class HungryServicesAgent(BaseAgent):
    """Agent for searching food information and providing recommendations"""

    def __init__(self, **kwargs):
        """Initialize Hungry Services Agent"""
        super().__init__(
            name="HungryServices",
            system_prompt=self._get_food_system_prompt(),
            **kwargs
        )
        self.tools = FoodSearchTools()
        self.model_with_tools = self.model.bind_tools(self.tools.get_tools())

    def _get_food_system_prompt(self) -> str:
        """Get specialized system prompt for food services"""
        return """You are a helpful food expert assistant called Hungry Services. 
        Your role is to:
        1. Search for food and restaurant information online
        2. Provide detailed nutritional information when available
        3. Give personalized food recommendations based on preferences
        4. Suggest recipes and cooking tips
        5. Find nearby restaurants and delivery options

        Always provide accurate, helpful, and appetizing descriptions of food.
        When searching, be thorough and provide multiple options when possible.
        Include relevant details like cuisine type, price range, ratings, and dietary information.
        """

    def get_state_schema(self) -> Type:
        """Get the state schema for this agent"""
        return FoodSearchState

    def build_graph(self) -> StateGraph:
        """Build the food search agent graph"""
        workflow = StateGraph(
            FoodSearchState,
            input=InputState,
            output=OutputState
        )

        # Add nodes
        workflow.add_node("analyze_query", self.analyze_food_query)
        workflow.add_node("search_food", self.search_food_info)
        workflow.add_node("generate_response", self.generate_food_response)

        # Add edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "search_food")
        workflow.add_edge("search_food", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow

    def analyze_food_query(self, state: FoodSearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Analyze the user's food query"""
        self.logger.info("Analyzing food query")

        messages = state["messages"]
        last_message = messages[-1].content if messages else ""

        # Extract food query
        analysis_prompt = """Analyze this food-related query and extract:
        1. The main food item or cuisine being asked about
        2. Any dietary restrictions or preferences mentioned
        3. Whether they want recipes, restaurants, or nutritional info
        4. Location if mentioned for restaurant searches

        Query: {query}
        """

        response = self.model.invoke([
            SystemMessage(content=analysis_prompt.format(query=last_message))
        ])

        return {
            "query": last_message,
            "metadata": {
                "analysis": response.content,
                "timestamp": str(config.get("timestamp", ""))
            }
        }

    def search_food_info(self, state: FoodSearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Search for food information online"""
        self.logger.info(f"Searching for food info: {state['query']}")

        try:
            # Use tools to search
            search_query = state["query"]

            # Search for general food info
            food_results = self.tools.search_food(search_query)

            # Search for recipes if requested
            recipe_results = []
            if "recipe" in search_query.lower() or "how to make" in search_query.lower():
                recipe_results = self.tools.search_recipes(search_query)

            # Search for restaurants if requested
            restaurant_results = []
            if "restaurant" in search_query.lower() or "near me" in search_query.lower():
                restaurant_results = self.tools.search_restaurants(search_query)

            # Combine results
            all_results = {
                "food_info": food_results,
                "recipes": recipe_results,
                "restaurants": restaurant_results
            }

            return {
                "search_results": [all_results],
                "metadata": {
                    **state.get("metadata", {}),
                    "search_completed": True
                }
            }

        except Exception as e:
            self.logger.error(f"Error searching food info: {str(e)}")
            return {
                "error": f"Search error: {str(e)}",
                "search_results": []
            }

    def generate_food_response(self, state: FoodSearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Generate comprehensive food response"""
        self.logger.info("Generating food response")

        query = state["query"]
        search_results = state.get("search_results", [])

        # Prepare context from search results
        context = self._prepare_food_context(search_results)

        # Generate response
        response_prompt = """Based on the search results, provide a comprehensive and helpful response about the food query.

        Query: {query}

        Search Results:
        {context}

        Please provide:
        1. Direct answer to the query
        2. Additional helpful information
        3. Recommendations if applicable
        4. Any warnings or dietary considerations

        Make the response appetizing and informative."""

        response = self.model.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=response_prompt.format(
                query=query,
                context=context
            ))
        ])

        # Generate recommendations
        recommendations = self._generate_recommendations(query, search_results)

        return {
            "messages": [AIMessage(content=response.content)],
            "response": response.content,
            "recommendations": recommendations,
            "food_info": search_results[0] if search_results else None,
            "metadata": {
                **state.get("metadata", {}),
                "response_generated": True
            }
        }

    def _prepare_food_context(self, search_results: list) -> str:
        """Prepare context from search results"""
        if not search_results:
            return "No search results available."

        context_parts = []

        for result in search_results:
            if isinstance(result, dict):
                # Food info
                if "food_info" in result and result["food_info"]:
                    context_parts.append("=== Food Information ===")
                    for info in result["food_info"][:3]:  # Top 3 results
                        if isinstance(info, dict):
                            context_parts.append(f"- {info.get('title', 'No title')}: {info.get('snippet', '')}")

                # Recipes
                if "recipes" in result and result["recipes"]:
                    context_parts.append("\n=== Recipes ===")
                    for recipe in result["recipes"][:2]:  # Top 2 recipes
                        if isinstance(recipe, dict):
                            context_parts.append(f"- {recipe.get('title', 'No title')}: {recipe.get('snippet', '')}")

                # Restaurants
                if "restaurants" in result and result["restaurants"]:
                    context_parts.append("\n=== Restaurants ===")
                    for restaurant in result["restaurants"][:3]:  # Top 3 restaurants
                        if isinstance(restaurant, dict):
                            context_parts.append(
                                f"- {restaurant.get('title', 'No title')}: {restaurant.get('snippet', '')}")

        return "\n".join(context_parts) if context_parts else "No relevant information found."

    def _generate_recommendations(self, query: str, search_results: list) -> list:
        """Generate food recommendations"""
        recommendations = []

        try:
            prompt = f"""Based on the query '{query}', suggest 3 related food recommendations.
            These should be similar dishes, complementary items, or alternatives.
            Return as a simple list."""

            response = self.model.invoke([
                SystemMessage(content=prompt)
            ])

            # Parse recommendations from response
            lines = response.content.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Clean up the line
                    recommendation = line.lstrip('0123456789.-•').strip()
                    if recommendation:
                        recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")

        return recommendations[:3]  # Return top 3 recommendations