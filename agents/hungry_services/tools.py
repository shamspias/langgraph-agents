"""
Tools for Hungry Services Agent
"""
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import logging


class FoodSearchTools:
    """Tools for searching food information"""

    def __init__(self):
        """Initialize food search tools"""
        self.logger = logging.getLogger("tools.food_search")
        self.search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        self.search_tool = DuckDuckGoSearchRun(api_wrapper=self.search_wrapper)

    @tool
    def search_food(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for general food information online

        Args:
            query: Food item or cuisine to search for

        Returns:
            List of search results with food information
        """
        try:
            # Enhance query for food-specific search
            enhanced_query = f"{query} food nutrition cuisine"
            results = self.search_wrapper.results(enhanced_query, max_results=5)

            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", "")
                })

            return formatted_results
        except Exception as e:
            logging.error(f"Error searching food: {str(e)}")
            return []

    @tool
    def search_recipes(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for recipes online

        Args:
            query: Recipe or dish to search for

        Returns:
            List of recipe search results
        """
        try:
            # Enhance query for recipe-specific search
            enhanced_query = f"{query} recipe ingredients instructions cooking"
            results = self.search_wrapper.results(enhanced_query, max_results=3)

            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", ""),
                    "type": "recipe"
                })

            return formatted_results
        except Exception as e:
            logging.error(f"Error searching recipes: {str(e)}")
            return []

    @tool
    def search_restaurants(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for restaurants

        Args:
            query: Restaurant or cuisine type to search for

        Returns:
            List of restaurant search results
        """
        try:
            # Enhance query for restaurant-specific search
            enhanced_query = f"{query} restaurant near me delivery menu"
            results = self.search_wrapper.results(enhanced_query, max_results=5)

            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", ""),
                    "type": "restaurant"
                })

            return formatted_results
        except Exception as e:
            logging.error(f"Error searching restaurants: {str(e)}")
            return []

    @tool
    def get_nutritional_info(self, food_item: str) -> Dict[str, Any]:
        """
        Get nutritional information for a food item

        Args:
            food_item: Name of the food item

        Returns:
            Nutritional information if available
        """
        try:
            # Search for nutritional information
            query = f"{food_item} calories protein carbs fat nutrition facts"
            results = self.search_wrapper.results(query, max_results=2)

            if results:
                return {
                    "food_item": food_item,
                    "nutritional_info": results[0].get("snippet", ""),
                    "source": results[0].get("link", "")
                }

            return {"food_item": food_item, "nutritional_info": "No information found"}
        except Exception as e:
            logging.error(f"Error getting nutritional info: {str(e)}")
            return {"food_item": food_item, "error": str(e)}

    @tool
    def search_dietary_options(self, query: str, dietary_restriction: str) -> List[Dict[str, Any]]:
        """
        Search for food with specific dietary restrictions

        Args:
            query: Food or cuisine to search for
            dietary_restriction: Dietary restriction (vegan, gluten-free, keto, etc.)

        Returns:
            List of dietary-specific search results
        """
        try:
            # Enhance query with dietary restriction
            enhanced_query = f"{query} {dietary_restriction} options alternatives"
            results = self.search_wrapper.results(enhanced_query, max_results=4)

            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", ""),
                    "dietary": dietary_restriction
                })

            return formatted_results
        except Exception as e:
            logging.error(f"Error searching dietary options: {str(e)}")
            return []

    def get_tools(self) -> List:
        """Get all available tools"""
        return [
            self.search_food,
            self.search_recipes,
            self.search_restaurants,
            self.get_nutritional_info,
            self.search_dietary_options
        ]
