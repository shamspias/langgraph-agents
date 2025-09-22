"""
Multipurpose Bot Agent with routing to specialized subagents
"""
from typing import Dict, Any, Type, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from core.base_agent import BaseAgent
from core.state import MultipurposeState, InputState, OutputState, IntentClassification
from .subagents.math_agent import MathAgent
from .subagents.chitchat_agent import ChitchatAgent
from .subagents.headphones_agent import HeadphonesAgent


class MultipurposeBot(BaseAgent):
    """Main multipurpose bot that routes to specialized subagents"""

    def __init__(self, **kwargs):
        """Initialize Multipurpose Bot"""
        super().__init__(
            name="MultipurposeBot",
            system_prompt=self._get_multipurpose_prompt(),
            **kwargs
        )

        # Initialize subagents
        self.math_agent = MathAgent()
        self.chitchat_agent = ChitchatAgent()
        self.headphones_agent = HeadphonesAgent()

        # Intent classifier
        self.intent_classifier = self.model.with_structured_output(IntentClassification)

    def _get_multipurpose_prompt(self) -> str:
        """Get system prompt for multipurpose bot"""
        return """You are a versatile AI assistant that can handle multiple types of queries:

        1. Mathematical calculations - I can solve complex math problems step by step
        2. General chitchat - I can have friendly conversations on various topics
        3. Headphones expertise - I have detailed knowledge about headphones, audio equipment, and recommendations

        I analyze each query to understand what you need and provide the most appropriate response.
        I'm here to help with accuracy, friendliness, and expertise in my specialized areas."""

    def get_state_schema(self) -> Type:
        """Get the state schema for this agent"""
        return MultipurposeState

    def build_graph(self) -> StateGraph:
        """Build the multipurpose bot graph"""
        workflow = StateGraph(
            MultipurposeState,
            input=InputState,
            output=OutputState
        )

        # Add nodes
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("route_math", self.route_to_math)
        workflow.add_node("route_chitchat", self.route_to_chitchat)
        workflow.add_node("route_headphones", self.route_to_headphones)
        workflow.add_node("handle_unknown", self.handle_unknown)
        workflow.add_node("prepare_response", self.prepare_response)

        # Add routing logic
        workflow.add_edge(START, "classify_intent")

        # Conditional routing based on intent
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_by_intent,
            {
                "math": "route_math",
                "chitchat": "route_chitchat",
                "headphones": "route_headphones",
                "unknown": "handle_unknown"
            }
        )

        # All routes lead to prepare_response
        workflow.add_edge("route_math", "prepare_response")
        workflow.add_edge("route_chitchat", "prepare_response")
        workflow.add_edge("route_headphones", "prepare_response")
        workflow.add_edge("handle_unknown", "prepare_response")

        # Final response preparation
        workflow.add_edge("prepare_response", END)

        return workflow

    def classify_intent(self, state: MultipurposeState, config: RunnableConfig) -> Dict[str, Any]:
        """Classify the intent of the user's query"""
        self.logger.info("Classifying user intent")

        messages = state["messages"]
        last_message = messages[-1].content if messages else ""

        # Classify intent
        classification_prompt = """Classify the following query into one of these categories:
        - 'math': Mathematical calculations, equations, or numerical problems
        - 'chitchat': General conversation, greetings, personal questions, or casual topics
        - 'headphones': Questions about headphones, audio equipment, or sound-related topics
        - 'unknown': Queries that don't clearly fit the above categories

        Query: {query}

        Provide your classification with reasoning."""

        classification = self.intent_classifier.invoke([
            SystemMessage(content=classification_prompt.format(query=last_message))
        ])

        self.logger.info(f"Intent classified as: {classification.intent} (confidence: {classification.confidence})")

        return {
            "query": last_message,
            "intent": classification.intent,
            "metadata": {
                "intent_confidence": classification.confidence,
                "reasoning": classification.reasoning
            }
        }

    def route_by_intent(self, state: MultipurposeState) -> Literal["math", "chitchat", "headphones", "unknown"]:
        """Determine routing based on classified intent"""
        intent = state.get("intent", "unknown")
        self.logger.info(f"Routing to {intent} handler")
        return intent

    def route_to_math(self, state: MultipurposeState, config: RunnableConfig) -> Dict[str, Any]:
        """Route to math agent for calculation"""
        self.logger.info("Routing to math agent")

        query = state["query"]

        # Compile math agent if needed
        if self.math_agent.compiled_graph is None:
            self.math_agent.compile()

        # Invoke math agent
        result = self.math_agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )

        return {
            "sub_state": result,
            "response": result.get("response", ""),
            "metadata": {
                **state.get("metadata", {}),
                "handler": "math"
            }
        }

    def route_to_chitchat(self, state: MultipurposeState, config: RunnableConfig) -> Dict[str, Any]:
        """Route to chitchat agent for conversation"""
        self.logger.info("Routing to chitchat agent")

        query = state["query"]

        # Compile chitchat agent if needed
        if self.chitchat_agent.compiled_graph is None:
            self.chitchat_agent.compile()

        # Invoke chitchat agent
        result = self.chitchat_agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )

        return {
            "sub_state": result,
            "response": result.get("response", ""),
            "metadata": {
                **state.get("metadata", {}),
                "handler": "chitchat"
            }
        }

    def route_to_headphones(self, state: MultipurposeState, config: RunnableConfig) -> Dict[str, Any]:
        """Route to headphones agent for audio expertise"""
        self.logger.info("Routing to headphones agent")

        query = state["query"]

        # Compile headphones agent if needed
        if self.headphones_agent.compiled_graph is None:
            self.headphones_agent.compile()

        # Invoke headphones agent
        result = self.headphones_agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )

        return {
            "sub_state": result,
            "response": result.get("response", ""),
            "metadata": {
                **state.get("metadata", {}),
                "handler": "headphones"
            }
        }

    def handle_unknown(self, state: MultipurposeState, config: RunnableConfig) -> Dict[str, Any]:
        """Handle unknown intents"""
        self.logger.info("Handling unknown intent")

        query = state["query"]

        response_prompt = """The user's query doesn't clearly match our specialized capabilities.
        Provide a helpful response that:
        1. Acknowledges their question
        2. Explains what we can help with (math, general chat, headphones)
        3. Attempts to provide some value based on the query

        Query: {query}"""

        response = self.model.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=response_prompt.format(query=query))
        ])

        return {
            "response": response.content,
            "metadata": {
                **state.get("metadata", {}),
                "handler": "unknown"
            }
        }

    def prepare_response(self, state: MultipurposeState, config: RunnableConfig) -> Dict[str, Any]:
        """Prepare final response"""
        self.logger.info("Preparing final response")

        response = state.get("response", "I'm sorry, I couldn't process your request.")

        # Add any post-processing or formatting here
        final_response = self._format_response(response, state.get("metadata", {}))

        return {
            "messages": [AIMessage(content=final_response)],
            "response": final_response,
            "metadata": state.get("metadata", {})
        }

    def _format_response(self, response: str, metadata: Dict[str, Any]) -> str:
        """Format the final response"""
        # Add any special formatting based on handler type
        handler = metadata.get("handler", "unknown")

        if handler == "math":
            # Math responses might already be well-formatted
            return response
        elif handler == "headphones":
            # Add disclaimer if needed
            if "recommendation" in response.lower():
                response += ("\n\n*Note: These recommendations are based on general information. Please consider your "
                             "specific needs and budget.*")

        return response
