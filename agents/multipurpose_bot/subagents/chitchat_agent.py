"""
Chitchat Agent for general conversation
"""
from typing import Dict, Any, Type
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from core.base_agent import BaseAgent
from core.state import BaseState, InputState, OutputState


class ChitchatAgent(BaseAgent):
    """Agent for handling general conversation and chitchat"""

    def __init__(self, **kwargs):
        """Initialize Chitchat Agent"""
        super().__init__(
            name="ChitchatAgent",
            system_prompt=self._get_chitchat_prompt(),
            **kwargs
        )

    def _get_chitchat_prompt(self) -> str:
        """Get specialized system prompt for chitchat"""
        return """You are a friendly, engaging conversationalist. Your personality is:

        - Warm and approachable
        - Curious and interested in what others have to say
        - Knowledgeable but not pedantic
        - Empathetic and supportive
        - Occasionally humorous but always appropriate

        You can discuss a wide range of topics including:
        - Daily life and experiences
        - Hobbies and interests
        - Current events (with a balanced perspective)
        - Philosophy and ideas
        - Culture and arts
        - Personal growth and wellbeing

        Always maintain a positive, respectful tone while being genuine and authentic in your responses."""

    def get_state_schema(self) -> Type:
        """Get the state schema for this agent"""
        return BaseState

    def build_graph(self) -> StateGraph:
        """Build the chitchat agent graph"""
        workflow = StateGraph(
            BaseState,
            input=InputState,
            output=OutputState
        )

        # Add nodes
        workflow.add_node("analyze_mood", self.analyze_conversation_mood)
        workflow.add_node("generate_response", self.generate_conversational_response)
        workflow.add_node("add_personality", self.add_personality_touches)

        # Add edges
        workflow.add_edge(START, "analyze_mood")
        workflow.add_edge("analyze_mood", "generate_response")
        workflow.add_edge("generate_response", "add_personality")
        workflow.add_edge("add_personality", END)

        return workflow

    def analyze_conversation_mood(self, state: BaseState, config: RunnableConfig) -> Dict[str, Any]:
        """Analyze the mood and context of the conversation"""
        self.logger.info("Analyzing conversation mood")

        messages = state["messages"]
        last_message = messages[-1].content if messages else ""

        # Analyze mood and intent
        mood_prompt = """Analyze the emotional tone and context of this message:

        Message: {message}

        Consider:
        1. Emotional tone (happy, sad, curious, frustrated, etc.)
        2. Formality level (casual, formal, friendly)
        3. Topic category (personal, philosophical, practical, etc.)
        4. Engagement level (just starting, deep conversation, wrapping up)

        Provide a brief analysis."""

        mood_analysis = self.model.invoke([
            SystemMessage(content=mood_prompt.format(message=last_message))
        ])

        return {
            "metadata": {
                "mood_analysis": mood_analysis.content,
                "message_length": len(last_message),
                "conversation_depth": len(messages)
            }
        }

    def generate_conversational_response(self, state: BaseState, config: RunnableConfig) -> Dict[str, Any]:
        """Generate a conversational response"""
        self.logger.info("Generating conversational response")

        messages = state["messages"]
        mood_analysis = state.get("metadata", {}).get("mood_analysis", "")

        # Prepare conversation context
        recent_messages = messages[-5:] if len(messages) > 5 else messages

        # Generate response with context
        response_prompt = """Based on the conversation mood and context, provide an engaging response.

        Mood Analysis: {mood}

        Guidelines:
        - Match the user's energy and formality level
        - Be genuinely interested and ask follow-up questions when appropriate
        - Share relevant thoughts or experiences when it adds value
        - Keep responses concise but meaningful
        - If the user seems to need support, be empathetic
        - If they're playful, be playful back
        """

        # Combine system prompt with mood-aware instructions
        full_messages = [
                            SystemMessage(
                                content=self.system_prompt + "\n\n" + response_prompt.format(mood=mood_analysis))
                        ] + recent_messages

        response = self.model.invoke(full_messages)

        return {
            "metadata": {
                **state.get("metadata", {}),
                "base_response": response.content
            }
        }

    def add_personality_touches(self, state: BaseState, config: RunnableConfig) -> Dict[str, Any]:
        """Add personality touches to make the response more engaging"""
        self.logger.info("Adding personality touches")

        base_response = state.get("metadata", {}).get("base_response", "")
        messages = state["messages"]

        # Determine if we should add personality elements
        conversation_depth = len(messages)

        # For deeper conversations, occasionally add personal touches
        if conversation_depth > 2 and len(base_response) > 50:
            personality_prompt = """Review this response and enhance it with subtle personality touches if appropriate:

            Response: {response}

            Consider adding (only if natural):
            - A relevant emoji (sparingly)
            - A thoughtful question to continue the conversation
            - A brief, relevant observation
            - A touch of appropriate humor if the mood is light

            Keep the core message intact, just make it more engaging.
            If the response is already perfect, return it as is."""

            enhanced = self.model.invoke([
                SystemMessage(content=personality_prompt.format(response=base_response))
            ])

            final_response = enhanced.content
        else:
            final_response = base_response

        # Ensure response isn't too long for chitchat
        if len(final_response) > 500:
            # Summarize if too long
            summary_prompt = "Make this response more concise while keeping the key points: {response}"
            summarized = self.model.invoke([
                SystemMessage(content=summary_prompt.format(response=final_response))
            ])
            final_response = summarized.content

        return {
            "messages": [AIMessage(content=final_response)],
            "response": final_response,
            "metadata": {
                **state.get("metadata", {}),
                "enhanced": True
            }
        }


class PersonalityChitchatAgent(ChitchatAgent):
    """Extended chitchat agent with configurable personality"""

    def __init__(self, personality_type: str = "friendly", **kwargs):
        """
        Initialize with specific personality

        Args:
            personality_type: Type of personality (friendly, professional, quirky, etc.)
        """
        self.personality_type = personality_type
        super().__init__(**kwargs)

    def _get_chitchat_prompt(self) -> str:
        """Get personality-specific prompt"""
        personalities = {
            "friendly": """You are warm, encouraging, and genuinely interested in others.
                         You often use positive language and show enthusiasm.""",

            "professional": """You are courteous, knowledgeable, and maintain appropriate boundaries.
                            You communicate clearly and respectfully.""",

            "quirky": """You have a unique perspective and occasionally share unusual observations.
                       You're creative, playful, and enjoy wordplay.""",

            "wise": """You offer thoughtful insights and ask profound questions.
                    You speak with calm confidence and share wisdom when appropriate.""",

            "enthusiastic": """You're energetic, optimistic, and passionate about life.
                           You celebrate small wins and encourage others."""
        }

        base_prompt = personalities.get(self.personality_type, personalities["friendly"])

        return f"""{base_prompt}

        You engage in natural conversation while maintaining your distinct personality.
        You're helpful, genuine, and adapt to the conversation's flow while staying true to your character."""
