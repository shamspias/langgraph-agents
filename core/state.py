"""
State definitions for agent workflows
"""
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel, Field


class BaseState(TypedDict):
    """Base state for all agents"""
    messages: Annotated[List[AnyMessage], add_messages]
    error: Optional[str]
    metadata: Dict[str, Any]


class FoodSearchState(BaseState):
    """State for food search agent"""
    query: str
    search_results: Optional[List[Dict[str, Any]]]
    food_info: Optional[Dict[str, Any]]
    recommendations: Optional[List[str]]


class MathCalculationState(BaseState):
    """State for math calculation"""
    expression: str
    formula: Optional[str]
    result: Optional[float]
    step_by_step: Optional[List[str]]


class RAGState(BaseState):
    """State for RAG operations"""
    query: str
    documents: Optional[List[Dict[str, Any]]]
    answer: Optional[str]
    sources: Optional[List[str]]


class MultipurposeState(BaseState):
    """State for multipurpose bot"""
    query: str
    intent: Optional[str]  # math, chitchat, headphones, unknown
    sub_state: Optional[Dict[str, Any]]
    response: Optional[str]


class EmbeddingState(BaseState):
    """State for embedding service"""
    content: str
    content_type: str  # text, document, url
    chunks: Optional[List[str]]
    embeddings_stored: bool = False
    collection_name: str


# Input/Output State Schemas
class InputState(TypedDict):
    """Input state schema"""
    messages: Annotated[List[AnyMessage], add_messages]


class OutputState(TypedDict):
    """Output state schema"""
    messages: Annotated[List[AnyMessage], add_messages]
    response: str
    metadata: Optional[Dict[str, Any]]


# Intent Classification Schema
class IntentClassification(BaseModel):
    """Schema for intent classification"""
    intent: str = Field(
        description="The classified intent: math, chitchat, headphones, food, or unknown"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation for the classification"
    )


# Math Formula Schema
class MathFormula(BaseModel):
    """Schema for math formula extraction"""
    has_math: bool = Field(description="Whether the query contains math")
    formula: Optional[str] = Field(description="Extracted mathematical formula")
    variables: Optional[Dict[str, float]] = Field(
        description="Variables and their values if any"
    )
