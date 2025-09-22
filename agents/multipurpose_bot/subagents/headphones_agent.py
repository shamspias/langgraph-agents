"""
Headphones Agent with RAG from vector storage
"""
from typing import Dict, Any, Type, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from core.base_agent import BaseAgent
from core.state import RAGState, InputState, OutputState
from storage.vector_store import VectorStoreManager


class HeadphonesAgent(BaseAgent):
    """Agent specialized in headphones knowledge using RAG"""

    def __init__(self, **kwargs):
        """Initialize Headphones Agent"""
        super().__init__(
            name="HeadphonesAgent",
            system_prompt=self._get_headphones_prompt(),
            **kwargs
        )
        self.vector_store = VectorStoreManager()
        self.collection_name = "headphones_knowledge"
        self._initialize_knowledge_base()

    def _get_headphones_prompt(self) -> str:
        """Get specialized system prompt for headphones"""
        return """You are an expert audio engineer and headphones specialist. Your expertise includes:

        1. Headphone types: Over-ear, on-ear, in-ear, true wireless, bone conduction
        2. Technical specifications: Drivers, impedance, sensitivity, frequency response
        3. Sound signatures: Balanced, V-shaped, bass-heavy, bright, warm
        4. Use cases: Gaming, music production, casual listening, sports, travel
        5. Brands and models: From budget to high-end audiophile options

        Always provide accurate, detailed information based on the knowledge base.
        Give personalized recommendations based on user needs and preferences.
        Explain technical terms in an accessible way when needed."""

    def _initialize_knowledge_base(self):
        """Initialize headphones knowledge base if not exists"""
        # Check if collection exists
        collection = self.vector_store.get_or_create_collection(self.collection_name)

        # Add initial knowledge if collection is empty
        # In production, this would load from a comprehensive database
        initial_knowledge = [
            """Sony WH-1000XM5: Premium noise-canceling headphones with excellent sound quality. 
            30-hour battery life, LDAC support, multipoint connectivity. Price: $399. 
            Best for: Travel, commuting, office use. Sound signature: Balanced with slight bass emphasis.""",

            """Sennheiser HD600: Open-back reference headphones beloved by audiophiles. 
            300-ohm impedance requires amplification. Neutral sound signature perfect for critical listening. 
            Price: $399. Best for: Home listening, mixing, mastering.""",

            """Apple AirPods Pro 2: Premium true wireless earbuds with H2 chip. 
            Active noise cancellation, transparency mode, spatial audio. 6-hour battery life. 
            Price: $249. Best for: Apple ecosystem users, daily commuting.""",

            """Audio-Technica ATH-M50x: Professional studio monitor headphones. 
            Closed-back design, collapsible, detachable cables. V-shaped sound signature. 
            Price: $149. Best for: Studio monitoring, DJing, casual listening.""",

            """Beyerdynamic DT 770 Pro: Closed-back studio headphones available in 32, 80, and 250 ohm. 
            Excellent comfort for long sessions. Bright, detailed sound. Price: $179. 
            Best for: Studio work, gaming, home listening.""",

            """Headphone Impedance Guide: Low impedance (16-32 ohm) works with phones and portable devices. 
            Medium impedance (32-100 ohm) benefits from portable amps. High impedance (250+ ohm) requires 
            dedicated amplification for optimal performance.""",

            """Open-back vs Closed-back: Open-back headphones provide wider soundstage and natural sound 
            but leak sound and offer no isolation. Closed-back headphones provide isolation and stronger 
            bass but narrower soundstage. Semi-open combines elements of both.""",

            """Gaming Headphones: Key features include positional audio, comfortable for long sessions, 
            good microphone quality. Popular options: SteelSeries Arctis Pro, HyperX Cloud II, 
            Astro A50. Virtual 7.1 surround can help with competitive gaming.""",

            """Budget Recommendations Under $100: Koss Porta Pro ($39) - retro style, lifetime warranty. 
            Samson SR850 ($49) - semi-open, great soundstage. Philips SHP9500 ($79) - open-back value king. 
            KZ ZSN Pro X ($25) - budget IEM champion.""",

            """Wireless Codecs: SBC - universal but compressed. AAC - good for Apple devices. 
            aptX - better quality for Android. LDAC - Sony's high-res codec. aptX Adaptive - 
            Qualcomm's variable bitrate codec for low latency and quality."""
        ]

        # Only add if collection is empty (first time initialization)
        try:
            test_search = collection.similarity_search("test", k=1)
            if not test_search:
                self.logger.info("Initializing headphones knowledge base...")
                self.vector_store.add_texts(
                    texts=initial_knowledge,
                    collection_name=self.collection_name,
                    metadatas=[{"source": "internal_knowledge"} for _ in initial_knowledge]
                )
        except Exception as e:
            self.logger.info(f"Knowledge base initialization check: {str(e)}")

    def get_state_schema(self) -> Type:
        """Get the state schema for this agent"""
        return RAGState

    def build_graph(self) -> StateGraph:
        """Build the headphones agent graph"""
        workflow = StateGraph(
            RAGState,
            input=InputState,
            output=OutputState
        )

        # Add nodes
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("rank_documents", self.rank_documents)
        workflow.add_node("generate_answer", self.generate_answer)

        # Add edges
        workflow.add_edge(START, "retrieve_documents")
        workflow.add_edge("retrieve_documents", "rank_documents")
        workflow.add_edge("rank_documents", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow

    def retrieve_documents(self, state: RAGState, config: RunnableConfig) -> Dict[str, Any]:
        """Retrieve relevant documents from vector store"""
        self.logger.info("Retrieving headphones information")

        messages = state["messages"]
        query = messages[-1].content if messages else ""

        try:
            # Search for relevant documents
            documents = self.vector_store.similarity_search_with_score(
                query=query,
                k=5,
                collection_name=self.collection_name
            )

            # Convert to document format
            formatted_docs = []
            for doc, score in documents:
                formatted_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score)
                })

            self.logger.info(f"Retrieved {len(formatted_docs)} documents")

            return {
                "query": query,
                "documents": formatted_docs,
                "metadata": {
                    "retrieval_count": len(formatted_docs)
                }
            }
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            return {
                "query": query,
                "documents": [],
                "error": f"Retrieval error: {str(e)}"
            }

    def rank_documents(self, state: RAGState, config: RunnableConfig) -> Dict[str, Any]:
        """Rank and filter retrieved documents"""
        self.logger.info("Ranking documents by relevance")

        documents = state.get("documents", [])
        query = state.get("query", "")

        if not documents:
            return state

        # Sort by relevance score
        ranked_docs = sorted(
            documents,
            key=lambda x: x.get("relevance_score", 0),
            reverse=False  # Lower scores are better for distance metrics
        )

        # Filter out low-relevance documents (threshold can be adjusted)
        threshold = 1.5  # Adjust based on your embedding model
        filtered_docs = [
                            doc for doc in ranked_docs
                            if doc.get("relevance_score", float('inf')) < threshold
                        ][:3]  # Keep top 3 most relevant

        self.logger.info(f"Filtered to {len(filtered_docs)} highly relevant documents")

        return {
            "documents": filtered_docs,
            "metadata": {
                **state.get("metadata", {}),
                "filtered_count": len(filtered_docs)
            }
        }

    def generate_answer(self, state: RAGState, config: RunnableConfig) -> Dict[str, Any]:
        """Generate answer based on retrieved documents"""
        self.logger.info("Generating headphones expert answer")

        query = state.get("query", "")
        documents = state.get("documents", [])

        # Prepare context from documents
        context = self._prepare_context(documents)

        # Generate answer
        if context:
            answer_prompt = """Based on the following knowledge base information, provide a comprehensive answer to the user's question.

            Question: {query}

            Context:
            {context}

            Instructions:
            1. Provide accurate information from the context
            2. If recommending products, explain why they fit the user's needs
            3. Include relevant technical details when appropriate
            4. If the context doesn't fully answer the question, acknowledge this
            5. Format the response clearly with sections if needed"""

            response = self.model.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=answer_prompt.format(
                    query=query,
                    context=context
                ))
            ])

            answer = response.content
            sources = [doc.get("metadata", {}).get("source", "knowledge_base") for doc in documents]
        else:
            # No relevant documents found
            answer = self._generate_general_response(query)
            sources = []

        return {
            "messages": [AIMessage(content=answer)],
            "response": answer,
            "answer": answer,
            "sources": sources,
            "metadata": {
                **state.get("metadata", {}),
                "answer_generated": True
            }
        }

    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents"""
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            score = doc.get("relevance_score", "N/A")
            context_parts.append(f"[Document {i} - Relevance: {score:.2f}]\n{content}")

        return "\n\n".join(context_parts)

    def _generate_general_response(self, query: str) -> str:
        """Generate a general response when no documents are found"""
        general_prompt = """The user is asking about headphones, but I don't have specific information in my knowledge base.
        Provide a helpful general response that:
        1. Acknowledges the question
        2. Offers general guidance based on audio expertise
        3. Suggests what information would be helpful to provide better recommendations

        Question: {query}"""

        response = self.model.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=general_prompt.format(query=query))
        ])

        return response.content
