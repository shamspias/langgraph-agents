"""
Embedding Service Agent for adding content to vector storage
"""
from typing import Dict, Any, Type, List
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from core.base_agent import BaseAgent
from core.state import EmbeddingState, InputState, OutputState
from storage.vector_store import VectorStoreManager
import re


class EmbeddingServiceAgent(BaseAgent):
    """Agent for embedding and storing content in vector storage"""

    def __init__(self, **kwargs):
        """Initialize Embedding Service Agent"""
        super().__init__(
            name="EmbeddingService",
            system_prompt=self._get_embedding_prompt(),
            **kwargs
        )
        self.vector_store = VectorStoreManager()

    def _get_embedding_prompt(self) -> str:
        """Get system prompt for embedding service"""
        return """You are an intelligent content processing assistant that helps store and organize information.
        Your role is to:
        1. Analyze content to understand its type and structure
        2. Extract relevant metadata
        3. Chunk content appropriately for vector storage
        4. Ensure content is properly indexed for retrieval

        You handle various content types including text, documents, and web pages."""

    def get_state_schema(self) -> Type:
        """Get the state schema for this agent"""
        return EmbeddingState

    def build_graph(self) -> StateGraph:
        """Build the embedding service graph"""
        workflow = StateGraph(
            EmbeddingState,
            input=InputState,
            output=OutputState
        )

        # Add nodes
        workflow.add_node("analyze_content", self.analyze_content)
        workflow.add_node("process_content", self.process_content)
        workflow.add_node("chunk_content", self.chunk_content)
        workflow.add_node("store_embeddings", self.store_embeddings)
        workflow.add_node("confirm_storage", self.confirm_storage)

        # Add edges
        workflow.add_edge(START, "analyze_content")
        workflow.add_edge("analyze_content", "process_content")
        workflow.add_edge("process_content", "chunk_content")
        workflow.add_edge("chunk_content", "store_embeddings")
        workflow.add_edge("store_embeddings", "confirm_storage")
        workflow.add_edge("confirm_storage", END)

        return workflow

    def analyze_content(self, state: EmbeddingState, config: RunnableConfig) -> Dict[str, Any]:
        """Analyze the content to determine type and processing strategy"""
        self.logger.info("Analyzing content for embedding")

        messages = state["messages"]
        content = messages[-1].content if messages else ""

        # Determine content type
        content_type = self._detect_content_type(content)

        # Determine collection name if not specified
        collection_name = state.get("collection_name", "general_knowledge")

        # Extract metadata
        metadata = self._extract_metadata(content, content_type)

        return {
            "content": content,
            "content_type": content_type,
            "collection_name": collection_name,
            "metadata": metadata
        }

    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content"""
        # Check if it's a URL
        url_pattern = r'https?://[^\s]+'
        if re.match(url_pattern, content):
            return "url"

        # Check if it's a file path
        if Path(content).exists():
            return "file"

        # Check if it's about headphones (for specialized storage)
        headphones_keywords = [
            "headphone", "earphone", "earbud", "audio", "driver",
            "impedance", "frequency response", "soundstage", "bass",
            "treble", "midrange", "ANC", "noise canceling"
        ]
        if any(keyword in content.lower() for keyword in headphones_keywords):
            return "headphones_content"

        # Default to text
        return "text"

    def _extract_metadata(self, content: str, content_type: str) -> Dict[str, Any]:
        """Extract metadata from content"""
        metadata = {
            "content_type": content_type,
            "length": len(content),
        }

        # Extract title if present
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1)

        # Add source information
        if content_type == "url":
            metadata["source"] = content.strip()
        elif content_type == "file":
            metadata["source"] = Path(content).name
        else:
            metadata["source"] = "direct_input"

        return metadata

    def process_content(self, state: EmbeddingState, config: RunnableConfig) -> Dict[str, Any]:
        """Process content based on its type"""
        self.logger.info(f"Processing {state['content_type']} content")

        content = state["content"]
        content_type = state["content_type"]

        try:
            if content_type == "url":
                # Load content from URL
                loader = WebBaseLoader(content.strip())
                documents = loader.load()
                processed_content = "\n\n".join([doc.page_content for doc in documents])

            elif content_type == "file":
                # Load content from file
                file_path = Path(content.strip())
                if file_path.suffix == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                else:
                    loader = TextLoader(str(file_path))
                documents = loader.load()
                processed_content = "\n\n".join([doc.page_content for doc in documents])

            elif content_type == "headphones_content":
                # Special processing for headphones content
                processed_content = self._enhance_headphones_content(content)
                # Update collection name for headphones content
                state["collection_name"] = "headphones_knowledge"

            else:
                # Direct text content
                processed_content = content

            return {
                "content": processed_content,
                "metadata": {
                    **state.get("metadata", {}),
                    "processed": True
                }
            }

        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            return {
                "error": f"Processing error: {str(e)}",
                "content": content
            }

    def _enhance_headphones_content(self, content: str) -> str:
        """Enhance headphones-related content with structure"""
        # Add structure to headphones content for better retrieval
        enhanced = f"=== Headphones Knowledge Entry ===\n\n{content}\n\n"

        # Extract and highlight key specifications if present
        specs_pattern = r'(\d+)\s*(ohm|Hz|dB|mm|hours?)'
        specs = re.findall(specs_pattern, content, re.IGNORECASE)
        if specs:
            enhanced += "\nKey Specifications:\n"
            for value, unit in specs:
                enhanced += f"- {value} {unit}\n"

        return enhanced

    def chunk_content(self, state: EmbeddingState, config: RunnableConfig) -> Dict[str, Any]:
        """Chunk content for optimal vector storage"""
        self.logger.info("Chunking content for embedding")

        content = state.get("content", "")

        if not content:
            return {
                "chunks": [],
                "error": "No content to chunk"
            }

        # Use the vector store's text splitter
        chunks = self.vector_store.text_splitter.split_text(content)

        self.logger.info(f"Split content into {len(chunks)} chunks")

        return {
            "chunks": chunks,
            "metadata": {
                **state.get("metadata", {}),
                "chunk_count": len(chunks)
            }
        }

    def store_embeddings(self, state: EmbeddingState, config: RunnableConfig) -> Dict[str, Any]:
        """Store embeddings in vector database"""
        self.logger.info(f"Storing embeddings in {state['collection_name']}")

        chunks = state.get("chunks", [])
        collection_name = state.get("collection_name", "general_knowledge")
        metadata = state.get("metadata", {})

        if not chunks:
            return {
                "embeddings_stored": False,
                "error": "No chunks to store"
            }

        try:
            # Prepare metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                chunk_meta = {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                chunk_metadata.append(chunk_meta)

            # Store in vector database
            ids = self.vector_store.add_texts(
                texts=chunks,
                metadatas=chunk_metadata,
                collection_name=collection_name
            )

            self.logger.info(f"Successfully stored {len(ids)} embeddings")

            return {
                "embeddings_stored": True,
                "metadata": {
                    **metadata,
                    "stored_ids": ids,
                    "chunks_stored": len(ids)
                }
            }

        except Exception as e:
            self.logger.error(f"Error storing embeddings: {str(e)}")
            return {
                "embeddings_stored": False,
                "error": f"Storage error: {str(e)}"
            }

    def confirm_storage(self, state: EmbeddingState, config: RunnableConfig) -> Dict[str, Any]:
        """Confirm successful storage and provide summary"""
        self.logger.info("Confirming embedding storage")

        embeddings_stored = state.get("embeddings_stored", False)
        metadata = state.get("metadata", {})
        collection_name = state.get("collection_name", "general_knowledge")

        if embeddings_stored:
            chunks_count = metadata.get("chunks_stored", 0)
            content_type = state.get("content_type", "unknown")

            response = f"""✅ Successfully embedded and stored content!

**Storage Details:**
- Collection: {collection_name}
- Content Type: {content_type}
- Chunks Created: {chunks_count}
- Source: {metadata.get('source', 'direct input')}

The content has been indexed and is now searchable in the vector database.
"""

            if collection_name == "headphones_knowledge":
                response += "\n📎 This content has been added to the specialized headphones knowledge base."
        else:
            error = state.get("error", "Unknown error")
            response = f"""❌ Failed to store content.

**Error:** {error}

Please check the content and try again. Make sure the content is properly formatted and accessible."""

        return {
            "messages": [AIMessage(content=response)],
            "response": response,
            "metadata": metadata
        }


class BatchEmbeddingAgent(EmbeddingServiceAgent):
    """Extended agent for batch embedding operations"""

    def embed_urls(self, urls: List[str], collection_name: str = "web_content") -> Dict[str, Any]:
        """Embed multiple URLs"""
        results = []
        for url in urls:
            try:
                result = self.invoke(
                    {"messages": [HumanMessage(content=url)]},
                    thread_id=f"batch_url_{hash(url)}"
                )
                results.append({"url": url, "success": True, "result": result})
            except Exception as e:
                results.append({"url": url, "success": False, "error": str(e)})

        return {"batch_results": results}

    def embed_files(self, file_paths: List[Path], collection_name: str = "documents") -> Dict[str, Any]:
        """Embed multiple files"""
        results = []
        for file_path in file_paths:
            try:
                result = self.invoke(
                    {"messages": [HumanMessage(content=str(file_path))]},
                    thread_id=f"batch_file_{hash(str(file_path))}"
                )
                results.append({"file": str(file_path), "success": True, "result": result})
            except Exception as e:
                results.append({"file": str(file_path), "success": False, "error": str(e)})

        return {"batch_results": results}
