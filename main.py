"""
Main application for LangGraph Agent System
"""
import asyncio
import uuid
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import agents
from agents.hungry_services.agent import HungryServicesAgent
from agents.multipurpose_bot.agent import MultipurposeBot
from agents.embedding_service.agent import EmbeddingServiceAgent, BatchEmbeddingAgent
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message")
    agent_type: str = Field("multipurpose", description="Agent type: multipurpose, hungry, embedding")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")
    collection_name: Optional[str] = Field(None, description="Collection name for embedding agent")


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="Agent response")
    thread_id: str = Field(..., description="Thread ID for conversation continuity")
    agent_type: str = Field(..., description="Agent that processed the request")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class EmbeddingRequest(BaseModel):
    """Embedding request model"""
    content: str = Field(..., description="Content to embed")
    collection_name: str = Field("general_knowledge", description="Collection name")
    content_type: Optional[str] = Field(None, description="Content type: text, url, file")


class BatchEmbeddingRequest(BaseModel):
    """Batch embedding request model"""
    items: list[str] = Field(..., description="List of items to embed")
    collection_name: str = Field("general_knowledge", description="Collection name")
    item_type: str = Field("text", description="Type of items: text, url, file")


# Agent Manager
class AgentManager:
    """Manages all agents in the system"""

    def __init__(self):
        """Initialize agent manager"""
        self.logger = logging.getLogger("agent_manager")
        self.agents = {}
        self._initialized = False

    async def initialize(self):
        """Initialize all agents"""
        if self._initialized:
            return

        try:
            self.logger.info("Initializing agents...")

            # Initialize agents
            self.agents["multipurpose"] = MultipurposeBot()
            self.agents["hungry"] = HungryServicesAgent()
            self.agents["embedding"] = EmbeddingServiceAgent()
            self.agents["batch_embedding"] = BatchEmbeddingAgent()

            # Compile all agents
            for name, agent in self.agents.items():
                self.logger.info(f"Compiling {name} agent...")
                agent.compile()

            self._initialized = True
            self.logger.info("All agents initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing agents: {str(e)}")
            raise

    def get_agent(self, agent_type: str):
        """Get agent by type"""
        return self.agents.get(agent_type)

    async def process_message(
            self,
            message: str,
            agent_type: str = "multipurpose",
            thread_id: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Process a message with the specified agent

        Args:
            message: User message
            agent_type: Type of agent to use
            thread_id: Thread ID for conversation continuity
            **kwargs: Additional arguments for the agent

        Returns:
            Agent response
        """
        agent = self.get_agent(agent_type)
        if not agent:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Generate thread ID if not provided
        if not thread_id:
            thread_id = str(uuid.uuid4())

        # Prepare input
        input_data = {
            "messages": [HumanMessage(content=message)]
        }

        # Add any additional kwargs
        input_data.update(kwargs)

        # Process with agent
        result = await agent.ainvoke(
            input_data,
            thread_id=thread_id
        )

        return {
            "result": result,
            "thread_id": thread_id,
            "agent_type": agent_type
        }


# Initialize agent manager
agent_manager = AgentManager()


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting application...")
    await agent_manager.initialize()
    yield
    # Shutdown
    logger.info("Shutting down application...")


app = FastAPI(
    title="LangGraph Agent System",
    description="Scalable multi-agent system with specialized capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LangGraph Agent System API",
        "agents": list(agent_manager.agents.keys()),
        "endpoints": {
            "/chat": "Chat with agents",
            "/embed": "Embed content",
            "/batch-embed": "Batch embed content",
            "/agents": "List available agents",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agents_initialized": agent_manager._initialized,
        "agents_count": len(agent_manager.agents)
    }


@app.get("/agents")
async def list_agents():
    """List available agents"""
    agents_info = {}
    for name, agent in agent_manager.agents.items():
        agents_info[name] = {
            "name": agent.name,
            "memory_enabled": agent.memory_enabled,
            "compiled": agent.compiled_graph is not None
        }
    return {"agents": agents_info}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with an agent

    Args:
        request: Chat request with message and agent type

    Returns:
        Agent response
    """
    try:
        result = await agent_manager.process_message(
            message=request.message,
            agent_type=request.agent_type,
            thread_id=request.thread_id,
            collection_name=request.collection_name
        )

        # Extract response
        agent_result = result["result"]
        response_text = agent_result.get("response", "")

        # If no response field, try to get from messages
        if not response_text and "messages" in agent_result:
            last_message = agent_result["messages"][-1]
            response_text = last_message.content

        return ChatResponse(
            response=response_text,
            thread_id=result["thread_id"],
            agent_type=result["agent_type"],
            metadata=agent_result.get("metadata")
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed")
async def embed_content(request: EmbeddingRequest):
    """
    Embed content into vector storage

    Args:
        request: Embedding request with content and collection name

    Returns:
        Embedding result
    """
    try:
        result = await agent_manager.process_message(
            message=request.content,
            agent_type="embedding",
            collection_name=request.collection_name
        )

        agent_result = result["result"]

        return {
            "success": agent_result.get("embeddings_stored", False),
            "collection_name": request.collection_name,
            "metadata": agent_result.get("metadata"),
            "message": agent_result.get("response", "")
        }

    except Exception as e:
        logger.error(f"Error in embed endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-embed")
async def batch_embed_content(
        request: BatchEmbeddingRequest,
        background_tasks: BackgroundTasks
):
    """
    Batch embed content into vector storage

    Args:
        request: Batch embedding request
        background_tasks: FastAPI background tasks

    Returns:
        Batch embedding result
    """
    try:
        batch_agent = agent_manager.get_agent("batch_embedding")

        if request.item_type == "url":
            result = batch_agent.embed_urls(
                urls=request.items,
                collection_name=request.collection_name
            )
        elif request.item_type == "file":
            file_paths = [Path(item) for item in request.items]
            result = batch_agent.embed_files(
                file_paths=file_paths,
                collection_name=request.collection_name
            )
        else:
            # Text items - process individually
            results = []
            for item in request.items:
                try:
                    agent_result = await agent_manager.process_message(
                        message=item,
                        agent_type="embedding",
                        collection_name=request.collection_name
                    )
                    results.append({
                        "item": item[:100],  # First 100 chars
                        "success": True
                    })
                except Exception as e:
                    results.append({
                        "item": item[:100],
                        "success": False,
                        "error": str(e)
                    })
            result = {"batch_results": results}

        return {
            "success": True,
            "collection_name": request.collection_name,
            "items_processed": len(request.items),
            "results": result.get("batch_results", [])
        }

    except Exception as e:
        logger.error(f"Error in batch embed endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# CLI Interface
async def cli_interface():
    """Command-line interface for testing agents"""
    print("\n🤖 LangGraph Agent System CLI")
    print("-" * 50)
    print("Available agents:")
    print("1. multipurpose - Math, chitchat, and headphones expert")
    print("2. hungry - Food search and recommendations")
    print("3. embedding - Store content in vector database")
    print("\nType 'exit' to quit, 'switch <agent>' to change agents")
    print("-" * 50)

    await agent_manager.initialize()

    current_agent = "multipurpose"
    thread_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input(f"\n[{current_agent}]> ").strip()

            if user_input.lower() == "exit":
                print("Goodbye! 👋")
                break

            if user_input.lower().startswith("switch "):
                new_agent = user_input[7:].strip()
                if new_agent in agent_manager.agents:
                    current_agent = new_agent
                    thread_id = str(uuid.uuid4())  # New thread for new agent
                    print(f"✓ Switched to {current_agent} agent")
                else:
                    print(f"✗ Unknown agent: {new_agent}")
                continue

            if user_input.lower() == "new":
                thread_id = str(uuid.uuid4())
                print("✓ Started new conversation")
                continue

            # Process message
            print("\n💭 Processing...")
            result = await agent_manager.process_message(
                message=user_input,
                agent_type=current_agent,
                thread_id=thread_id
            )

            # Display response
            agent_result = result["result"]
            response = agent_result.get("response", "")

            if not response and "messages" in agent_result:
                response = agent_result["messages"][-1].content

            print("\n🤖 Response:")
            print(response)

            # Show metadata if available
            if "metadata" in agent_result and agent_result["metadata"]:
                print("\n📊 Metadata:", agent_result["metadata"])

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit.")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


# Main entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Run CLI interface
        asyncio.run(cli_interface())
    else:
        # Run FastAPI server
        import uvicorn

        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
