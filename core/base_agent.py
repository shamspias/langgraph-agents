"""
Base Agent class with common functionality
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
import logging
from config.settings import settings


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(
            self,
            name: str,
            model: Optional[ChatOpenAI] = None,
            system_prompt: Optional[str] = None,
            enable_memory: bool = True,
            enable_tracing: bool = None
    ):
        """
        Initialize base agent

        Args:
            name: Agent name for identification
            model: LLM model instance
            system_prompt: System prompt for the agent
            enable_memory: Whether to enable conversation memory
            enable_tracing: Whether to enable LangSmith tracing
        """
        self.name = name
        self.logger = self._setup_logger()

        # Initialize model
        self.model = model or self._get_default_model()

        # System prompt
        self.system_prompt = system_prompt or self._get_default_prompt()

        # Memory and storage
        self.memory_enabled = enable_memory
        if enable_memory:
            self.checkpointer = MemorySaver()
            self.store = InMemoryStore()
        else:
            self.checkpointer = None
            self.store = None

        # Tracing
        self.tracing_enabled = (
            enable_tracing if enable_tracing is not None
            else settings.LANGSMITH_TRACING
        )

        # Build the graph
        self.graph = None
        self.compiled_graph = None

        self.logger.info(f"Initialized agent: {name}")

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the agent"""
        logger = logging.getLogger(f"agent.{self.name}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _get_default_model(self) -> ChatOpenAI:
        """Get default LLM model"""
        return ChatOpenAI(
            model=settings.DEFAULT_MODEL,
            temperature=settings.TEMPERATURE,
            api_key=settings.OPENAI_API_KEY
        )

    def _get_default_prompt(self) -> str:
        """Get default system prompt"""
        return f"You are {self.name}, a helpful AI assistant."

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        Build the agent's graph structure
        Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def get_state_schema(self) -> Type:
        """
        Get the state schema for this agent
        Must be implemented by subclasses
        """
        pass

    def compile(self) -> Any:
        """Compile the agent's graph"""
        if self.graph is None:
            self.graph = self.build_graph()

        compile_kwargs = {
            "checkpointer": self.checkpointer,
        }

        if self.store:
            compile_kwargs["store"] = self.store

        self.compiled_graph = self.graph.compile(**compile_kwargs)
        self.logger.info(f"Compiled graph for {self.name}")
        return self.compiled_graph

    def invoke(
            self,
            inputs: Dict[str, Any],
            config: Optional[RunnableConfig] = None,
            thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Invoke the agent

        Args:
            inputs: Input dictionary
            config: Optional runnable configuration
            thread_id: Thread ID for conversation memory

        Returns:
            Agent response
        """
        if self.compiled_graph is None:
            self.compile()

        # Prepare config
        if config is None:
            config = {}

        if thread_id and self.memory_enabled:
            config["configurable"] = {"thread_id": thread_id}

        try:
            self.logger.info(f"Invoking {self.name} with inputs: {inputs}")
            result = self.compiled_graph.invoke(inputs, config=config)
            self.logger.info(f"Invocation successful for {self.name}")
            return result
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}")
            raise

    async def ainvoke(
            self,
            inputs: Dict[str, Any],
            config: Optional[RunnableConfig] = None,
            thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Async invoke the agent

        Args:
            inputs: Input dictionary
            config: Optional runnable configuration
            thread_id: Thread ID for conversation memory

        Returns:
            Agent response
        """
        if self.compiled_graph is None:
            self.compile()

        # Prepare config
        if config is None:
            config = {}

        if thread_id and self.memory_enabled:
            config["configurable"] = {"thread_id": thread_id}

        try:
            self.logger.info(f"Async invoking {self.name} with inputs: {inputs}")
            result = await self.compiled_graph.ainvoke(inputs, config=config)
            self.logger.info(f"Async invocation successful for {self.name}")
            return result
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}")
            raise

    def stream(
            self,
            inputs: Dict[str, Any],
            config: Optional[RunnableConfig] = None,
            thread_id: Optional[str] = None
    ):
        """
        Stream agent responses

        Args:
            inputs: Input dictionary
            config: Optional runnable configuration
            thread_id: Thread ID for conversation memory

        Yields:
            Streaming responses
        """
        if self.compiled_graph is None:
            self.compile()

        # Prepare config
        if config is None:
            config = {}

        if thread_id and self.memory_enabled:
            config["configurable"] = {"thread_id": thread_id}

        try:
            self.logger.info(f"Streaming from {self.name}")
            for chunk in self.compiled_graph.stream(inputs, config=config):
                yield chunk
        except Exception as e:
            self.logger.error(f"Error in streaming from {self.name}: {str(e)}")
            raise

    def get_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get current state for a thread"""
        if not self.memory_enabled:
            return None

        config = {"configurable": {"thread_id": thread_id}}
        return self.compiled_graph.get_state(config)

    def update_state(
            self,
            thread_id: str,
            state_updates: Dict[str, Any]
    ) -> None:
        """Update state for a thread"""
        if not self.memory_enabled:
            return

        config = {"configurable": {"thread_id": thread_id}}
        self.compiled_graph.update_state(config, state_updates)
