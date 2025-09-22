"""
Vector Store manager for RAG operations
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from config.settings import settings


class VectorStoreManager:
    """Manages vector store operations with best practices"""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern for vector store"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize vector store manager"""
        if not self._initialized:
            self.logger = self._setup_logger()
            self.embeddings = self._get_embeddings()
            self.text_splitter = self._get_text_splitter()
            self.stores: Dict[str, Chroma] = {}
            self._initialized = True
            self.logger.info("Vector store manager initialized")

    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger("storage.vector_store")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _get_embeddings(self) -> OpenAIEmbeddings:
        """Get embedding model"""
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )

    def _get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get text splitter"""
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def get_or_create_collection(
            self,
            collection_name: str = None,
            persist_directory: Path = None
    ) -> Chroma:
        """
        Get or create a vector store collection

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the collection

        Returns:
            Chroma vector store instance
        """
        collection_name = collection_name or settings.COLLECTION_NAME
        persist_directory = persist_directory or settings.VECTOR_DB_PATH

        store_key = f"{persist_directory}/{collection_name}"

        if store_key not in self.stores:
            self.logger.info(f"Creating/loading collection: {collection_name}")
            self.stores[store_key] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(persist_directory)
            )

        return self.stores[store_key]

    def add_texts(
            self,
            texts: List[str],
            metadatas: Optional[List[Dict[str, Any]]] = None,
            collection_name: str = None
    ) -> List[str]:
        """
        Add texts to vector store

        Args:
            texts: List of texts to add
            metadatas: Optional metadata for texts
            collection_name: Collection to add to

        Returns:
            List of document IDs
        """
        collection = self.get_or_create_collection(collection_name)

        # Split texts into chunks
        documents = []
        for i, text in enumerate(texts):
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                metadata = metadatas[i] if metadatas else {}
                documents.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))

        self.logger.info(f"Adding {len(documents)} documents to {collection_name or settings.COLLECTION_NAME}")
        ids = collection.add_documents(documents)
        return ids

    def add_documents(
            self,
            documents: List[Document],
            collection_name: str = None
    ) -> List[str]:
        """
        Add documents to vector store

        Args:
            documents: List of documents to add
            collection_name: Collection to add to

        Returns:
            List of document IDs
        """
        collection = self.get_or_create_collection(collection_name)

        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(documents)

        self.logger.info(f"Adding {len(split_docs)} documents to {collection_name or settings.COLLECTION_NAME}")
        ids = collection.add_documents(split_docs)
        return ids

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            collection_name: str = None,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents

        Args:
            query: Query string
            k: Number of results to return
            collection_name: Collection to search in
            filter: Optional metadata filter

        Returns:
            List of similar documents
        """
        collection = self.get_or_create_collection(collection_name)

        self.logger.info(f"Searching for '{query}' in {collection_name or settings.COLLECTION_NAME}")

        if filter:
            results = collection.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
        else:
            results = collection.similarity_search(
                query=query,
                k=k
            )

        self.logger.info(f"Found {len(results)} results")
        return results

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            collection_name: str = None,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with scores

        Args:
            query: Query string
            k: Number of results to return
            collection_name: Collection to search in
            filter: Optional metadata filter

        Returns:
            List of (document, score) tuples
        """
        collection = self.get_or_create_collection(collection_name)

        self.logger.info(f"Searching with scores for '{query}' in {collection_name or settings.COLLECTION_NAME}")

        if filter:
            results = collection.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
        else:
            results = collection.similarity_search_with_score(
                query=query,
                k=k
            )

        self.logger.info(f"Found {len(results)} results with scores")
        return results

    def get_retriever(
            self,
            collection_name: str = None,
            k: int = 4,
            filter: Optional[Dict[str, Any]] = None
    ):
        """
        Get a retriever for the collection

        Args:
            collection_name: Collection name
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            Retriever instance
        """
        collection = self.get_or_create_collection(collection_name)

        return collection.as_retriever(
            search_kwargs={
                "k": k,
                **({"filter": filter} if filter else {})
            }
        )

    def delete_collection(self, collection_name: str = None) -> None:
        """Delete a collection"""
        collection_name = collection_name or settings.COLLECTION_NAME
        persist_directory = settings.VECTOR_DB_PATH
        store_key = f"{persist_directory}/{collection_name}"

        if store_key in self.stores:
            del self.stores[store_key]

        self.logger.info(f"Deleted collection: {collection_name}")

    def load_from_urls(
            self,
            urls: List[str],
            collection_name: str = None
    ) -> List[str]:
        """
        Load documents from URLs

        Args:
            urls: List of URLs to load
            collection_name: Collection to add to

        Returns:
            List of document IDs
        """
        documents = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                documents.extend(docs)
                self.logger.info(f"Loaded {len(docs)} documents from {url}")
            except Exception as e:
                self.logger.error(f"Error loading {url}: {str(e)}")

        if documents:
            return self.add_documents(documents, collection_name)
        return []

    def load_from_files(
            self,
            file_paths: List[Path],
            collection_name: str = None
    ) -> List[str]:
        """
        Load documents from files

        Args:
            file_paths: List of file paths to load
            collection_name: Collection to add to

        Returns:
            List of document IDs
        """
        documents = []
        for file_path in file_paths:
            try:
                loader = TextLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                self.logger.info(f"Loaded {len(docs)} documents from {file_path}")
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {str(e)}")

        if documents:
            return self.add_documents(documents, collection_name)
        return []
