"""
Utility functions for the agent system
"""
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from functools import wraps
import time
import asyncio


# Logging utilities
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with consistent formatting

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Message utilities
def format_messages(messages: List[BaseMessage], max_length: int = 1000) -> str:
    """
    Format messages for display

    Args:
        messages: List of messages
        max_length: Maximum length per message

    Returns:
        Formatted string
    """
    formatted = []
    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "")
        content = msg.content
        if len(content) > max_length:
            content = content[:max_length] + "..."
        formatted.append(f"[{role}]: {content}")

    return "\n".join(formatted)


def extract_last_user_message(messages: List[BaseMessage]) -> Optional[str]:
    """
    Extract the last user message from a list of messages

    Args:
        messages: List of messages

    Returns:
        Last user message content or None
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count (rough estimate)

    Args:
        text: Text to count tokens for

    Returns:
        Approximate token count
    """
    # Rough approximation: 1 token ≈ 4 characters
    return len(text) // 4


# String utilities
def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', '', text)
    return text.strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to specified length

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text

    Args:
        text: Text to extract URLs from

    Returns:
        List of URLs
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


# Hash utilities
def generate_hash(content: str) -> str:
    """
    Generate SHA256 hash of content

    Args:
        content: Content to hash

    Returns:
        Hash string
    """
    return hashlib.sha256(content.encode()).hexdigest()


def generate_thread_id(prefix: str = "") -> str:
    """
    Generate a unique thread ID

    Args:
        prefix: Optional prefix for the ID

    Returns:
        Thread ID
    """
    import uuid
    thread_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{thread_id}"
    return thread_id


# JSON utilities
def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON with fallback

    Args:
        json_str: JSON string
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def pretty_json(obj: Any) -> str:
    """
    Pretty print JSON object

    Args:
        obj: Object to format as JSON

    Returns:
        Pretty formatted JSON string
    """
    return json.dumps(obj, indent=2, default=str)


# File utilities
def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_file_safe(file_path: Union[str, Path]) -> Optional[str]:
    """
    Safely read file content

    Args:
        file_path: Path to file

    Returns:
        File content or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None


def write_file_safe(file_path: Union[str, Path], content: str) -> bool:
    """
    Safely write content to file

    Args:
        file_path: Path to file
        content: Content to write

    Returns:
        Success status
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logging.error(f"Error writing file {file_path}: {e}")
        return False


# Decorators
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator with exponential backoff

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logging.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logging.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            return None

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator


def timeit(func):
    """
    Decorator to measure function execution time
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper


def cache_result(ttl_seconds: int = 300):
    """
    Simple cache decorator with TTL

    Args:
        ttl_seconds: Time to live in seconds
    """
    cache = {}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str((args, tuple(sorted(kwargs.items()))))

            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl_seconds:
                    return result

            # Execute function
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result

        return wrapper

    return decorator


# Validation utilities
def validate_thread_id(thread_id: str) -> bool:
    """
    Validate thread ID format

    Args:
        thread_id: Thread ID to validate

    Returns:
        Validation status
    """
    # Basic UUID validation
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    # Also allow prefixed UUIDs
    prefixed_pattern = r'^[\w]+_[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'

    return bool(re.match(uuid_pattern, thread_id) or re.match(prefixed_pattern, thread_id))


def validate_collection_name(name: str) -> bool:
    """
    Validate collection name

    Args:
        name: Collection name

    Returns:
        Validation status
    """
    # Allow alphanumeric, underscore, and hyphen
    pattern = r'^[\w\-]+$'
    return bool(re.match(pattern, name)) and len(name) <= 63


# Conversion utilities
def messages_to_dict(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """
    Convert messages to dictionary format

    Args:
        messages: List of messages

    Returns:
        List of dictionaries
    """
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            role = "unknown"

        result.append({
            "role": role,
            "content": msg.content,
            "additional_kwargs": msg.additional_kwargs if hasattr(msg, 'additional_kwargs') else {}
        })

    return result


def dict_to_messages(message_dicts: List[Dict[str, Any]]) -> List[BaseMessage]:
    """
    Convert dictionary format to messages

    Args:
        message_dicts: List of message dictionaries

    Returns:
        List of messages
    """
    messages = []
    for msg_dict in message_dicts:
        role = msg_dict.get("role", "user")
        content = msg_dict.get("content", "")

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))

    return messages


# Date/Time utilities
def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse ISO format timestamp"""
    try:
        return datetime.fromisoformat(timestamp_str)
    except (ValueError, AttributeError):
        return None


def time_ago(timestamp: Union[str, datetime]) -> str:
    """
    Get human-readable time ago

    Args:
        timestamp: Timestamp string or datetime object

    Returns:
        Human-readable time difference
    """
    if isinstance(timestamp, str):
        timestamp = parse_timestamp(timestamp)

    if not timestamp:
        return "unknown time"

    now = datetime.now()
    diff = now - timestamp

    if diff < timedelta(minutes=1):
        return "just now"
    elif diff < timedelta(hours=1):
        minutes = int(diff.total_seconds() / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif diff < timedelta(days=1):
        hours = int(diff.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = diff.days
        return f"{days} day{'s' if days != 1 else ''} ago"


# Export all utilities
__all__ = [
    'setup_logger',
    'format_messages',
    'extract_last_user_message',
    'count_tokens_approximate',
    'clean_text',
    'truncate_text',
    'extract_urls',
    'generate_hash',
    'generate_thread_id',
    'safe_json_loads',
    'pretty_json',
    'ensure_directory',
    'read_file_safe',
    'write_file_safe',
    'retry',
    'timeit',
    'cache_result',
    'validate_thread_id',
    'validate_collection_name',
    'messages_to_dict',
    'dict_to_messages',
    'get_timestamp',
    'parse_timestamp',
    'time_ago'
]
