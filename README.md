# 🤖 LangGraph Agent System

A production-ready, scalable multi-agent system built with LangGraph, featuring specialized agents for different tasks with best coding practices.

## ✨ Features

- **🎯 Multipurpose Bot**: Intelligent routing to specialized sub-agents
  - 🧮 Math Agent: Complex calculations with step-by-step explanations
  - 💬 Chitchat Agent: Natural conversations with personality
  - 🎧 Headphones Agent: Expert knowledge with RAG from vector storage

- **🍕 Hungry Services**: Food search and recommendations
  - Online food search
  - Recipe discovery
  - Restaurant recommendations
  - Nutritional information

- **📚 Embedding Service**: Content storage and retrieval
  - Support for text, URLs, and files
  - Automatic chunking and indexing
  - Multiple collection management

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- API key for LLM (OpenAI/Antropic/Gemenai/xAI)
- Base URL if use Ollama or Proxy like litellm

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/shamspias/langgraph-agents.git
cd langgraph-agents
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

#### 🖥️ CLI Mode (Interactive Testing)

```bash
python main.py cli
```

Example CLI session:
```
🤖 LangGraph Agent System CLI
--------------------------------------------------
Available agents:
1. multipurpose - Math, chitchat, and headphones expert
2. hungry - Food search and recommendations
3. embedding - Store content in vector database

[multipurpose]> Calculate 234 * 567 / 89 + 23
💭 Processing...

🤖 Response:
Let me calculate: 234 * 567 / 89 + 23

Mathematical expression: 234*567/89+23

Step-by-step solution:
1. Perform multiplication/division from left to right
2. Perform addition/subtraction from left to right
3. Calculate: 234*567/89+23

**Result: 1513.898876404494**

The answer is 1513.8989

[multipurpose]> switch hungry
✓ Switched to hungry agent

[hungry]> Find me the best pizza recipes
💭 Processing...
```

#### 🌐 API Server Mode

```bash
python main.py
# Or
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

API will be available at `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📖 Usage Examples

### Python Client

```python
import httpx
import asyncio

async def chat_with_agent():
    async with httpx.AsyncClient() as client:
        # Multipurpose bot - Math calculation
        response = await client.post(
            "http://localhost:8000/chat",
            json={
                "message": "What is 45 * 78 - 234?",
                "agent_type": "multipurpose"
            }
        )
        print(response.json())
        
        # Hungry Services - Food search
        response = await client.post(
            "http://localhost:8000/chat",
            json={
                "message": "Find me healthy breakfast recipes",
                "agent_type": "hungry"
            }
        )
        print(response.json())
        
        # Embedding Service
        response = await client.post(
            "http://localhost:8000/embed",
            json={
                "content": "The Sony WH-1000XM5 are premium headphones with excellent noise cancellation.",
                "collection_name": "headphones_knowledge"
            }
        )
        print(response.json())

asyncio.run(chat_with_agent())
```

### cURL Examples

```bash
# Chat with multipurpose bot
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the best headphones for gaming?",
    "agent_type": "multipurpose"
  }'

# Food search
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find Italian restaurants near me",
    "agent_type": "hungry"
  }'

# Embed content
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "https://example.com/article-about-headphones",
    "collection_name": "headphones_knowledge"
  }'
```

### JavaScript/TypeScript

```javascript
// Using fetch API
async function chatWithAgent() {
    const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: 'Explain quantum computing',
            agent_type: 'multipurpose',
        }),
    });
    
    const data = await response.json();
    console.log(data.response);
}
```

## 🏗️ Architecture

### Design Patterns

1. **Abstract Base Class Pattern**: All agents inherit from `BaseAgent`
2. **Singleton Pattern**: `VectorStoreManager` ensures single instance
3. **State Pattern**: Each agent has typed state schemas
4. **Strategy Pattern**: Different agents for different strategies
5. **Factory Pattern**: `AgentManager` creates and manages agents

## 🎯 Agent Capabilities

### Multipurpose Bot
- **Intent Classification**: Automatically routes to appropriate sub-agent
- **Math Processing**: Handles complex calculations
- **Chitchat**: Natural conversation with personality
- **Headphones Expert**: RAG-based knowledge retrieval

### Hungry Services
- **Food Search**: Searches online for food information
- **Recipe Discovery**: Finds recipes with ingredients
- **Restaurant Finder**: Locates restaurants and delivery
- **Nutrition Info**: Provides nutritional data

### Embedding Service
- **Multi-format Support**: Text, URLs, PDFs
- **Auto-chunking**: Intelligent document splitting
- **Collection Management**: Multiple knowledge bases
- **Batch Processing**: Handle multiple documents

## 🛡️ Best Practices Implemented

1. **Type Safety**: Full type hints and Pydantic models
2. **Error Handling**: Comprehensive try-catch blocks
3. **Logging**: Structured logging throughout
4. **Async Support**: Full async/await implementation
5. **Memory Management**: Thread-based conversation memory
6. **Scalability**: Modular, extensible architecture
7. **Testing**: Unit test structure ready
8. **Documentation**: Comprehensive docstrings
9. **Security**: Input validation and sanitization
10. **Performance**: Efficient vector storage and retrieval

## 📊 Monitoring

The system includes built-in logging:

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.DEBUG)

# Access agent logs
logger = logging.getLogger("agent.MultipurposeBot")
```

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

Example test:
```python
import pytest
from agents.multipurpose_bot.agent import MultipurposeBot

@pytest.mark.asyncio
async def test_multipurpose_bot():
    bot = MultipurposeBot()
    bot.compile()
    
    result = await bot.ainvoke({
        "messages": [{"role": "user", "content": "Hello!"}]
    })
    
    assert "response" in result
    assert len(result["messages"]) > 0
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations

1. **Use environment variables** for all sensitive data
2. **Implement rate limiting** for API endpoints
3. **Add authentication** for production use
4. **Set up monitoring** with Prometheus/Grafana
5. **Use persistent storage** for vector database
6. **Implement caching** for frequently accessed data
7. **Set up load balancing** for multiple instances

## 🔄 Extending the System

### Adding a New Agent

1. Create agent directory:
```python
# agents/new_agent/agent.py
from core.base_agent import BaseAgent

class NewAgent(BaseAgent):
    def get_state_schema(self):
        return YourStateSchema
    
    def build_graph(self):
        # Build your graph
        pass
```

2. Register in `main.py`:
```python
self.agents["new_agent"] = NewAgent()
```

### Adding New Tools

```python
from langchain_core.tools import tool

@tool
def your_custom_tool(query: str) -> str:
    """Your tool description"""
    # Tool implementation
    return result
```

## 📝 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check |
| `/agents` | GET | List available agents |
| `/chat` | POST | Chat with an agent |
| `/embed` | POST | Embed content |
| `/batch-embed` | POST | Batch embed content |

### Response Format

```json
{
  "response": "Agent response text",
  "thread_id": "conversation-thread-id",
  "agent_type": "multipurpose",
  "metadata": {
    "handler": "math",
    "confidence": 0.95
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [LangChain](https://github.com/langchain-ai/langchain)

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.
