# 🧠 Deep Research Agent

An intelligent research agent built with **LangGraph** and **LangChain** that performs deep, iterative web research to answer complex questions. The agent uses a ReAct (Reason + Act) pattern with strategic thinking, web search, file management, and task delegation capabilities.

## 🎯 What It Does

This agent doesn't just search once and return results. It:

1. **Plans Research Strategy** - Creates TODO lists to break down complex questions
2. **Iterative Search** - Performs multiple web searches, refining based on findings
3. **Strategic Reflection** - Uses a "think" tool to analyze progress and decide next steps
4. **Context Management** - Saves findings to files to avoid context overflow
5. **Parallel Research** - Delegates subtasks to specialized sub-agents
6. **Comprehensive Answers** - Synthesizes all findings into detailed responses

**Example:** Ask "What are the latest developments in quantum computing?" and watch it:
- Plan research TODOs
- Search for recent breakthroughs
- Reflect on what's missing
- Search specific subtopics (hardware, algorithms, applications)
- Save detailed findings to files
- Generate comprehensive answer from all sources

---

## 📁 Project Architecture

```
deep-agents-from-scratch/
├── scripts/                          # Execution layer
│   ├── deep_agent.py                # Main agent orchestration
│   ├── api.py                       # FastAPI REST server
│   └── utils.py                     # Display utilities
│
├── src/deep_agents_from_scratch/    # Core components
│   ├── __init__.py
│   ├── state.py                     # Agent state schema (DeepAgentState)
│   ├── prompts.py                   # System instructions for agent
│   ├── file_tools.py                # Virtual filesystem (ls, read, write)
│   ├── todo_tools.py                # Task tracking (create/update TODOs)
│   └── task_tool.py                 # Sub-agent delegation system
│
├── .env                              # API keys (not in git)
├── pyproject.toml                    # Dependencies
└── README.md                         # This file
```

---

## 🧩 Core Components

### **1. Agent State (`state.py`)**
Manages agent memory across tool calls:
- `messages` - Conversation history
- `files` - Virtual filesystem for storing research
- `todos` - Task tracking list
- Agent metadata

### **2. Tools System**

#### **Search Tools**
- **`tavily_search`** - Web search with auto-summarization
  - Fetches web content
  - Summarizes using GPT-4o-mini
  - Saves to files with unique IDs
  - Returns concise summary to agent

#### **Cognitive Tools**  
- **`think_tool`** - Strategic reflection
  - Forces deliberate pause for analysis
  - Agent evaluates progress and gaps
  - Decides whether to continue or conclude

#### **File Management Tools**
- **`ls()`** - List all files in virtual filesystem
- **`read_file(path, offset, limit)`** - Read files with pagination
- **`write_file(path, content)`** - Create/overwrite files

#### **Task Management Tools**
- **`write_todos(todos)`** - Create/update TODO list
- **`read_todos()`** - Read current TODOs
- Tracks: pending, in_progress, completed

#### **Delegation Tool**
- **`research_task()`** - Delegate to sub-agent
  - Spawns specialized researcher
  - Max 3 concurrent sub-agents
  - Each limited to 3 iterations

### **3. Prompts (`prompts.py`)**
System instructions that guide agent behavior:
- Research methodology
- TODO usage patterns
- File management workflow
- Sub-agent delegation rules
- Strategic thinking guidelines

### **4. Execution Modes**

#### **CLI Mode** (`deep_agent.py`)
```python
python deep_agent.py
# Prompts for question interactively
```

#### **API Mode** (`api.py`)
```python
# FastAPI server with REST endpoints
uvicorn api:app --reload
# POST /api/query with JSON
```

---

## 🚀 Quick Start

### **Prerequisites**
- Python 3.11+
- OpenAI API key (for GPT-4o-mini)
- Tavily API key (for web search)

### **1. Installation**

```powershell
# Clone repository
git clone <your-repo-url>
cd deep-agents-from-scratch

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
# Or with uv:
uv sync
```

### **2. Configuration**

Create `.env` file in project root:

```env
# Required
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# Optional (for debugging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
```

### **3. Run CLI Mode**

```powershell
cd scripts
python deep_agent.py
```

**Example interaction:**
```
💬 Enter your question: What are the health benefits of intermittent fasting?

[Agent creates TODOs, searches, thinks, writes files, generates answer]
```

### **4. Run API Mode**

```powershell
cd scripts
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Test with cURL:**
```powershell
curl -X POST "http://localhost:8000/api/query" `
  -H "Content-Type: application/json" `
  -d '{"question":"What is quantum entanglement?","verbose":true}'
```

**Or use Swagger UI:**
- Open http://localhost:8000/docs
- Try the `/api/query` endpoint interactively

---

## 🔍 How It Works

### **Agent Workflow**

```
User Question
    ↓
┌─────────────────────────────────────┐
│  1. CREATE RESEARCH PLAN (TODOs)   │
│     - Break down complex questions  │
│     - Identify research areas       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. ITERATIVE RESEARCH LOOP         │
│     ┌─────────────────────────┐    │
│     │ Web Search (Tavily)     │    │
│     │ ↓                       │    │
│     │ Summarize & Save        │    │
│     │ ↓                       │    │
│     │ Strategic Reflection    │    │
│     │ ↓                       │    │
│     │ Decide: More? Or Done?  │    │
│     └─────────────────────────┘    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. SYNTHESIZE ANSWER               │
│     - Read saved files              │
│     - Combine findings              │
│     - Generate comprehensive answer │
└─────────────────────────────────────┘
    ↓
Final Answer
```

### **ReAct Pattern in Action**

The agent uses **Reason + Act** cycles:

1. **Reason** (Think Tool):
   ```
   "I found information about quantum entanglement basics.
    Still need: practical applications and recent research.
    Next: Search for 'quantum entanglement applications 2024'"
   ```

2. **Act** (Search Tool):
   ```
   Search → Summarize → Save to file
   ```

3. **Repeat** until satisfied

### **Virtual Filesystem**

Agent maintains context without overloading memory:

```python
files = {
    "quantum_basics_a3f8.md": "Content about quantum principles...",
    "applications_2024_b7e2.md": "Recent quantum computing uses...",
    "research_papers_c9d1.md": "Latest studies on entanglement..."
}
```

When answering, agent reads relevant files and synthesizes.

### **Sub-Agent Delegation**

For complex multi-topic questions:

```
Main Agent
  ├─ Sub-Agent 1: "Research quantum hardware"
  ├─ Sub-Agent 2: "Research quantum algorithms"  
  └─ Sub-Agent 3: "Research commercial applications"

Each sub-agent:
  - Has 3 iterations max
  - Uses tavily_search + think_tool
  - Returns findings to main agent
```

---

## 📚 API Reference

### **Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/api/query` | Ask the agent |
| GET | `/docs` | Swagger UI (interactive) |
| GET | `/redoc` | ReDoc documentation |

### **POST /api/query**

**Request Body:**
```json
{
  "question": "Your question here",
  "verbose": false  // Optional: enable debug logs
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "answer": "Detailed answer from the agent...",
  "question": "Your original question",
  "message_count": 15,
  "timestamp": "2025-12-23T10:30:00",
  "files_created": 3
}
```

**Error Response (500):**
```json
{
  "detail": "Agent execution failed: error message"
}
```

---

## 🎨 Frontend Integration

### **React Example**

```javascript
async function askAgent(question) {
  const response = await fetch('http://localhost:8000/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, verbose: false })
  });
  
  const data = await response.json();
  return data.answer;
}

// Usage
const answer = await askAgent("What is machine learning?");
console.log(answer);
```

### **Vanilla JavaScript**

```javascript
document.getElementById('askBtn').addEventListener('click', async () => {
  const question = document.getElementById('question').value;
  
  const response = await fetch('http://localhost:8000/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  
  const data = await response.json();
  document.getElementById('answer').textContent = data.answer;
});
```

---

## 🛠️ Configuration

### **Environment Variables**

Required in `.env`:

```env
# OpenAI (for LLM reasoning)
OPENAI_API_KEY=sk-proj-...

# Tavily (for web search)
TAVILY_API_KEY=tvly-...

# Optional: LangSmith (for debugging agent traces)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
```

### **Agent Configuration** (`deep_agent.py`)

```python
# Model configuration
SUMMARIZATION_MODEL_NAME = "openai:gpt-4o-mini"
MAIN_MODEL_NAME = "openai:gpt-4o-mini"
MAIN_MODEL_TEMPERATURE = 0.0

# Research limits
MAX_CONCURRENT_RESEARCH_UNITS = 3  # Max sub-agents
MAX_RESEARCHER_ITERATIONS = 3       # Max loops per sub-agent
RECURSION_LIMIT = 15                # Max main agent iterations

# Search configuration
HTTP_TIMEOUT = 30.0                 # HTTP request timeout
```

---

## 📦 Dependencies

Core packages (see `pyproject.toml`):

```toml
[dependencies]
# Web Framework
fastapi = "^0.115.12"
uvicorn = "^0.34.0"

# LLM & Agent Framework
langchain = "^0.3.19"
langgraph = "^0.2.67"
langchain-openai = "^0.2.14"

# Web Search
tavily-python = "^0.5.0"

# Utilities
pydantic = "^2.12.4"
httpx = "^0.28.1"
markdownify = "^1.2.2"
python-dotenv = "^1.0.1"
```

---

## 🐛 Troubleshooting

### **API won't start**
```powershell
# Check if port 8000 is occupied
netstat -ano | findstr :8000

# Use different port
uvicorn api:app --reload --port 3000
```

### **Import errors**
- Verify you're in `scripts/` directory when running
- Check virtual environment is activated
- Run `pip install -r requirements.txt` again

### **Agent not responding**
- Verify `.env` has valid API keys
- Check internet connection (required for Tavily)
- Enable verbose mode: `{"verbose": true}`
- Check terminal logs for error details

### **CORS errors in browser**
- Already configured to accept all origins in development
- For production, update `allow_origins` in `api.py`

### **Out of context errors**
- Agent automatically manages context with files
- If still occurring, reduce `MAX_RESEARCHER_ITERATIONS`

---

## 🧪 Example Use Cases

### **1. Research Questions**
```
"What are the environmental impacts of lithium mining for batteries?"
```
Agent will:
- Search general lithium mining info
- Search specific environmental impacts
- Search battery production context
- Search alternatives/solutions
- Synthesize comprehensive answer

### **2. Technical Comparisons**
```
"Compare React vs Vue.js for building enterprise applications"
```
Agent will:
- Research React features and ecosystem
- Research Vue.js features and ecosystem
- Search for enterprise use cases
- Compare performance, scalability, community
- Provide structured comparison

### **3. Recent Developments**
```
"What happened in AI regulation in 2024?"
```
Agent will:
- Search AI regulation news 2024
- Search specific countries/regions
- Search major policy changes
- Compile chronological summary

---

## 🔐 Security Notes

- **Never commit `.env`** - Already in `.gitignore`
- **API Keys** - Store securely, rotate regularly
- **CORS** - In production, specify exact allowed origins
- **Rate Limiting** - Consider adding to API for production
- **Input Validation** - Already handled by Pydantic models

---

## 📈 Future Enhancements

Potential additions:
- [ ] Streaming responses (SSE/WebSocket)
- [ ] Session management for conversation history
- [ ] User authentication and API keys
- [ ] Caching frequently asked questions
- [ ] Database integration for persistent storage
- [ ] Advanced error recovery and retry logic
- [ ] Custom search domain filtering
- [ ] Multi-language support

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit Pull Request

---

## 💬 Support

- **Issues**: Open an issue on GitHub
- **Documentation**: See code comments in each module
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **LangChain Docs**: https://python.langchain.com/

---

## 🎉 Ready to Research!

Start the agent and ask complex questions. Watch it:
- 🧠 Plan research strategy
- 🔍 Search multiple sources
- 💭 Think strategically
- 📝 Manage context efficiently
- ✨ Generate comprehensive answers

**CLI Mode:**
```powershell
cd scripts
python deep_agent.py
```

**API Mode:**
```powershell
cd scripts
uvicorn api:app --reload
# Open http://localhost:8000/docs
```

Happy researching! 🚀
