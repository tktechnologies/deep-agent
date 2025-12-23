"""Deep Research Agent - Main Script.

This script implements a deep research agent using LangGraph that can:
- Perform iterative web searches using Tavily API
- Summarize and store search results in files
- Manage research tasks with TODO tracking
- Delegate work to specialized sub-agents
- Reflect on research progress strategically

The agent uses a ReAct (Reason + Act) pattern with multiple tools including
web search, file management, and task delegation capabilities.
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import os
import sys
import warnings
from datetime import datetime
import uuid
import base64

# Third-party packages
import httpx
from dotenv import load_dotenv
from IPython.display import Image, display
from markdownify import markdownify
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

# LangChain and LangGraph
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from tavily import TavilyClient

# Local imports
from deep_agents_from_scratch.file_tools import ls, read_file, write_file
from deep_agents_from_scratch.prompts import (
    FILE_USAGE_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    SUMMARIZE_WEB_SEARCH,
    TODO_USAGE_INSTRUCTIONS,
)
from deep_agents_from_scratch.state import DeepAgentState
from deep_agents_from_scratch.task_tool import _create_task_tool
from deep_agents_from_scratch.todo_tools import write_todos, read_todos
from utils import show_prompt, stream_agent, format_messages


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Add scripts folder to path to import utils
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables (.env)
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path, override=True)

# Suppress LangSmith warnings
warnings.filterwarnings("ignore", message="LangSmith now uses UUID v7", category=UserWarning)

# Configuration constants
SUMMARIZATION_MODEL_NAME = "openai:gpt-4o-mini"
MAIN_MODEL_NAME = "openai:gpt-4o-mini"
MAIN_MODEL_TEMPERATURE = 0.0
MAX_CONCURRENT_RESEARCH_UNITS = 3
MAX_RESEARCHER_ITERATIONS = 3
RECURSION_LIMIT = 15
HTTP_TIMEOUT = 30.0

# Initialize global models and clients
summarization_model = init_chat_model(model=SUMMARIZATION_MODEL_NAME)
tavily_client = TavilyClient()


# ============================================================================
# DATA CLASSES
# ============================================================================

class Summary(BaseModel):
    """Schema for webpage content summarization."""
    filename: str = Field(description="Name of the file to store.")
    summary: str = Field(description="Key learnings from the webpage.")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    print("📅 Running: get_today_str()")
    return datetime.now().strftime("%a %b %d, %Y")


def run_tavily_search(
    search_query: str,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> dict:
    """Perform search using Tavily API for a single query.

    Args:
        search_query: Search query to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        Search results dictionary
    """
    print(f"🔍 Running: run_tavily_search(query='{search_query}', max_results={max_results})")
    result = tavily_client.search(
        search_query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic
    )
    return result


def summarize_webpage_content(webpage_content: str) -> Summary:
    """Summarize webpage content using the configured summarization model.
    
    Args:
        webpage_content: Raw webpage content to summarize
        
    Returns:
        Summary object with filename and summary
    """
    print(f"📝 Running: summarize_webpage_content(content_length={len(webpage_content)})")
    try:
        structured_model = summarization_model.with_structured_output(Summary)
        summary_and_filename = structured_model.invoke([
            HumanMessage(content=SUMMARIZE_WEB_SEARCH.format(
                webpage_content=webpage_content,
                date=get_today_str()
            ))
        ])
        return summary_and_filename
    except Exception:
        # Return basic summary on failure
        return Summary(
            filename="search_result.md",
            summary=webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content
        )


def process_search_results(results: dict) -> list[dict]:
    """Process search results by summarizing content where available.

    Args:
        results: Tavily search results dictionary

    Returns:
        List of processed results with summaries
    """
    print(f"⚙️ Running: process_search_results(num_results={len(results.get('results', []))})")
    processed_results = []
    httpx_client = httpx.Client(timeout=HTTP_TIMEOUT)

    for result in results.get('results', []):
        url = result['url']

        try:
            response = httpx_client.get(url)
        
            if response.status_code == 200:
                raw_content = markdownify(response.text)
                summary_obj = summarize_webpage_content(raw_content)
            else:
                raw_content = result.get('raw_content', '')
                summary_obj = Summary(
                    filename="URL_error.md",
                    summary=result.get('content', 'Error reading URL; try another search.')
                )
        except (httpx.TimeoutException, httpx.RequestError):
            raw_content = result.get('raw_content', '')
            summary_obj = Summary(
                filename="connection_error.md",
                summary=result.get('content', 'Could not fetch URL (timeout/connection error). Try another search.')
            )

        # Generate unique filename
        uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
        name, ext = os.path.splitext(summary_obj.filename)
        summary_obj.filename = f"{name}_{uid}{ext}"

        processed_results.append({
            'url': result['url'],
            'title': result['title'],
            'summary': summary_obj.summary,
            'filename': summary_obj.filename,
            'raw_content': raw_content,
        })

    return processed_results


# ============================================================================
# AGENT TOOLS
# ============================================================================

@tool(parse_docstring=True)
def tavily_search(
    query: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> Command:
    """Search web and save detailed results to files while returning minimal context.

    Performs web search and saves full content to files for context offloading.
    Returns only essential information to help the agent decide on next steps.

    Args:
        query: Search query to execute
        state: Injected agent state for file storage
        tool_call_id: Injected tool call identifier
        max_results: Maximum number of results to return (default: 1)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Command that saves full results to files and provides minimal summary
    """
    print(f"🔧 Running TOOL: tavily_search(query='{query}')")
    
    search_results = run_tavily_search(query, max_results=max_results, topic=topic, include_raw_content=True)
    processed_results = process_search_results(search_results)
    
    files = state.get("files", {})
    saved_files = []
    summaries = []
    
    for result in processed_results:
        filename = result['filename']
        
        file_content = f"""# Search Result: {result['title']}  

**URL:** {result['url']}  
**Query:** {query}  
**Date:** {get_today_str()}  

## Summary
{result['summary']}  
## Raw Content
{result['raw_content'] if result['raw_content'] else 'No raw content available'}  
"""
        
        files[filename] = file_content
        saved_files.append(filename)
        summaries.append(f"- {filename}: {result['summary']}...")
    
    summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':  

{chr(10).join(summaries)}  

Files: {', '.join(saved_files)}  
💡 Use read_file() to access full details when needed.""" 

    return Command(
        update={
            "files": files,
            "messages": [ToolMessage(summary_text, tool_call_id=tool_call_id)],
        }
    )


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?
    - How complex is the question: Have I reached the number of search limits?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    print(f"💭 Running TOOL: think_tool(reflection='{reflection[:50]}...')")
    return f"Reflection recorded: {reflection}"


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

# Initialize main model
model = init_chat_model(model=MAIN_MODEL_NAME, temperature=MAIN_MODEL_TEMPERATURE)

# Define available tools
sub_agent_tools = [tavily_search, think_tool]
built_in_tools = [ls, read_file, write_file, write_todos, read_todos]

# Configure research sub-agent
research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    "tools": ["tavily_search", "think_tool"],
}

# Create task delegation tool
task_tool = _create_task_tool(sub_agent_tools, [research_sub_agent], model, DeepAgentState)
delegation_tools = [task_tool]
all_tools = sub_agent_tools + built_in_tools + delegation_tools

# Build system instructions
SUBAGENT_INSTRUCTIONS = SUBAGENT_USAGE_INSTRUCTIONS.format(
    max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
    max_researcher_iterations=MAX_RESEARCHER_ITERATIONS,
    date=datetime.now().strftime("%a %b %d, %Y"),
)

INSTRUCTIONS = (
    "# TODO MANAGEMENT\n"
    + TODO_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# FILE SYSTEM USAGE\n"
    + FILE_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# SUB-AGENT DELEGATION\n"
    + SUBAGENT_INSTRUCTIONS
)

# Create main agent
agent = create_agent(
    model,
    all_tools,
    system_prompt=INSTRUCTIONS,
    state_schema=DeepAgentState,
)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_agent(question: str, verbose: bool = False) -> dict:
    """Run the agent programmatically (for API usage).
    
    Args:
        question: User question to process
        verbose: Whether to print execution logs
        
    Returns:
        Dictionary with answer and metadata
    """
    if verbose:
        print(f"🤖 Processing question: {question}")
    
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
        },
        config={"recursion_limit": RECURSION_LIMIT}
    )
    
    # Extract final answer
    messages = result.get("messages", [])
    final_message = messages[-1] if messages else None
    
    # Get answer text
    if final_message:
        if hasattr(final_message, 'content'):
            answer = final_message.content
        else:
            answer = str(final_message)
    else:
        answer = "No response generated"
    
    return {
        "answer": answer,
        "question": question,
        "message_count": len(messages),
        "files": result.get("files", {}),
        "todos": result.get("todos", [])
    }


def main():
    """Execute the deep research agent (CLI mode)."""
    print("\n" + "="*80)
    print("🚀 STARTING AGENT EXECUTION")
    print("="*80 + "\n")

    user_question = input("💬 Enter your question: ")
    print()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_question,
                }
            ],
        },
        config={"recursion_limit": RECURSION_LIMIT}
    )

    print("\n" + "="*80)
    print("✅ EXECUTION COMPLETED")
    print("="*80 + "\n")

    format_messages(result["messages"])


if __name__ == "__main__":
    main()