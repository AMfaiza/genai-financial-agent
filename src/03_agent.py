import os
import sqlite3
import json
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import instructor
from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI

load_dotenv()

# Initialisation des clients
qdrant = QdrantClient(url="http://localhost:6333")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
DB_PATH = "../data/financial_data.db"
COLLECTION_NAME = "financial_reports"

# LLM Groq
llm = ChatGroq(
    api_key=os.getenv("API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Pydantic pour extraire les filtres
class SearchIntent(BaseModel):
    company_filter: Optional[str] = Field(description="Company name if mentioned")
    year_filter: Optional[int] = Field(description="Year if mentioned")
    semantic_query: str = Field(description="Core question for vector search")

# Client instructor pour extraire les filtres
instructor_client = instructor.from_openai(
    OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    ),
    mode=instructor.Mode.JSON
)


#   SQL

@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query on the financial database and return results.
    The database has a table 'financial_metrics' with columns:
    company, year, total_revenue, net_income, operating_income, gross_profit, total_assets, employees
    All financial values are in millions USD."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()

        if not results:
            return "No results found."

        formatted = []
        for row in results:
            formatted.append(dict(zip(columns, row)))
        return json.dumps(formatted, indent=2)

    except Exception as e:
        return f"SQL Error: {str(e)}. Please fix the query and try again."


# xsearch Vector DB

@tool
def searchreports(query: str) -> str:
    """Search financial reports for qualitative information about strategy risks and business outlook."""
    try:
        query_vector = embedder.encode(query).tolist()
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        )
        if not results.points:
            return "No relevant information found."
        output = []
        for hit in results.points:
            output.append(f"[{hit.payload['company']} {hit.payload['year']}] {hit.payload['text'][:300]}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search Error: {str(e)}"

# Agent State

class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    token_cost: float


# Agent Graph
# ============================================
tools = [execute_sql, searchreports]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """Le cerveau de l'agent"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    # Calcul du coût en tokens
    cost = 0.0
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        input_tokens = response.usage_metadata.get('input_tokens', 0)
        output_tokens = response.usage_metadata.get('output_tokens', 0)
        cost = (input_tokens / 1_000_000) * 0.075 + (output_tokens / 1_000_000) * 0.30

    return {
        "messages": [response],
        "token_cost": state.get("token_cost", 0) + cost
    }

def should_continue(state: AgentState):
    """Décide si l'agent doit continuer ou s'arrêter"""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# Construction du graph
tool_node = ToolNode(tools)
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
app = graph.compile()


# Test de l'agent

def run_agent(question: str):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print('='*60)

    result = app.invoke({
        "messages": [HumanMessage(content=question)],
        "token_cost": 0.0
    })

    final_answer = result["messages"][-1].content
    total_cost = result.get("token_cost", 0)

    print(f"\nANSWER: {final_answer}")
    print(f"\nTotal Token Cost: ${total_cost:.6f}")
    return final_answer

if __name__ == "__main__":
    # Test 
    run_agent("What was Amazon's total revenue in 2025?")
    run_agent("What did Apple's report say about supply chain risks?")
    run_agent("Compare Microsoft and Google revenue in 2024")