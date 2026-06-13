import os
import sqlite3
import json
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import uvicorn


load_dotenv()

# Initialisation
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    check_compatibility=False
)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "financial_data.db")
COLLECTION_NAME = "financial_reports"

app = FastAPI(title="Financial AI Agent")

# LLM
llm = ChatGroq(
    api_key=os.getenv("API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

#============================================
@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query on the financial database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        if not results:
            return "No results found."
        return json.dumps([dict(zip(columns, row)) for row in results])
    except Exception as e:
        return f"SQL Error: {str(e)}"

@tool
def searchreports(query: str) -> str:
    """Search financial reports for qualitative information."""
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


# AGENT

class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    token_cost: float

tools = [execute_sql, searchreports]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    cost = 0.0
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        input_tokens = response.usage_metadata.get('input_tokens', 0)
        output_tokens = response.usage_metadata.get('output_tokens', 0)
        cost = (input_tokens / 1_000_000) * 0.075 + (output_tokens / 1_000_000) * 0.30
    return {"messages": [response], "token_cost": state.get("token_cost", 0) + cost}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
agent_app = graph.compile()


# API ENDPOINTS

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    token_cost: float

@app.get("/")
def root():
    return {"status": "Financial AI Agent is running !"}

@app.post("/query", response_model=QueryResponse)

@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    question = request.question.lower()
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("API_KEY"))

    # Questions sur des chiffres → SQL via Groq directement
    if any(word in question for word in ["revenue", "income", "profit", "assets", "employees", "total", "compare", "how much"]):
        try:
            # Groq génère le SQL
            sql_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": """Generate ONLY a SQL query for SQLite. No explanation.
                        Table: financial_metrics
                        Columns: company, year, total_revenue, net_income, operating_income, gross_profit, total_assets, employees
                        ALWAYS include company and year in SELECT.
                        Return ONLY the SQL query, nothing else."""},
                    {"role": "user", "content": request.question}
                ]
            )

            sql_query = sql_response.choices[0].message.content.strip()
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            print(f"SQL GENERATED: {sql_query}") 
            # Exécute le SQL
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.close()

            data = [dict(zip(columns, row)) for row in results]

            # Groq formule la réponse
            final_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Answer the question clearly based on the data provided. Always mention the company name and year."},
                    {"role": "user", "content": f"Data: {json.dumps(data)}\n\nQuestion: {request.question}"}
                ]
            )
            answer = final_response.choices[0].message.content
            cost = 0.0

        except Exception as e:
            answer = f"Error: {str(e)}"
            cost = 0.0

    # Questions qualitatives → Qdrant direct
    else:
        try:
            query_vector = embedder.encode(request.question).tolist()
            results = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=3
            )
            if results.points:
                output = []
                for hit in results.points:
                    output.append(f"[{hit.payload['company']} {hit.payload['year']}] {hit.payload['text'][:400]}")
                context = "\n\n".join(output)

                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Answer based on the provided context from financial reports."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
                    ]
                )
                answer = response.choices[0].message.content
                cost = 0.0
            else:
                answer = "No relevant information found."
                cost = 0.0
        except Exception as e:
            answer = f"Error: {str(e)}"
            cost = 0.0

    print(f"Query: {request.question} | Cost: ${cost:.6f}")
    return QueryResponse(answer=answer, token_cost=cost)

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)