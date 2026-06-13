import os
import sqlite3
import json
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
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

# ============================================
# TOOLS
# ============================================
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

# ============================================
# API ENDPOINTS
# ============================================
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    token_cost: float

@app.get("/")
def root():
    return {"status": "Financial AI Agent is running !"}

@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    question = request.question.lower()
    groq_client = Groq(api_key=os.getenv("API_KEY"))
    #  Questions sur des chiffres → SQL via Groq directement
    if any(word in question for word in ["revenue", "income", "profit", "assets", "employees", "total", "compare", "how much"]):
        try:
            sql_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": """Generate ONLY a SQLite query. No explanation. No markdown.
                        Table: financial_metrics
                        Columns: company, year, total_revenue, net_income, operating_income, gross_profit, total_assets, employees
                        STRICT RULES:
                        - ALWAYS start with: SELECT company, year,
                        - ALWAYS add WHERE with company name and/or year
                        - Return ONLY raw SQL
                        Example: SELECT company, year, total_revenue FROM financial_metrics WHERE company='Amazon' AND year=2025"""},
                                            {"role": "user", "content": request.question}
                                        ],
                temperature=0
            )

            sql_query = sql_response.choices[0].message.content.strip()
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            print(f"SQL: {sql_query}")

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.close()

            data = [dict(zip(columns, row)) for row in results]
            print(f"DATA: {data}")

            final_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Answer clearly. Always mention company name and year from the data."},
                    {"role": "user", "content": f"Data: {json.dumps(data)}\n\nQuestion: {request.question}"}
                ],
                temperature=0
            )
            answer = final_response.choices[0].message.content
            cost = 0.0

        except Exception as e:
            answer = f"Error: {str(e)}"
            cost = 0.0
    # # Questions qualitatives → Qdrant direct
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
                    ],
                    temperature=0
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