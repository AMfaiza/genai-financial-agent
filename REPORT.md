# Multi-Modal Enterprise Agent — Project Report

## What I Built

I built an AI agent that can answer complex business questions about major tech companies (Apple, Amazon, Google, Microsoft, Tesla) by combining two data sources:
- **Structured data** (financial numbers) stored in a SQLite database
- **Unstructured data** (annual report text) stored in a Qdrant vector database

For example, instead of manually searching through 300-page PDFs, you can just ask:
> "What did Apple say about supply chain risks in 2025?"
> "Compare Microsoft and Google net income in 2024"


## System Architecture

User Question

↓

FastAPI REST Endpoint (port 8000)

↓

Question Router (keyword detection)

↓                    ↓

Is it a number?      Is it qualitative?

↓                    ↓

Groq generates SQL   Qdrant semantic search

↓                    ↓

SQLite executes it   Returns relevant chunks

↓                    ↓

Groq formats answer  Groq summarizes answer

↓                    ↓

JSON Response


## Phase 1 — Vector ETL Pipeline

I downloaded 10 annual reports (10-K) from 5 companies over 2 years (2024-2025).

**The pipeline does 4 things:**
1. Reads each PDF/DOCX and extracts the raw text
2. Cleans the text (removes special characters, double spaces)
3. Splits the text into chunks of 500 words with 50-word overlap
4. Embeds each chunk using `all-MiniLM-L6-v2` and stores it in Qdrant with metadata

**Result:** 1,289 chunks inserted into Qdrant, each tagged with `company` and `year`.

**Why semantic chunking matters:** If a user asks about "supply chain disruptions", the system finds relevant passages even if they use different words like "logistics constraints" or "manufacturing delays".


## Phase 2 — Agentic State Machine

I used LangGraph to build a ReAct agent with two tools:

**Tool 1 — execute_sql:**
Handles quantitative questions. The agent writes a SQL query on the fly and executes it against the `financial_metrics` table.

**Tool 2 — searchreports:**
Handles qualitative questions. Uses cosine similarity to find the most relevant chunks in Qdrant.

**Error recovery:** SQL errors are caught and returned as readable messages instead of crashing the pipeline.


## Phase 3 — Cloud Deployment

The agent is packaged as a FastAPI REST API and containerized with Docker.

**API Endpoints:**
- `GET /` — health check
- `POST /query` — send a question, get an answer + token cost
- `GET /health` — service status

**FinOps tracking:** Every query logs the token cost in the terminal. Based on Groq pricing, the average cost per query is ~$0.00012, meaning 100 queries cost approximately $0.012.


## Phase 4 — RAGAS Evaluation

I tested the agent with 5 queries and manually evaluated the outputs:

| Query | Tool Used | Answer Quality | Faithful? | Relevant? |
|-------|-----------|---------------|-----------|-----------|
| Amazon revenue 2025 | SQL | $716,924M | Yes | Yes |
| Compare Google vs Microsoft net income 2024 | SQL | Google $100,118M > Microsoft $88,136M |  Yes |  Yes |
| Apple supply chain risks | Qdrant | Detailed risk factors listed | Yes | Yes |
| Tesla employees 2024 | SQL | 125,665 employees |  Yes |  Yes |
| Amazon AWS growth strategy | Qdrant | Relevant passages from 10-K |  Yes |  Yes |

**Faithfulness score: 5/5** — No hallucinations observed. All answers come directly from the source data.

**Relevance score: 5/5** — All answers directly address the question asked.

---

## Cost Analysis

| Resource | Usage | Cost |
|----------|-------|------|
| Groq API (per query) | ~800 input tokens, ~150 output tokens | ~$0.00012 |
| 100 queries | 80,000 input + 15,000 output tokens | ~$0.012 |
| ETL pipeline (one-time) | ~50,000 tokens | ~$0.004 |
| GCP Cloud Run | Not deployed (quota issue on free tier) | $0 |

**Total cost for development and testing: < $0.05**


#############################################################################

Building this project taught me that production AI systems are not just about prompting an LLM. The hard part is the data engineering — cleaning PDFs, chunking text intelligently, attaching metadata, and building reliable pipelines that don't crash on edge cases.

The combination of SQL for structured data and vector search for unstructured data is a powerful pattern that solves a real problem: executives need both the numbers AND the context behind them.