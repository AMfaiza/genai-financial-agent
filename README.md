# Multi-Modal Financial AI Agent

An AI agent that answers complex business questions about major tech companies by combining SQL queries on structured financial data and semantic search on annual reports (10-K).

## Prerequisites

- Python 3.9+
- Docker
- A Groq API key (free at console.groq.com)

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/AMfaiza/genai-financial-agent.git
cd genai-financial-agent
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up your API key
```bash
touch .env
```
Add this to `.env`:
API_KEY=your_groq_api_key_here

### 4. Start Qdrant with Docker
```bash
cd docker
docker-compose up -d
cd ..
```

### 5. Run the ETL Pipeline (loads documents into Qdrant)
```bash
cd src
python 01_etl_pipeline.py
```

### 6. Create the SQL database
```bash
python 02_create_database.py
```

### 7. Start the API
```bash
python 04_api.py
```

## Usage

The API runs on `http://localhost:8000`

### Query the agent
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Amazon total revenue in 2025?"}'
```

### Example questions

**Financial metrics (SQL):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare Microsoft and Google net income in 2024"}'
```

**Qualitative insights (Vector search):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What did Apple report say about supply chain risks?"}'
```

## Project Structure
projet-final/

├── src/

│   ├── 01_etl_pipeline.py      # Loads PDFs into Qdrant

│   ├── 02_create_database.py   # Creates SQLite database

│   ├── 03_agent.py             # LangGraph agent (standalone)

│   └── 04_api.py               # FastAPI REST endpoint

├── data/

│   ├── raw_pdfs/               # Annual reports (10-K)

│   └── financial_data.db       # SQLite database

├── docker/

│   └── docker-compose.yml      # Qdrant container

├── Dockerfile                  # API container

├── docker-compose.yml          # Full stack deployment

├── requirements.txt

├── REPORT.md

└── README.md

## Architecture
User Question → FastAPI → Question Router

↓              ↓

SQL Query    Vector Search

↓              ↓

SQLite         Qdrant

↓              ↓

Groq LLM     Groq LLM

↓              ↓

JSON Response

## Dataset

10 annual reports (10-K) from 5 major tech companies:
- Apple (2024, 2025)
- Amazon (2024, 2025)
- Google (2024, 2025)
- Microsoft (2024, 2025)
- Tesla (2024, 2025)
Sauvegarde puis push :
bashgit add README.md
git commit -m "Add README"
git push
