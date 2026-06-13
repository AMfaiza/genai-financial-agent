FROM python:3.11-slim

WORKDIR /app

# Copie les fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY .env .

EXPOSE 8000

CMD ["python", "src/04_api.py"]