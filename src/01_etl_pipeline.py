import os
import re
import fitz  # pymupdf
import docx
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()


COLLECTION_NAME = "financial_reports"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Mapping des fichiers vers les métadonnées
FILE_METADATA = {
    "apple-2024-10k.pdf": {"company": "Apple", "year": 2024},
    "apple-2025-10k.pdf": {"company": "Apple", "year": 2025},
    "amazon-2024-10k.pdf": {"company": "Amazon", "year": 2024},
    "amazon-2025-10k.pdf": {"company": "Amazon", "year": 2025},
    "goog-10-k-2024.pdf": {"company": "Google", "year": 2024},
    "GOOG-10-K-2025.pdf": {"company": "Google", "year": 2025},
    "MSFT_FY24Q4_10K.docx": {"company": "Microsoft", "year": 2024},
    "MSFT_FY25q4_10K.docx": {"company": "Microsoft", "year": 2025},
    "tsla-20241231-gen.pdf": {"company": "Tesla", "year": 2024},
    "tsla-20251231-gen.pdf": {"company": "Tesla", "year": 2025},
}

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """hunks sémantiques"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

def main():
    print("Initialisation de Qdrant et du modèle d'embedding...")
    qdrant = QdrantClient(url="http://localhost:6333")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )
    print(f"Collection '{COLLECTION_NAME}' créée !")

    # Traitement de chaque document
    raw_pdfs_path = "../data/raw_pdfs"
    all_points = []
    point_id = 0

    for filename, metadata in FILE_METADATA.items():
        filepath = os.path.join(raw_pdfs_path, filename)
        if not os.path.exists(filepath):
            print(f"Fichier non trouvé: {filename}")
            continue

        print(f"Traitement de {filename}...")

        # Extraction du texte
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_docx(filepath)

        # Nettoyage
        text = clean_text(text)

        # Chunking
        chunks = chunk_text(text)
        print(f"  → {len(chunks)} chunks créés")

        # Embedding et insertion
        for chunk in chunks:
            vector = embedder.encode(chunk).tolist()
            all_points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk,
                    "company": metadata["company"],
                    "year": metadata["year"],
                    "source": filename
                }
            ))
            point_id += 1

    # Insertion dans Qdrant
    print(f"\nInsertion de {len(all_points)} chunks dans Qdrant...")
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=all_points
    )
    print("ETL Pipeline terminé avec succès !")

if __name__ == "__main__":
    main() 

# resulat Insertion de 1289 chunks dans Qdrant...