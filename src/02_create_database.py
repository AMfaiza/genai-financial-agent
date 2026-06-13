import sqlite3
import os

db_path = "../data/financial_data.db"
os.makedirs("../data", exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS financial_metrics")
cursor.execute("""
CREATE TABLE financial_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,
    year INTEGER NOT NULL,
    total_revenue REAL,
    net_income REAL,
    operating_income REAL,
    gross_profit REAL,
    total_assets REAL,
    employees INTEGER
)
""")

# Chiffres tirés des rapports 10-K 
# Jai essaye avec une fonction qui utilise Gemini/Groq pour extraire automatiquement les chiffres depuis les PDFs.
#Comme ce qu'on a fait dans le Lab 1 avec Pydantic + Instructor mais les PDF avait beaucoup de page ca prenait que les 15 premieres pages
data = [
    # Apple
    ("Apple", 2024, 391035.0, 93736.0, 123216.0, 180683.0, 364980.0, 164000),
    ("Apple", 2025, 416161.0, 112010.0, 133050.0, 195201.0, 359241.0, 166000),
    # Amazon
    ("Amazon", 2024, 637959.0, 59248.0, 68593.0, 311671.0, 624894.0, 1556000),
    ("Amazon", 2025, 716924.0, 77670.0, 79975.0, 360510.0, 818042.0, 1576000),
    # Google
    ("Google", 2024, 350018.0, 100118.0, 112390.0, 203712.0, 450256.0, 183323),
    ("Google", 2025, 402836.0, 132170.0, 129039.0, 240301.0, 595281.0, 190820),
    # Microsoft
    ("Microsoft", 2024, 245122.0, 88136.0, 109433.0, 171008.0, 512163.0, 228000),
    ("Microsoft", 2025, 281724.0, 101832.0, 128528.0, 193893.0, 619003.0, 228000),
    # Tesla
    ("Tesla", 2024, 97690.0, 7153.0, 7076.0, 17450.0, 122070.0, 125665),
    ("Tesla", 2025, 94827.0, 3855.0, 4355.0, 17094.0, 137806.0, 134785),
]

cursor.executemany("""
INSERT INTO financial_metrics 
(company, year, total_revenue, net_income, operating_income, gross_profit, total_assets, employees)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", data)

conn.commit()
conn.close()
print(" Base de données créée avec succès !")
print(f"{len(data)} enregistrements insérés")

# resultats 10 enregistrements insérés