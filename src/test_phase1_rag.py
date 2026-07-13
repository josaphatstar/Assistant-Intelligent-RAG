# src/test_phase1_rag.py
import requests

API_URL = "http://localhost:8000/query-rag"

print("=== DEMONSTRATION PHASE 1 - RAG CLASSIQUE ===\n")

questions = [
    "Comment cloturer la paie mensuelle dans Tom2Paie ?",
    "Comment creer un compte dans le plan comptable Tom2Pro ?",
]

for q in questions:
    print(f"QUESTION : {q}")
    print("-" * 60)
    r = requests.post(API_URL, json={"question": q}, timeout=300)
    print(r.json()["answer"])
    print("\n")