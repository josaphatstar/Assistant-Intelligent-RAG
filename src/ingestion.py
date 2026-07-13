# src/ingestion.py

import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "documentation_tomate"
DATA_DIR = "/app/data/documents"

def load_documents():
    documents = []
    fichiers = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

    if not fichiers:
        print("ATTENTION : aucun PDF trouve dans", DATA_DIR)
        return []

    for fichier in fichiers:
        chemin = os.path.join(DATA_DIR, fichier)
        print(f"Chargement : {fichier}...")
        try:
            loader = PyPDFLoader(chemin)
            docs = loader.load()
            for doc in docs:
                doc.metadata["module"] = fichier.replace(".pdf", "")
            documents.extend(docs)
            print(f"  OK — {len(docs)} pages")
        except Exception as e:
            print(f"  IGNORE — {str(e)[:80]}")
            continue

    print(f"\n{len(documents)} pages chargées au total")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunks generés")
    return chunks


def store_embeddings(chunks):
    print("\nChargement du modele d'embedding...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Connexion explicite au serveur ChromaDB
    print(f"Connexion a ChromaDB sur {CHROMA_HOST}:{CHROMA_PORT}...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # Vérification de la connexion
    print(f"Collections existantes : {client.list_collections()}")

    # Supprimer l'ancienne collection si elle existe
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Ancienne collection supprimee")
    except Exception:
        pass

    print("Stockage dans ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client
    )

    count = vectorstore._collection.count()
    print(f"Termine ! {count} vecteurs stockes dans ChromaDB")
    return vectorstore


if __name__ == "__main__":
    print("=== INGESTION DOCUMENTATION TOMATE ===\n")
    docs = load_documents()
    if docs:
        chunks = split_documents(docs)
        store_embeddings(chunks)
        print("\n=== INGESTION TERMINEE AVEC SUCCES ===")