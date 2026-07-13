# src/rag_chain.py

import os
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "documentation_tomate"

def create_rag_chain():

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = Ollama(
        base_url=OLLAMA_URL,
        model="llama3",
        temperature=0.3,
        num_predict=300
    )

    template = """Assistant technique TOMATE (Tom2Pro, Tom2Paie, Tom2Stock, Tom2Monitoring) - TECHEXPERT.
Regles :
- Reponds UNIQUEMENT avec le contexte fourni.
- Si absent : "Information non disponible, ouvrir ticket niveau 3."
- Cite la source. Sois concis et direct.

Contexte documentaire :
{context}

Question du technicien :
{question}

Reponse technique detaillee :"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([
            f"[Source : {d.metadata.get('module', 'Inconnu')}]\n{d.page_content}"
            for d in docs
        ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain