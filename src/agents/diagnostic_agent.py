# src/agents/diagnostic_agent.py

from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage
from agents.state import AgentState
import os
import re

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Codes d'erreur connus de la suite TOMATE
ERREURS_CONNUES = {
    "ERR-COMP-001": {
        "description": "Desequilibre debit/credit",
        "solution": "Verifier que Total Debit = Total Credit dans l'ecriture. "
                   "Menu Comptabilite > Saisie Journal > corriger les montants."
    },
    "ERR-PAIE-042": {
        "description": "Rubrique sans taux defini",
        "solution": "Aller dans Parametrage > Rubriques, "
                   "selectionner la rubrique et renseigner le taux manquant."
    },
    "ERR-STOCK-007": {
        "description": "Stock negatif detecte",
        "solution": "Stock > Inventaire > Verification Coherence, "
                   "puis Stock > Correction > Ajustement Inventaire."
    },
}

def diagnostic_agent(state: AgentState) -> AgentState:
    """Diagnostique les erreurs et codes d'erreur TOMATE."""

    question = state["question"]
    print(f"[Agent Diagnostic] Analyse : {question[:50]}...")

    # Chercher un code d'erreur dans la question
    codes_trouves = []
    for code in ERREURS_CONNUES:
        if code.lower() in question.lower():
            codes_trouves.append(code)

    if codes_trouves:
        # Erreur connue — réponse directe
        reponses = []
        for code in codes_trouves:
            info = ERREURS_CONNUES[code]
            reponses.append(
                f"Code {code} : {info['description']}\n"
                f"Solution : {info['solution']}"
            )
        reponse = "\n\n".join(reponses)
        niveau = 1

    else:
        # Erreur inconnue — LLM pour analyser
        llm = Ollama(
            base_url=OLLAMA_URL,
            model="llama3",
            temperature=0.1,
            num_predict=200
        )

        prompt = f"""Tu es expert du support TOMATE. Analyse ce probleme technique.
Si tu ne peux pas le resoudre avec certitude, dis-le clairement.
Probleme : {question}
Diagnostic et solution proposee :"""

        reponse = llm.invoke(prompt)
        niveau = 2

    return {
        **state,
        "messages": list(state["messages"]) + [
            AIMessage(content=f"[Diagnostic] {reponse}")
        ],
        "past_steps": state["past_steps"] + [f"Diagnostic: niveau {niveau}"],
        "niveau_support": niveau
    }