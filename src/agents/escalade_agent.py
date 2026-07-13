# src/agents/escalade_agent.py

from langchain_core.messages import AIMessage
from agents.state import AgentState
from datetime import datetime

def escalade_agent(state: AgentState) -> AgentState:
    """Génère un ticket de support niveau 3."""

    question = state["question"]
    historique = "\n".join(state["past_steps"])
    horodatage = datetime.now().strftime("%Y-%m-%d %H:%M")

    ticket = f"""
=== TICKET DE SUPPORT NIVEAU 3 ===
Date       : {horodatage}
Priorite   : HAUTE
Statut     : OUVERT

DESCRIPTION DU PROBLEME :
{question}

ETAPES DE DIAGNOSTIC EFFECTUEES :
{historique}

CONCLUSION :
Ce probleme depasse le niveau de support automatise.
Une intervention humaine est requise.

ACTION REQUISE :
Contacter l'equipe technique TECHEXPERT
==================================="""

    print(f"[Agent Escalade] Ticket genere")

    return {
        **state,
        "messages": list(state["messages"]) + [
            AIMessage(content=ticket)
        ],
        "past_steps": state["past_steps"] + ["Escalade: ticket niveau 3 genere"],
        "niveau_support": 3
    }	