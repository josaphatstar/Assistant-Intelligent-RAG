# src/app.py

import streamlit as st
import requests

st.set_page_config(
    page_title="Assistant TOMATE",
    page_icon="T",
    layout="wide"
)

st.title("Assistant Support Technique TOMATE")

mode = st.radio(
    "Mode de fonctionnement :",
    ["Phase 1 - RAG classique", "Phase 2 - Architecture Agentic AI"],
    horizontal=True
)

st.markdown("Posez votre question sur Tom2Pro, Tom2Paie, Tom2Stock ou Tom2Monitoring.")

if mode == "Phase 1 - RAG classique":
    API_URL = "http://python-api:8000/query-rag"
else:
    API_URL = "http://python-api:8000/query-agent"

question = st.text_input("Votre question :")

if st.button("Rechercher", type="primary") and question:
    with st.spinner(f"Traitement en cours ({mode})... patience, 1-2 minutes"):
        try:
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=600
            )
            data = response.json()

            st.success("Reponse trouvee")
            st.markdown("### Reponse")
            st.write(data["answer"])

            if mode == "Phase 2 - Architecture Agentic AI":
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Niveau de support", data["niveau_support"])
                with col2:
                    st.write("**Agents utilises :**")
                    for agent in data["agents_utilises"]:
                        st.write(f"- {agent}")

        except requests.exceptions.Timeout:
            st.error("Le modele prend trop de temps. Reessayez.")
        except Exception as e:
            st.error(f"Erreur : {e}")