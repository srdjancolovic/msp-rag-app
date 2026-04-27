"""
RAG STREAMLIT APP — AI Forward Dan 3
Pretraga + Claude koristeći Pinecone RAG.

Pokretanje: streamlit run rag_app.py

PRILAGODBA:
- COMPANY_NAME, COMPANY_COLOR, SYSTEM_PROMPT za vaš MSP
"""
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from anthropic import Anthropic

load_dotenv()

# === PRILAGODI ===
COMPANY_NAME = "Vaša Firma"
COMPANY_ICON = "🤖"
COMPANY_COLOR = "#0F2A47"

SYSTEM_PROMPT = """Ti si AI asistent firme [PROMIJENI].

PRAVILA:
1. Odgovaraj SAMO iz priloženog konteksta
2. Ako informacija nije u kontekstu: "Nemam tu informaciju u trenutnoj bazi znanja"
3. Nikad ne izmišljaj cijene, dimenzije, rokove
4. Uvijek navedi izvor (naziv dokumenta)
5. Kratko, koncizno, BCS jezik
6. Profesionalan, ljubazan ton"""

# === KLIJENTI ===
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

st.set_page_config(
    page_title=f"{COMPANY_NAME} — AI Asistent",
    page_icon=COMPANY_ICON,
    layout="wide"
)

st.markdown(f"<style>h1 {{ color: {COMPANY_COLOR}; }}</style>", unsafe_allow_html=True)
st.title(f"{COMPANY_ICON} {COMPANY_NAME} — AI Asistent")
st.caption("Postavi pitanje — AI odgovara iz baze znanja.")


def get_embedding(text):
    return openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding


def search_kb(query, top_k=5):
    emb = get_embedding(query)
    results = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [{
        "text": m.metadata["text"],
        "source": m.metadata["source"],
        "score": m.score
    } for m in results.matches]


def generate_answer(query, chunks):
    context = "\n\n".join([
        f"[Izvor: {c['source']}]\n{c['text']}" for c in chunks
    ])
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"KONTEKST:\n\n{context}\n\nPITANJE: {query}"
        }]
    )
    return response.content[0].text


with st.sidebar:
    st.header("⚙ Podešavanja")
    top_k = st.slider("Broj chunk-ova", 3, 10, 5)
    show_sources = st.checkbox("Prikaži izvore", True)
    show_scores = st.checkbox("Prikaži scores", False)
    
    st.markdown("---")
    st.caption("💡 **Kako radi:**")
    st.caption("1. Pitanje → vektor")
    st.caption("2. Pinecone traži 5 najsličnijih")
    st.caption("3. Claude odgovara")
    
    if st.button("🗑 Obriši istoriju"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander("📚 Izvori"):
                for s in msg["sources"]:
                    score = f" ({s['score']:.2%})" if show_scores else ""
                    st.caption(f"**{s['source']}**{score}")
                    st.text(s['text'][:300] + "...")

if query := st.chat_input("Postavi pitanje..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Tražim..."):
            try:
                chunks = search_kb(query, top_k)
                answer = generate_answer(query, chunks) if chunks else "Nisam pronašao relevantne informacije."
                
                st.write(answer)
                
                if show_sources and chunks:
                    with st.expander("📚 Izvori"):
                        for c in chunks:
                            score = f" ({c['score']:.2%})" if show_scores else ""
                            st.caption(f"**{c['source']}**{score}")
                            st.text(c['text'][:300] + "...")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": chunks if show_sources else None
                })
            except Exception as e:
                st.error(f"Greška: {e}")
