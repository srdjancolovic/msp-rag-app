"""
app.py - Streamlit RAG aplikacija.
Korisnik postavlja pitanje, sistem pronalazi relevantne chunkove
iz Pinecone-a i šalje ih Claudeu kao kontekst.

Pokretanje: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import anthropic

# Učitaj .env fajl
load_dotenv()

# Inicijalizacija klijenata
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Parametri
EMBEDDING_MODEL = "text-embedding-3-small"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
TOP_K = 5  # Broj chunkova koji se dohvataju iz Pinecone-a


def kreiraj_embedding_upita(upit: str) -> list[float]:
    """Kreira embedding za korisnikov upit."""
    odgovor = openai_client.embeddings.create(
        input=upit,
        model=EMBEDDING_MODEL
    )
    return odgovor.data[0].embedding


def pretrazi_pinecone(upit: str) -> list[dict]:
    """Pronalazi najsličnije chunkove u Pinecone-u."""
    embedding = kreiraj_embedding_upita(upit)
    
    rezultati = index.query(
        vector=embedding,
        top_k=TOP_K,
        include_metadata=True
    )
    
    return rezultati.matches


def pripremi_kontekst(matches: list[dict]) -> str:
    """Spaja pronađene chunkove u jedan kontekstni string."""
    dijelovi = []
    for i, match in enumerate(matches, 1):
        tekst = match.metadata.get("tekst", "")
        fajl = match.metadata.get("fajl", "nepoznato")
        score = round(match.score, 3)
        dijelovi.append(f"[Izvor {i} | Fajl: {fajl} | Relevantnost: {score}]\n{tekst}")
    
    return "\n\n---\n\n".join(dijelovi)


def pitaj_claudea(upit: str, kontekst: str) -> str:
    """Šalje upit i kontekst Claudeu i vraća odgovor."""
    
    sistem_prompt = """Ti si koristan asistent koji odgovara ISKLJUČIVO na osnovu dostavljenog konteksta.
Ako odgovor nije u kontekstu, reci: "Na osnovu dostupnih dokumenata, ne mogu odgovoriti na ovo pitanje."
Uvijek navedi na koji izvor se pozivas kada daješ odgovor."""

    poruka = f"""Kontekst iz dokumenata:
{kontekst}

Pitanje korisnika:
{upit}"""

    odgovor = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=sistem_prompt,
        messages=[
            {"role": "user", "content": poruka}
        ]
    )
    
    return odgovor.content[0].text


# ─── Streamlit UI ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Imprimatur AI Assistant",
    page_icon="✨",
    layout="wide"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.6rem;
            padding-bottom: 1.2rem;
            max-width: 980px;
        }
        .app-header {
            background: linear-gradient(130deg, #f7f8ff 0%, #eef4ff 100%);
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }
        .logo-badge {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            background: linear-gradient(140deg, #6366f1 0%, #22c55e 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.3rem;
            font-weight: 700;
            flex-shrink: 0;
        }
        .app-title {
            font-size: 1.5rem;
            font-weight: 650;
            color: #0f172a;
            margin: 0;
        }
        .app-subtitle {
            color: #475569;
            margin-top: 0.2rem;
            margin-bottom: 0;
            font-size: 0.95rem;
        }
        .chat-hint {
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            background: #ffffff;
            color: #334155;
            margin-bottom: 0.7rem;
        }
        .stChatMessage {
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            background: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="logo-badge">I✦</div>
        <div>
            <p class="app-title">Imprimatur - AI asistent</p>
            <p class="app-subtitle">Postavi pitanje i dobij odgovor na osnovu tvojih dokumenata u Pinecone-u.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="chat-hint">
        Primjer: <b>"Koja su pravila saradnje sa autorima?"</b> ili <b>"Sažmi FAQ u 5 tačaka."</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# Inicijalizacija historije poruka u session state
if "poruke" not in st.session_state:
    st.session_state.poruke = []

with st.sidebar:
    st.markdown("### Istorija chatova")
    korisnicka_pitanja = [
        poruka["sadrzaj"]
        for poruka in st.session_state.poruke
        if poruka.get("uloga") == "user"
    ]
    if not korisnicka_pitanja:
        st.caption("Još nema poruka.")
    else:
        for i, pitanje in enumerate(reversed(korisnicka_pitanja), 1):
            st.markdown(f"**{i}.** {pitanje}")

col_left, col_right = st.columns([0.85, 0.15])
with col_right:
    if st.button("Obriši chat", use_container_width=True):
        st.session_state.poruke = []
        st.rerun()

# Prikaz historije razgovora
for poruka in st.session_state.poruke:
    with st.chat_message(poruka["uloga"]):
        st.markdown(poruka["sadrzaj"])
        # Prikaži izvore ako postoje
        if "izvori" in poruka:
            with st.expander("📄 Korišteni chunkovi"):
                st.text(poruka["izvori"])

# Polje za unos pitanja
upit = st.chat_input("Postavi pitanje o tvojim dokumentima...")

if upit:
    # Prikaži korisnikovo pitanje
    with st.chat_message("user"):
        st.markdown(upit)
    st.session_state.poruke.append({"uloga": "user", "sadrzaj": upit})
    
    # Pretraži i generiraj odgovor
    with st.chat_message("assistant"):
        with st.spinner("Pretražujem dokumente..."):
            try:
                # 1. Pronađi relevantne chunkove
                matches = pretrazi_pinecone(upit)
                
                if not matches:
                    odgovor_tekst = "Nisam pronašao relevantne informacije u dokumentima."
                    st.markdown(odgovor_tekst)
                    st.session_state.poruke.append({
                        "uloga": "assistant",
                        "sadrzaj": odgovor_tekst
                    })
                else:
                    # 2. Pripremi kontekst
                    kontekst = pripremi_kontekst(matches)
                    
                    # 3. Generiraj odgovor sa Claudeom
                    odgovor_tekst = pitaj_claudea(upit, kontekst)
                    st.markdown(odgovor_tekst)
                    
                    # 4. Prikaži izvore
                    with st.expander("📄 Korišteni chunkovi"):
                        st.text(kontekst)
                    
                    # Spremi u historiju sa izvorima
                    st.session_state.poruke.append({
                        "uloga": "assistant",
                        "sadrzaj": odgovor_tekst,
                        "izvori": kontekst
                    })
                    
            except Exception as e:
                greska = f"Greška: {str(e)}"
                st.error(greska)
                st.session_state.poruke.append({
                    "uloga": "assistant",
                    "sadrzaj": greska
                })
