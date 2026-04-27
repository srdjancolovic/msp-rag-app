"""
app.py - Streamlit RAG aplikacija.
Korisnik postavlja pitanje, sistem pronalazi relevantne chunkove
iz Pinecone-a i šalje ih Claudeu kao kontekst.

Pokretanje: streamlit run app.py
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 100


def kreiraj_embedding_upita(upit: str) -> list[float]:
    """Kreira embedding za korisnikov upit."""
    odgovor = openai_client.embeddings.create(
        input=upit,
        model=EMBEDDING_MODEL
    )
    return odgovor.data[0].embedding


def podijeli_na_chunkove(tekst: str) -> list[str]:
    """Dijeli tekst na manje dijelove za embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_text(tekst)


def kreiraj_embedding_teksta(tekst: str) -> list[float]:
    """Kreira embedding za tekstualni chunk dokumenta."""
    odgovor = openai_client.embeddings.create(
        input=tekst,
        model=EMBEDDING_MODEL
    )
    return odgovor.data[0].embedding


def ingestuj_uploadane_fajlove(uploadani_fajlovi: list) -> tuple[int, int]:
    """Ingestuje više uploadanih fajlova u Pinecone."""
    ukupno_fajlova = 0
    ukupno_chunkova = 0

    for fajl in uploadani_fajlovi:
        try:
            sadrzaj = fajl.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.warning(f"Preskačem '{fajl.name}' jer nije UTF-8 tekstualni fajl.")
            continue

        if not sadrzaj.strip():
            st.warning(f"Preskačem '{fajl.name}' jer je fajl prazan.")
            continue

        chunkovi = podijeli_na_chunkove(sadrzaj)
        vektori = []
        prefiks = str(int(time.time() * 1000))

        for i, chunk in enumerate(chunkovi):
            embedding = kreiraj_embedding_teksta(chunk)
            vektori.append({
                "id": f"upload-{prefiks}-{fajl.name}-chunk-{i}",
                "values": embedding,
                "metadata": {
                    "tekst": chunk,
                    "fajl": fajl.name,
                    "chunk_index": i,
                    "izvor": "upload",
                },
            })

            if len(vektori) >= BATCH_SIZE:
                index.upsert(vectors=vektori)
                vektori = []

        if vektori:
            index.upsert(vectors=vektori)

        ukupno_fajlova += 1
        ukupno_chunkova += len(chunkovi)

    return ukupno_fajlova, ukupno_chunkova


def dohvati_preview_fajlova_iz_pinecone(limit: int = 200) -> list[str]:
    """
    Vraća preview liste fajlova iz Pinecone metadata polja `fajl`.
    Napomena: ovo je preview (sample), ne garantuje 100% svih fajlova.
    """
    probe_embedding = kreiraj_embedding_upita("prikazi listu svih dokumenata")
    rezultati = index.query(
        vector=probe_embedding,
        top_k=limit,
        include_metadata=True,
    )

    fajlovi = set()
    for match in rezultati.matches:
        metadata = getattr(match, "metadata", None) or {}
        naziv = metadata.get("fajl")
        if naziv:
            fajlovi.add(naziv)

    return sorted(fajlovi)


def obrisi_dokument_iz_pinecone(naziv_fajla: str):
    """Briše sve vektore dokumenta iz Pinecone po metadata polju `fajl`."""
    index.delete(filter={"fajl": {"$eq": naziv_fajla}})


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
    page_title="Imprimatur AI",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Root background ── */
.stApp {
    background: #0b1220;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid #1e2030;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 10px !important;
    border: 1px solid #2a2f45 !important;
    background: #151a2a !important;
    color: #cbd5e1 !important;
    text-align: left !important;
    padding: 0.5rem 0.7rem !important;
    margin-bottom: 0.35rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #6366f1 !important;
    color: #ffffff !important;
    background: #1f2540 !important;
}

/* ── Main container ── */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── App header ── */
.app-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 0 0 24px 24px;
    padding: 1.6rem 2rem 1.4rem 2rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.13);
}
.logo-mark {
    width: 46px;
    height: 46px;
    border-radius: 13px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -1px;
    box-shadow: 0 2px 12px rgba(99,102,241,0.45);
    flex-shrink: 0;
}
.app-title {
    font-size: 1.22rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
    letter-spacing: -0.3px;
}
.app-subtitle {
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 0.18rem 0 0 0;
}

/* ── Section label ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.65rem;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #111a2b;
    border: 1px solid #24324a;
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #fff !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    color: #000000 !important;
}

/* ── History panel ── */
.history-panel {
    background: #111a2b;
    border: 1px solid #24324a;
    border-radius: 16px;
    padding: 1rem 1rem 0.4rem 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.history-item {
    padding: 0.55rem 0.7rem;
    border-radius: 10px;
    background: #18243a;
    border: 1px solid #283954;
    margin-bottom: 0.45rem;
    font-size: 0.85rem;
    color: #cbd5e1;
    line-height: 1.4;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── File grid card ── */
.file-card {
    background: #111a2b;
    border: 1px solid #24324a;
    border-radius: 14px;
    padding: 1rem 0.9rem;
    margin-bottom: 0.75rem;
    transition: box-shadow 0.18s, transform 0.18s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.file-card:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.10);
    transform: translateY(-2px);
}
.file-card-icon {
    font-size: 1.5rem;
    margin-bottom: 0.4rem;
}
.file-card-name {
    font-size: 0.84rem;
    font-weight: 500;
    color: #e2e8f0;
    word-break: break-word;
    margin: 0;
    line-height: 1.35;
}
.file-card-ext {
    font-size: 0.72rem;
    color: #7c8da9;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: all 0.15s !important;
    border: 1px solid #2b3a54 !important;
    background: #141f33 !important;
    color: #cbd5e1 !important;
}
.stButton > button:hover {
    background: #1a2942 !important;
    border-color: #6366f1 !important;
    color: #e2e8f0 !important;
}

/* ── File uploader button label override ── */
[data-testid="stFileUploaderDropzone"] button {
    font-size: 0 !important;
}
[data-testid="stFileUploaderDropzone"] button::after {
    content: "Dodaj dokumente";
    font-size: 0.88rem;
}

/* ── Sidebar brand block ── */
.sidebar-brand {
    padding: 1.4rem 1rem 1rem 1rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #1e2030;
}
.sidebar-brand-title {
    font-size: 1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
}
.sidebar-brand-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin: 0.2rem 0 0 0;
}
.sidebar-nav-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #475569;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.8rem 1rem 0.3rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ──
if "poruke" not in st.session_state:
    st.session_state.poruke = []
if "pinecone_preview_fajlovi" not in st.session_state:
    st.session_state["pinecone_preview_fajlovi"] = dohvati_preview_fajlova_iz_pinecone()
if "aktivna_stranica" not in st.session_state:
    st.session_state["aktivna_stranica"] = "Chat"
if "dok_search_input" not in st.session_state:
    st.session_state["dok_search_input"] = ""


def ponisti_pretragu():
    st.session_state["dok_search_input"] = ""

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <p class="sidebar-brand-title">Imprimatur AI</p>
        <p class="sidebar-brand-sub">Pametni asistent za dokumente</p>
    </div>
    <div class="sidebar-nav-label">Navigacija</div>
    """, unsafe_allow_html=True)

    if st.button(
        "Chat",
        key="nav_chat",
        use_container_width=True,
        type="primary" if st.session_state["aktivna_stranica"] == "Chat" else "secondary",
    ):
        st.session_state["aktivna_stranica"] = "Chat"
    if st.button(
        "Dokumenti",
        key="nav_dokumenti",
        use_container_width=True,
        type="primary" if st.session_state["aktivna_stranica"] == "Dokumenti" else "secondary",
    ):
        st.session_state["aktivna_stranica"] = "Dokumenti"

aktivna_stranica = st.session_state["aktivna_stranica"]

# ── Top header ──
st.markdown("""
<div class="app-header">
    <div class="logo-mark">Im</div>
    <div>
        <p class="app-title">Imprimatur — AI asistent</p>
        <p class="app-subtitle">Postavi pitanje i dobij precizan odgovor iz tvojih dokumenata</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PAGE: CHAT
# ──────────────────────────────────────────────
if aktivna_stranica == "Chat":
    col_chat, col_history = st.columns([0.68, 0.32], gap="large")

    with col_chat:
        header_l, header_r = st.columns([0.75, 0.25])
        with header_l:
            st.markdown('<div class="section-label">Razgovor</div>', unsafe_allow_html=True)
        with header_r:
            if st.button("Obriši chat", use_container_width=True, key="obrisi_chat"):
                st.session_state.poruke = []
                st.rerun()

        for poruka in st.session_state.poruke:
            with st.chat_message(poruka["uloga"]):
                st.markdown(poruka["sadrzaj"])
                if "izvori" in poruka:
                    with st.expander("Prikazati izvore"):
                        st.text(poruka["izvori"])

        upit = st.chat_input("Postavi pitanje o tvojim dokumentima...")

        if upit:
            with st.chat_message("user"):
                st.markdown(upit)
            st.session_state.poruke.append({"uloga": "user", "sadrzaj": upit})

            with st.chat_message("assistant"):
                with st.spinner("Tražim odgovor..."):
                    try:
                        matches = pretrazi_pinecone(upit)
                        if not matches:
                            odgovor_tekst = "Nisam pronašao relevantne informacije u dokumentima."
                            st.markdown(odgovor_tekst)
                            st.session_state.poruke.append({
                                "uloga": "assistant",
                                "sadrzaj": odgovor_tekst
                            })
                        else:
                            kontekst = pripremi_kontekst(matches)
                            odgovor_tekst = pitaj_claudea(upit, kontekst)
                            st.markdown(odgovor_tekst)
                            with st.expander("Prikazati izvore"):
                                st.text(kontekst)
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

    with col_history:
        st.markdown('<div class="section-label">Istorija</div>', unsafe_allow_html=True)
        st.markdown('<div class="history-panel">', unsafe_allow_html=True)
        korisnicka_pitanja = [
            p["sadrzaj"] for p in st.session_state.poruke if p.get("uloga") == "user"
        ]
        if not korisnicka_pitanja:
            st.caption("Postavi prvo pitanje da bi se istorija pojavila.")
        else:
            for pitanje in reversed(korisnicka_pitanja):
                skraceno = pitanje[:72] + "..." if len(pitanje) > 72 else pitanje
                st.markdown(
                    f'<div class="history-item">{skraceno}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PAGE: DOKUMENTI
# ──────────────────────────────────────────────
else:
    st.markdown('<div class="section-label">Upload dokumenata</div>', unsafe_allow_html=True)
    uploadani_fajlovi = st.file_uploader(
        "Dodaj dokumente",
        type=["txt", "md", "csv"],
        accept_multiple_files=True,
        help="Podržani formati: .txt, .md, .csv",
        label_visibility="visible",
    )
    if st.button("Ucitaj dokument u bazu", key="ingest_upload", use_container_width=False):
        if not uploadani_fajlovi:
            st.warning("Odaberi bar jedan fajl.")
        else:
            with st.spinner("Ingestion u toku..."):
                try:
                    broj_fajlova, broj_chunkova = ingestuj_uploadane_fajlove(uploadani_fajlovi)
                    st.success(f"Ingestovano {broj_fajlova} fajlova — {broj_chunkova} chunkova dodato u Pinecone.")
                    st.session_state["pinecone_preview_fajlovi"] = dohvati_preview_fajlova_iz_pinecone()
                except Exception as e:
                    st.error(f"Greška: {str(e)}")

    hdr_l, hdr_r = st.columns([0.7, 0.3])
    with hdr_l:
        st.markdown('<div class="section-label">Indeksirani dokumenti</div>', unsafe_allow_html=True)
    with hdr_r:
        if st.button("Osvježi", key="osvjezi_preview", use_container_width=True):
            st.session_state["pinecone_preview_fajlovi"] = dohvati_preview_fajlova_iz_pinecone()

    preview_fajlovi = st.session_state.get("pinecone_preview_fajlovi", [])
    search_col, reset_col, _ = st.columns([0.5, 0.2, 0.3], gap="small")
    with search_col:
        pretraga_dokumenata = st.text_input(
            "Pretraga dokumenata",
            key="dok_search_input",
            placeholder="Unesi naziv dokumenta...",
        ).strip().lower()
    with reset_col:
        st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
        st.button("Ponisti pretragu", key="ponisti_pretragu_btn", on_click=ponisti_pretragu)
    if pretraga_dokumenata:
        preview_fajlovi = [
            naziv for naziv in preview_fajlovi
            if pretraga_dokumenata in naziv.lower()
        ]

    def ikona_fajla(naziv: str) -> str:
        ext = naziv.rsplit(".", 1)[-1].lower() if "." in naziv else ""
        return {"md": "📝", "csv": "📊", "txt": "📄"}.get(ext, "📁")

    if not preview_fajlovi:
        st.info("Nema pronađenih fajlova. Uploadaj dokumente ili pokreni ingest iz terminala.")
    else:
        st.caption(f"{len(preview_fajlovi)} dokumenata indeksirano u Pinecone")
        cols = st.columns(3, gap="medium")
        for idx, naziv in enumerate(preview_fajlovi):
            ext = naziv.rsplit(".", 1)[-1].upper() if "." in naziv else "FILE"
            clean_name = naziv.rsplit(".", 1)[0] if "." in naziv else naziv
            with cols[idx % 3]:
                st.markdown(
                    f"""
                    <div class="file-card">
                        <div class="file-card-icon">{ikona_fajla(naziv)}</div>
                        <p class="file-card-name">{clean_name}</p>
                        <p class="file-card-ext">{ext}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Obrisi dokument", key=f"obrisi_{idx}_{naziv}", use_container_width=False):
                    try:
                        with st.spinner(f"Brišem dokument '{naziv}'..."):
                            obrisi_dokument_iz_pinecone(naziv)
                            st.session_state["pinecone_preview_fajlovi"] = dohvati_preview_fajlova_iz_pinecone()
                        st.toast(f"Dokument '{naziv}' je obrisan.", icon="✅")
                        st.success(f"Dokument '{naziv}' je obrisan iz Pinecone-a.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Brisanje nije uspjelo: {str(e)}")
