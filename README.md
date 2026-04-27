# MSP RAG App

Jednostavna RAG aplikacija sa:

- OpenAI embeddings (`text-embedding-3-small`)
- Pinecone vector bazom
- Claude modelom za generisanje odgovora
- Streamlit interfejsom

## 1) Preduvjeti

- Python 3.10+
- OpenAI API key
- Pinecone API key i kreiran Pinecone index
- Anthropic API key

## 2) Instalacija

U root folderu projekta pokreni:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Konfiguracija `.env`

Kreiraj ili uredi `.env` fajl u rootu projekta:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 4) Ubacivanje dokumenata u Pinecone

### Opcija A (preporučeno): ingest svih fajlova iz `imprimatur/dokumenti`

```bash
python3 ingest.py
```

Ova komanda automatski učitava sve fajlove iz foldera `imprimatur/dokumenti`.

### Opcija B: ingest jednog fajla

```bash
python3 ingest.py --file imprimatur/dokumenti/01_katalog_knjiga.md
```

### Opcija C: ingest svih fajlova iz custom foldera

```bash
python3 ingest.py --folder imprimatur/dokumenti
```

Skripta:

- učita sadržaj fajla
- podijeli tekst na chunkove
- kreira embeddinge
- upiše vektore u Pinecone

## 5) Pokretanje aplikacije

Pokreni Streamlit:

```bash
streamlit run rag_app.py
```

Nakon pokretanja, otvori URL koji Streamlit prikaže u terminalu (obično `http://localhost:8501`).

## 6) Kako koristiti

1. Ingestuj jedan ili više `.txt` dokumenata preko `ingest.py`
2. Otvori aplikaciju
3. Postavi pitanje u chat
4. Aplikacija povlači relevantne chunkove iz Pinecone-a i Claude generiše odgovor
