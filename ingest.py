"""
ingest.py - Učitava tekstualne fajlove, dijeli ih na chunkove,
kreira embeddings i upisuje u Pinecone index.

Pokretanje: python ingest.py --file putanja/do/fajla.txt
"""

import os
import argparse
import time
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Učitaj .env fajl
load_dotenv()

# Inicijalizacija klijenata
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Parametri za dijeljenje teksta
CHUNK_SIZE = 500          # Broj karaktera po chunku
CHUNK_OVERLAP = 50        # Preklapanje između chunkova
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
BATCH_SIZE = 100          # Broj vektora koji se upisuju odjednom
DEFAULT_FOLDER = "imprimatur/dokumenti"


def ucitaj_tekst(putanja: str) -> str:
    """Čita sadržaj tekstualnog fajla."""
    with open(putanja, "r", encoding="utf-8") as f:
        return f.read()


def podijeli_na_chunkove(tekst: str) -> list[str]:
    """Dijeli tekst na manje dijelove za embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_text(tekst)


def kreiraj_embedding(tekst: str) -> list[float]:
    """Kreira embedding vektor za dati tekst koristeći OpenAI."""
    odgovor = openai_client.embeddings.create(
        input=tekst,
        model=EMBEDDING_MODEL
    )
    return odgovor.data[0].embedding


def upisi_u_pinecone(chunkovi: list[str], naziv_fajla: str):
    """Kreira embeddings za sve chunkove i upisuje ih u Pinecone."""
    print(f"Ukupno chunkova: {len(chunkovi)}")
    
    vektori = []
    
    for i, chunk in enumerate(chunkovi):
        print(f"  Procesiranje chunka {i+1}/{len(chunkovi)}...")
        
        embedding = kreiraj_embedding(chunk)
        
        # Svaki vektor ima ID, embedding i metadata
        vektori.append({
            "id": f"{naziv_fajla}-chunk-{i}",
            "values": embedding,
            "metadata": {
                "tekst": chunk,
                "fajl": naziv_fajla,
                "chunk_index": i
            }
        })
        
        # Upisuj u batchevima da izbjegneš limite
        if len(vektori) >= BATCH_SIZE:
            index.upsert(vectors=vektori)
            print(f"  Upisano {len(vektori)} vektora u Pinecone.")
            vektori = []
            time.sleep(0.5)  # Kratka pauza između batcheva
    
    # Upiši ostatak
    if vektori:
        index.upsert(vectors=vektori)
        print(f"  Upisano {len(vektori)} vektora u Pinecone.")
    
    print("Ingestion završen!")


def ingestuj_fajl(putanja: str):
    """Ingestuje jedan fajl u Pinecone."""
    naziv_fajla = os.path.basename(putanja)
    print(f"\nUčitavam: {putanja}")
    tekst = ucitaj_tekst(putanja)
    print(f"Dužina teksta: {len(tekst)} karaktera")

    print("Dijelim na chunkove...")
    chunkovi = podijeli_na_chunkove(tekst)

    print("Kreiram embeddings i upisujem u Pinecone...")
    upisi_u_pinecone(chunkovi, naziv_fajla)


def main():
    parser = argparse.ArgumentParser(description="Upis dokumenta/foldera u Pinecone.")
    parser.add_argument("--file", help="Putanja do jednog tekstualnog fajla")
    parser.add_argument(
        "--folder",
        default=DEFAULT_FOLDER,
        help=f"Folder iz kojeg ingestuje sve fajlove (default: {DEFAULT_FOLDER})",
    )
    args = parser.parse_args()

    if args.file:
        ingestuj_fajl(args.file)
        return

    if not os.path.isdir(args.folder):
        raise FileNotFoundError(f"Folder ne postoji: {args.folder}")

    fajlovi = [
        os.path.join(args.folder, ime)
        for ime in sorted(os.listdir(args.folder))
        if os.path.isfile(os.path.join(args.folder, ime))
    ]

    if not fajlovi:
        print(f"Nema fajlova za ingest u folderu: {args.folder}")
        return

    print(f"Pronađeno fajlova u folderu '{args.folder}': {len(fajlovi)}")
    for putanja in fajlovi:
        ingestuj_fajl(putanja)


if __name__ == "__main__":
    main()
