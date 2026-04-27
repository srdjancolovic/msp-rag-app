# Imprimatur — Demo set za RAG

## Šta sadrži

5 dokumenata o Imprimaturu — izdavač akademske, stručne i beletristike.

1. `01_katalog_knjiga.md` — knjige po kategoriji (pravo, ekonomija, beletristika, dječja, biznis)
2. `02_distribucija_kanali.md` — kanali prodaje, top 10, analiza Q1
3. `03_pravila_autori.md` — proces objavljivanja, honorari, autorska prava
4. `04_FAQ.md` — pitanja kupaca, biblioteka, autora, distributera
5. `05_prodaja_q1.csv` — prodaja Q1 (26 redova)

**Očekivan broj chunkova nakon ingest:** ~25

## 5 pitanja koja garantovano rade

#### 1. Cijena specifičnog naslova

**P:** Koliko košta knjiga "AI za menadžere"?

**Očekivan odgovor:** IM-BI-002 "Vještačka inteligencija za menadžere" autor Đuro Grubišić. Cijena 49 KM. 280 stranica. Izdato 2026.

**Izvor:** 01_katalog_knjiga.md

#### 2. Najprodavaniji naslovi

**P:** Koji su top 3 naslova po prihodu u Q1 2026?

**Očekivan odgovor:** 
1. Krivično pravo — 11.303 KM (4 univerziteta)
2. Mikroekonomija (Mankiw) — 7.975 KM
3. AI za menadžere — 7.350 KM (rast 80% qoq)

**Izvor:** 02_distribucija_kanali.md

#### 3. Honorar autora

**P:** Koliki honorar dobijam ako objavim akademski udžbenik?

**Očekivan odgovor:** 8-12% od MOC po prodatoj knjizi za akademske/pravne naslove. Avans na potpis: 1.500-3.000 KM. Imprimatur dobija ekskluzivno izdavačko pravo na 5 godina (BCS jezik).

**Izvor:** 03_pravila_autori.md

#### 4. Pravo studenta na popust

**P:** Imam li popust kao student?

**Očekivan odgovor:** Da, 20% popust za akademske naslove uz važeći indeks. Aktivira se na web shop-u nakon verifikacije fotografije indeksa. Specifične cijene: Građansko pravo 95 KM (umjesto 125 KM) za studente.

**Izvori:** 04_FAQ.md + 01_katalog_knjiga.md

#### 5. Analiza CSV — kanal prodaje

**P:** Koji kanal najviše prodaje knjige u Q1 2026?

**Očekivan odgovor:** Univerzitetske knjižare (38% volumena, 45% marže) — najprodavaniji kanal. Web shop (22%, 65% marže) — najprofitabilniji. Buybook lanac (18%) — stabilan.

**Izvori:** 02_distribucija_kanali.md + 05_prodaja_q1.csv

### 5 dodatnih pitanja

6. Imate li audio knjige?
7. Mogu li dobiti popust za biblioteku?
8. Koliko traje proces objavljivanja od potpisa ugovora?
9. Šta vraćate ako reklamiram knjigu?
10. Koje izdavačke kuće distribuirate?

## Format prezentacije

### Primjer ROI argumenta za Imprimatur

Marketing odjel Imprimatura analizira prodaju svake nedjelje, priprema izvještaj za direktora. To traje 6 sati × 25 EUR/h = 150 EUR sedmično, ili 600 EUR mjesečno.

Sa RAG sistemom: direktor pita "Koji kanal je najbolji za AI knjige?" → odgovor sa brojevima za 3 sekunde. Marketing odjel oslobođen za prave kreativne aktivnosti.

**Ušteda: 450 EUR/mjesečno × 12 = 5.400 EUR godišnje. Plus bolja brzina odlučivanja (vrijednost teško kvantifikovati ali realna).**

**Cijena sistema: ~50 EUR/mjesečno. ROI x9.**

## Posebnost — Imprimatur ima dva tipa korisnika

RAG aplikacija može imati DVA system prompta:

1. **Za KUPCE knjiga** — fokus na cijene, dostupnost, popuste, dostavu
2. **Za AUTORE i izdavače** — fokus na proces, honorare, autorska prava

Možete demonstrirati prebacivanjem između system prompt-ova u istoj aplikaciji.
