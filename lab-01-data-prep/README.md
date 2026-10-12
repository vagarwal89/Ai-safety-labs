# Lab 1 — Frontier Data Prep

**Goal:** Build the pipeline that turns raw text into training-ready data, the way frontier labs do it: filter → dedup → PII-scrub → tokenize.

## What you'll run
`pipeline.py` does four things on a slice of TinyStories:
1. **Load** — pulls 5,000 stories via HuggingFace `datasets`.
2. **Dedup** — MinHash near-duplicate detection (`datasketch`); reports duplicate rate.
3. **PII scan** — Microsoft Presidio finds PERSON/LOCATION/PHONE etc.; reports entity counts.
4. **Tokenize** — trains a byte-level BPE tokenizer (`tokenizers`) on the cleaned text; reports vocab stats.

## Run it (Colab, free T4)
```
pip install -r requirements.txt
python pipeline.py
```

## What to observe
- The duplicate rate. Frontier labs remove 10%+ of web data as near-dups. Why does dedup matter for safety? (Memorization. You'll prove this in Lab 9.)
- What Presidio catches — and what it misses. That gap is a governance finding; write it down.
- How BPE handles rare words vs common ones. Tokenization shapes everything downstream.

## Deliverable
Fill in `DATASHEET.md` (template in this folder) for your processed dataset: why it exists, what's in it, what you removed, known limitations. This is the data-governance artifact — Gebru et al.'s "Datasheets for Datasets" in practice.

## Interview version (write yours after the lab)
- What you did: _
- What you found: _
- Why it matters: _
