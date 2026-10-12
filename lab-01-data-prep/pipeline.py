"""
Lab 1 — Frontier data prep pipeline.
Load -> MinHash dedup -> Presidio PII scan -> BPE tokenizer training.
Run: pip install -r requirements.txt && python pipeline.py
"""
import re
from collections import Counter

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from presidio_analyzer import AnalyzerEngine
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

N = 5000  # stories to process (bump it once this works)

# ---------- 1. LOAD ----------
print("Loading TinyStories...")
ds = load_dataset("roneneldan/TinyStories", split=f"train[:{N}]")
texts = [t for t in ds["text"] if t and t.strip()]
print(f"  loaded {len(texts)} stories\n")

# ---------- 2. DEDUP (MinHash) ----------
print("Running MinHash near-dedup...")
def shingles(text, k=5):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}

lsh = MinHashLSH(threshold=0.85, num_perm=128)
dupes = 0
kept = []
for idx, text in enumerate(texts):
    mh = MinHash(num_perm=128)
    for s in shingles(text):
        mh.update(s.encode("utf8"))
    if lsh.query(mh):
        dupes += 1
        continue
    lsh.insert(idx, mh)
    kept.append(text)
print(f"  duplicates removed: {dupes} ({dupes / len(texts):.1%})")
print(f"  kept: {len(kept)}\n")

# ---------- 3. PII SCAN (Presidio) ----------
print("Scanning sample for PII with Presidio...")
analyzer = AnalyzerEngine()
sample = kept[:200]
entity_counts = Counter()
for text in sample:
    for r in analyzer.analyze(text=text, language="en"):
        entity_counts[r.entity_type] += 1
print("  PII entities found in 200-story sample:")
for ent, cnt in entity_counts.most_common():
    print(f"    {ent}: {cnt}")
if not entity_counts:
    print("    none found (note this in your datasheet — why?)")
print()

# ---------- 4. BPE TOKENIZER ----------
print("Training byte-level BPE tokenizer...")
tok = Tokenizer(BPE(unk_token="[UNK]"))
tok.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=8000, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
tok.train_from_iterator(kept, trainer)
tok.save("tinystories-bpe.json")
print("  saved tinystories-bpe.json")
for probe in ["the quick brown fox", "supercalifragilisticexpialidocious", "AI safety"]:
    ids = tok.encode(probe).ids
    print(f"  '{probe}' -> {len(ids)} tokens")
print("\nDone. Next: fill in DATASHEET.md.")
