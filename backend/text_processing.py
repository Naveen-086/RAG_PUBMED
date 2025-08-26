import json, os
from pathlib import Path

def clean_and_chunk(in_file="data/raw/pubmed_raw.json", out_file="data/processed/pubmed_clean.json", chunk_size=200):
    with open(in_file, "r") as f:
        articles = json.load(f)

    processed = []
    for art in articles:
        text = f"{art['title']} {art['abstract']}".strip()
        # Semantic chunking (sentence based)
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            processed.append({
                "pmid": art["pmid"],
                "chunk": chunk
            })

    os.makedirs("data/processed", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(processed, f, indent=2)

    return processed
