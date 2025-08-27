import json, os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_and_chunk(
    in_file="data/raw/pubmed_raw.json", 
    out_file="data/processed/pubmed_clean.json", 
    chunk_size=500, 
    chunk_overlap=50
):
    # Load input file
    with open(in_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    # Define recursive text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""], 
    )

    processed = []
    for art in articles:
        text = f"{art['title']} {art['abstract']}".strip()

        # Perform semantic chunking
        chunks = splitter.split_text(text)

        for chunk in chunks:
            processed.append({
                "pmid": art["pmid"],
                "chunk": chunk
            })

    # Save processed chunks
    os.makedirs("data/processed", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    return processed
