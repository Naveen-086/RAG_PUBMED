from Bio import Entrez
import json, os

Entrez.email = "221501086@rajalakshmi.edu.in"

def fetch_pubmed_articles(query, max_results=50, out_file="data/raw/pubmed_raw.json"):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]

    results = []
    for pmid in ids:
        details = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="xml")
        details_record = Entrez.read(details)
        if "PubmedArticle" in details_record:
            article = details_record["PubmedArticle"][0]
            title = article["MedlineCitation"]["Article"]["ArticleTitle"]
            abstract = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [""])[0]
            results.append({"pmid": pmid, "title": title, "abstract": abstract})

    os.makedirs("data/raw", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    return results
