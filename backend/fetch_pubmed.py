import requests
from bs4 import BeautifulSoup
import time
import json

def get_pubmed_articles(query, retmax=2000, topic="general"):
    # Step 1: Search for PMIDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json"}
    search_res = requests.get(search_url, params=params).json()
    pmids = search_res["esearchresult"]["idlist"]

    articles = []
    
    for idx, pmid in enumerate(pmids):
        # Step 2: Fetch details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        xml_data = requests.get(fetch_url, params=params).text

        # Step 3: Parse XML
        soup = BeautifulSoup(xml_data, "xml")
        title = soup.find("ArticleTitle").text if soup.find("ArticleTitle") else ""
        abstract = " ".join([a.text for a in soup.find_all("AbstractText")]) if soup.find("AbstractText") else ""

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "topic": topic,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        })

        # polite delay (avoid hammering PubMed servers)
        time.sleep(0.3)

        # Debug progress
        if (idx+1) % 100 == 0:
            print(f"Fetched {idx+1}/{len(pmids)} for topic '{topic}'")

    return articles


if __name__ == "__main__":
    final_data = []

    # Fetch ~666 per topic
    topics = {
        "heart attack": "heart attack",
        "covid-19": "covid-19",
        "cancer": "cancer"
    }

    for topic, query in topics.items():
        print(f"Fetching data for {topic}...")
        articles = get_pubmed_articles(query, retmax=666, topic=topic)
        final_data.extend(articles)

    # Save as JSON
    with open("data/raw/fetch_pubmed.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(final_data)} articles into pubmed_dataset.json")

