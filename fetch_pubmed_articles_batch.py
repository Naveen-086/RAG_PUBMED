from backend.fetch_pubmed import fetch_pubmed_articles
import json
import os

TOPICS = ["cancer", "covid-19", "heart attack"]
MAX_RESULTS = 50
all_articles = []

for topic in TOPICS:
    print(f"Fetching {MAX_RESULTS} articles for topic: {topic}")
    articles = fetch_pubmed_articles(topic, max_results=MAX_RESULTS)
    for article in articles:
        article["topic"] = topic
        all_articles.append(article)

os.makedirs("data/raw", exist_ok=True)
with open("data/raw/pubmed_raw.json", "w", encoding="utf-8") as f:
    json.dump(all_articles, f, indent=2, ensure_ascii=False)

print(f"Saved {len(all_articles)} articles to data/raw/pubmed_raw.json")
