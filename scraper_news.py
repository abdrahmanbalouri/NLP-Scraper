import json
import os
import uuid
from datetime import datetime
import feedparser

def run_scraper():
    if not os.path.exists("data"):
        os.path.makedirs("data")
        
    articles = []
    feed = feedparser.parse("http://feeds.bbci.co.uk/news/rss.xml")
    
    for item in feed.entries:
        articles.append({
            "uuid": str(uuid.uuid4()),
            "url": item.link,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "headline": item.title,
            "body": item.get("summary", item.title)
        })
        
    with open("data/articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
        
    print(f"Done! Scraped {len(articles)} articles.")

if __name__ == "__main__":
    run_scraper()