import json
import os
import uuid
from datetime import datetime, timedelta
import feedparser
from time import mktime

def run_scraper():
    if not os.path.exists("data"):
        os.makedirs("data")
        
    rss_urls = [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "http://feeds.bbci.co.uk/news/business/rss.xml",
        "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.cnn.com/rss/edition.rss",
        "https://rss.cnn.com/rss/edition_technology.rss",
        "https://rss.cnn.com/rss/edition_space.rss",
        "https://rss.cnn.com/rss/edition_business.rss",
        "https://www.wired.com/feed/rss",
        "https://www.theverge.com/rss/index.xml",
        "https://techcrunch.com/feed/",
        "https://www.nasa.gov/rss/dyn/breaking_news.rss",
        "https://export.arxiv.org/rss/cs",
        "https://feeds.feedburner.com/oreilly/radar",
        "https://www.cnbc.com/id/10000311/device/rss/rss.html",
        "https://feeds.a.dj.com/rss/RSSWSJ.xml",
        "https://www.politico.com/rss/politicopicks.xml",
        "https://moxie.foxnews.com/feedburner/latest.xml",
        "https://www.abc.net.au/news/feed/51120/rss.xml",
        "https://feeds.feedburner.com/NDTV-LatestNews",
        "https://www.france24.com/en/rss"
    ]
    
    articles = []
    seen_urls = set()
    counter = 1
    
    one_week_ago = datetime.now() - timedelta(days=7)
    
    for url in rss_urls:
        
        if len(articles) >= 400:
            break
            
        print(f"\n{counter}. scraping {url}")
        print("\trequesting ...")
        try:
            feed = feedparser.parse(url)
            print("\tparsing ...")
            
            added_count = 0
            for item in feed.entries:
                if len(articles) >= 400:
                    break
                
                pub_date = datetime.now()
                if hasattr(item, "published_parsed") and item.published_parsed:
                    pub_date = datetime.fromtimestamp(mktime(item.published_parsed))
                
                if pub_date >= one_week_ago and item.link not in seen_urls:
                    seen_urls.add(item.link)
                    articles.append({
                        "uuid": str(uuid.uuid4()),
                        "url": item.link,
                        "date": pub_date.strftime("%Y-%m-%d"),
                        "headline": item.title,
                        "body": item.get("summary", item.title)
                    })
                    added_count += 1
            
            path = "data/articles.json"
            print(f"\tsaved in {path} (Added {added_count}. Total: {len(articles)})")
        except Exception as e:
            print(f"\t⚠️ Error: {e}")
            
        counter += 1
        
    with open("data/articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Done! Total unique articles from last week: {len(articles)}")

if __name__ == "__main__":
    run_scraper()