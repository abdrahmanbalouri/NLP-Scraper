#!/usr/bin/env python3
"""Scrape news articles from BBC News and save them in data/articles/."""

import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

DATA_DIR = "data"
ARTICLES_DIR = "data/articles"

SECTIONS = [
    "news", "news/business", "news/technology", "news/world",
    "news/entertainment_and_arts", "news/politics", "news/science_and_environment",
    "news/health", "sport",
    "news/uk", "news/uk/england", "news/uk/scotland", "news/uk/wales",
    "news/world/europe", "news/world/asia", "news/world/us_and_canada",
    "news/explainers", "news/in_pictures",
    "sport/football", "sport/cricket", "sport/formula1", "sport/tennis",
    "news/technology/innovation", "news/business/technology",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


def get(url):
    print("        requesting ...")
    return requests.get(url, headers=HEADERS, timeout=25).text


def get_headline(soup):
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else ""


def get_body(soup):
    """Join the text of all paragraphs inside text-block components."""
    paragraphs = []
    for block in soup.find_all(attrs={"data-component": "text-block"}):
        for p in block.find_all("p"):
            text = " ".join(p.get_text(" ", strip=True).split())
            if text:
                paragraphs.append(text)
    return "\n\n".join(paragraphs)


def get_date(soup):
    """Read datePublished from the JSON-LD embedded in the page."""
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script.get_text())
        except json.JSONDecodeError:
            continue
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and item.get("datePublished"):
                return item["datePublished"]
    return None


def scrape_article(url, week_ago, known_ids):
    """Scrape one article. Returns a dict or None."""
    print(f"scraping {url}")
    try:
        soup = BeautifulSoup(get(url), "lxml")
    except requests.RequestException as exc:
        print(f"        error: {exc}")
        return None

    headline, body, date = get_headline(soup), get_body(soup), get_date(soup)
    if not headline or not body:
        return None

    try:
        published = datetime.fromisoformat(date.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        published = None
    if published and published < week_ago:
        print("        older than one week, skipped")
        return None

    unique_id = str(abs(hash(url)))
    if unique_id in known_ids:
        return None

    return {
        "unique_id": unique_id,
        "url": url,
        "date": date,
        "headline": headline,
        "body": body,
        "date_scraped": datetime.now(timezone.utc).isoformat(),
    }


def save(article):
    """Append the article to the JSON file of today."""
    os.makedirs(ARTICLES_DIR, exist_ok=True)
    path = os.path.join(ARTICLES_DIR, f"articles_{datetime.now().strftime('%Y-%m-%d')}.json")

    articles = []
    if os.path.exists(path):
        articles = json.load(open(path))
    if not any(a["unique_id"] == article["unique_id"] for a in articles):
        articles.append(article)
    json.dump(articles, open(path, "w"), indent=2)
    print(f"        saved in {path}")
    return path


def main():
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)

    known_ids = set()
    for f in os.listdir(ARTICLES_DIR) if os.path.isdir(ARTICLES_DIR) else []:
        for a in json.load(open(os.path.join(ARTICLES_DIR, f))):
            known_ids.add(a["unique_id"])

    # collect all article URLs from the section pages
    urls = set()
    for section in SECTIONS:
        print(f"scraping {section}")
        try:
            soup = BeautifulSoup(get(f"https://www.bbc.com/{section}"), "lxml")
        except requests.RequestException as exc:
            print(f"        error: {exc}")
            continue
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/articles/" in href and href.startswith(("/news/", "/sport/")):
                urls.add("https://www.bbc.com" + href.split("?")[0])
        time.sleep(1)

    print(f"\n{len(urls)} article URLs found\n")

    count = 0
    for url in sorted(urls):
        if count >= 300:
            break
        article = scrape_article(url, week_ago, known_ids)
        if article:
            save(article)
            known_ids.add(article["unique_id"])
            count += 1
            print(f"        -> {count} articles stored so far")
        time.sleep(1)

    print(f"\nDone. {count} new articles stored in {ARTICLES_DIR}")


if __name__ == "__main__":
    main()
