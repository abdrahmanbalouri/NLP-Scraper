#!/usr/bin/env python3
"""Enrich the scraped news with NLP and save results/enhanced_news.csv.

Usage:
    python nlp_enriched_news.py
"""

import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

SCANDAL_KEYWORDS = [
    "oil spill", "deforestation", "toxic waste", "chemical pollution",
    "radioactive contamination", "greenhouse gas emissions", "carbon emissions",
    "plastic pollution", "wastewater discharge", "habitat destruction",
    "illegal dumping",
]


def get_articles():
    articles = []
    for path in glob.glob("data/articles/*.json"):
        articles += json.load(open(path))
    return articles


def detect_entities(text, nlp):
    return list(dict.fromkeys(
        e.text.strip() for e in nlp(text).ents if e.label_ == "ORG"
    ))


def detect_topic(text, model):
    return model["classifier"].predict(model["vectorizer"].transform([text]))[0].title()


def get_sentiment(text, vader):
    compound = vader.polarity_scores(text)["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return compound, label


def scandal_distance(text, entities, keywords, embedder):
    """Min cosine distance between entity sentences and scandal keywords."""
    sentences = [s.strip() for s in text.split(".") if any(e in s for e in entities)]
    if not sentences:
        return np.nan
    emb_sent = embedder.encode(sentences, normalize_embeddings=True)
    distances = 1 - emb_sent @ keywords.T
    return float(distances.min())


def main():
    import spacy
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sentence_transformers import SentenceTransformer

    nlp = spacy.load("en_core_web_sm")
    vader = SentimentIntensityAnalyzer()
    model = pickle.load(open("results/topic_classifier.pkl", "rb"))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    keywords = embedder.encode(SCANDAL_KEYWORDS, normalize_embeddings=True)

    rows = []
    for article in get_articles():
        url, headline, body = article["url"], article["headline"], article["body"]

        print(f"\nEnriching {url}")

        print("\n---------- Detect entities ----------")
        orgs = detect_entities(body, nlp)
        print(f"Detected {len(orgs)} companies: {', '.join(orgs[:3])}")

        print("\n---------- Topic detection ----------")
        topic = detect_topic(headline + " " + body, model)
        print(f"The topic of the article is: {topic}")

        print("\n---------- Sentiment analysis ----------")
        compound, sentiment = get_sentiment(body, vader)
        print(f"The article '{headline[:50]}' has a {sentiment} sentiment")

        print("\n---------- Scandal detection ----------")
        distance = scandal_distance(body, orgs, keywords, embedder)
        print(f"Scandal distance: {distance:.3f}")

        rows.append({
            "Unique ID": article["unique_id"],
            "URL": url,
            "Date scraped": article["date_scraped"],
            "Headline": headline,
            "Body": body,
            "Org": orgs,
            "Topics": topic,
            "Sentiment": compound,
            "Scandal_distance": distance,
            "Top_10": False,
        })

    df = pd.DataFrame(rows)
    top = df["Scandal_distance"].dropna().nsmallest(10).index
    df.loc[top, "Top_10"] = True

    for _, row in df[df["Top_10"]].iterrows():
        print(f"\nEnvironmental scandal detected for {row['Org']}")

    df.to_csv("results/enhanced_news.csv", index=False)
    print(f"\nEnriched news saved in results/enhanced_news.csv")


if __name__ == "__main__":
    main()
