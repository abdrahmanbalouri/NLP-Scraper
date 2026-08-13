import json
import os
import joblib
import numpy as np
import pandas as pd
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_md")
sid = SentimentIntensityAnalyzer()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

KEYWORDS = [
    "oil spill",
    "deforestation",
    "toxic waste",
    "chemical pollution",
    "radioactive contamination",
    "greenhouse gas emissions",
    "carbon emissions",
    "plastic pollution",
    "wastewater discharge",
    "habitat destruction",
    "illegal dumping",
]

KEYWORD_EMBEDDINGS = embedder.encode(KEYWORDS, convert_to_tensor=True)


def detect_entities(doc):
    return list(dict.fromkeys([ent.text for ent in doc.ents if ent.label_ == "ORG"]))


def scandal_distance(doc, orgs):
    sentences = [sent.text for sent in doc.sents if any(o in sent.text for o in orgs)]
    if not sentences:
        return None
    sent_embeddings = embedder.encode(sentences, convert_to_tensor=True)
    cosine_sim = util.cos_sim(sent_embeddings, KEYWORD_EMBEDDINGS)
    return float((1 - cosine_sim).min())


def sentiment_label(compound):
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def process():
    if not os.path.exists("results/topic_classifier.pkl") or not os.path.exists("data/articles.json"):
        print("Run scraper and training first!")
        return

    pipeline = joblib.load("results/topic_classifier.pkl")

    with open("data/articles.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    results = []

    for art in articles:
        text = f"{art['headline']} {art['body']}"
        url = art["url"]
        print(f"\nEnriching {url}:\n")

        print("---------- Detect entities ----------")
        doc = nlp(text)
        orgs = detect_entities(doc)
        print(f"Detected {len(orgs)} companies which are {', '.join(orgs) or 'none'}")

        print("\n---------- Topic detection ----------")
        topic = pipeline.predict([text])[0]
        print(f"The topic of the article is: {topic}")

        print("\n---------- Sentiment analysis ----------")
        compound = sid.polarity_scores(text)["compound"]
        label = sentiment_label(compound)
        print(f"The article {art['headline']!r} has a {label} sentiment")

        print("\n---------- Scandal detection ----------")
        print("Computing embeddings and distance ...")
        distance = scandal_distance(doc, orgs)
        if distance is None:
            print("No sentence containing an entity was found")
        else:
            print(f"Environmental scandal detected for {orgs[0]} (distance {distance:.3f})")

        results.append({
            "uuid": art["uuid"],
            "url": url,
            "date": art["date"],
            "headline": art["headline"],
            "body": art["body"],
            "Org": orgs,
            "Topics": [topic],
            "Sentiment": float(compound),
            "Scandal_distance": distance,
            "Top_10": False,
        })

    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values(by="Scandal_distance", na_position="last")
        df.loc[df["Scandal_distance"].notna().index[:10], "Top_10"] = True

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/enhanced_news.csv", index=False)
    print("\nEnhanced news saved to results/enhanced_news.csv!")


if __name__ == "__main__":
    process()
