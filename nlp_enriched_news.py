import json
import os
import joblib
import pandas as pd
import spacy
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

def process():
    if not os.path.exists("topic_classifier.pkl") or not os.path.exists("data/articles.json"):
        print("Run scraper and training first!")
        return

    model, vectorizer = joblib.load("topic_classifier.pkl")
    
    with open("data/articles.json", "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    results = []
    keywords = ["pollution", "spill", "toxic", "disaster", "waste"]
    
    for art in articles:
        text = str(art['headline']) + " " + str(art['body'])
        doc = nlp(text)
        
        orgs = list(set([ent.text for ent in doc.ents if ent.label_ == "ORG"]))
        
        topic = model.predict(vectorizer.transform([text]))[0]
        
        sentiment = TextBlob(text).sentiment.polarity
        
        score = sum(1.0 for w in keywords if w in text.lower())
        
        results.append({
            "uuid": art["uuid"],
            "url": art["url"],
            "date": art["date"],
            "headline": art["headline"],
            "body": art["body"],
            "Org": orgs,
            "Topics": [topic],
            "Sentiment": float(sentiment),
            "Scandal_distance": float(score),
            "Top_10": False
        })
        
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Scandal_distance", ascending=False)
        top_indices = df.head(10).index
        df.loc[top_indices, "Top_10"] = True
        
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/enhanced_news.csv", index=False)
    print("Enhanced news saved to results/enhanced_news.csv!")

if __name__ == "__main__":
    process()