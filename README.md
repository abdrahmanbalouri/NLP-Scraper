# NLP Scraper

News Intelligence platform: scrapes news articles from BBC News and enriches
them with NLP (entities detection, topic detection, sentiment analysis and
scandal detection).

## Project structure

```
.
├── data
│   ├── articles.json            articles stored by the scraper
│   ├── bbc_news_train.csv       labelled dataset used to train the topic model
│   └── bbc_news_tests.csv       labelled dataset used to evaluate the topic model
├── scraper_news.py              the news scraper
├── nlp_enriched_news.py         the NLP engine
├── requirements.txt
├── README.md
└── results
    ├── training_model.py        trains and saves the topic classifier
    ├── topic_classifier.pkl     the trained topic classifier
    ├── learning_curves.png      learning curves of the topic classifier
    └── enhanced_news.csv        the enriched articles
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# models used by the NLP engine
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download(['vader_lexicon', 'stopwords'])"
```

`sentence-transformers` downloads the `all-MiniLM-L6-v2` model automatically on
first use.

## Usage

1. Scrape the news (stores at least 300 articles from the last week in
   `data/articles/`):

```bash
python scraper_news.py
```

```
scraping https://www.bbc.com/news/articles/c0qv2nn1gpeo
        requesting ...
        parsing ...
        saved in data/articles/articles_2026-08-12.json
```

2. Train the topic classifier and produce the learning curves:

```bash
python results/training_model.py
```

The script saves `results/topic_classifier.pkl` and
`results/learning_curves.png`. Test accuracy is 98% (requirement: > 95%).

3. Enrich the articles and produce `results/enhanced_news.csv`:

```bash
python nlp_enriched_news.py
```

```
Enriching <URL>:

---------- Detect entities ----------
Detected 3 companies which are Thames Water, BBC, The Home Office

---------- Topic detection ----------
The topic of the article is: Business

---------- Sentiment analysis ----------
The article <headline> has a negative sentiment

---------- Scandal detection ----------
Computing embeddings and distance ...
Environmental scandal detected for Thames Water (distance 0.42)
```

## How each NLP step works

### Entities detection
SpaCy NER (`en_core_web_sm`) runs on the article body. The `ORG` entities
(companies and organisations) are kept and deduplicated.

### Topic detection
A TF-IDF vectoriser + Logistic Regression classifier is trained on the labelled
BBC dataset. The whole pipeline is saved in `topic_classifier.pkl` and reaches
98% accuracy on the test set. The learning curves (`learning_curves.png`) show
that the training score and the cross-validation score converge, with a small
gap, which proves the model is correctly trained and not overfitted.

### Sentiment analysis
The pre-trained NLTK VADER model scores the article body. The compound score is
stored and mapped to `positive`, `negative` or `neutral`.

### Scandal detection
1. A list of non-ambiguous keywords related to environmental disasters is
   defined: `oil spill`, `deforestation`, `toxic waste`, `chemical pollution`,
   `radioactive contamination`, `greenhouse gas emissions`, `carbon emissions`,
   `plastic pollution`, `wastewater discharge`, `habitat destruction`,
   `illegal dumping`. Single ambiguous words such as `fire` or `gas` are
   avoided to prevent false positives.
2. The keywords are embedded with `sentence-transformers`
   (`all-MiniLM-L6-v2`). **Why this embedding**: it is a small and fast
   sentence embedding model that captures the semantic meaning of a whole
   sentence in 384 dimensions, while being light enough to run on any machine.
3. Every sentence of the article that contains a detected entity is embedded
   the same way.
4. **Why cosine distance**: the sentence embeddings are normalised, so the
   cosine distance (`1 - cosine similarity`) lies in `[0, 2]` and focuses on
   the *direction* of the vectors rather than their length, which works well
   with sentence embeddings.
5. The per-article metric is the **minimum** cosine distance over all
   (sentence, keyword) pairs: it unifies all the distances computed for the
   article into a single number. The closer to 0, the more the article is
   related to an environmental disaster.
6. The 10 articles with the smallest `scandal_distance` are flagged with
   `Top_10 = True`.

## Output

`results/enhanced_news.csv` contains one row per article with the columns:

| Column            | Type        |
|-------------------|-------------|
| Unique ID         | str (uuid)  |
| URL               | str         |
| Date scraped      | date        |
| Headline          | str         |
| Body              | str         |
| Org               | list of str |
| Topics            | list of str |
| Sentiment         | float       |
| Scandal_distance  | float       |
| Top_10            | bool        |
