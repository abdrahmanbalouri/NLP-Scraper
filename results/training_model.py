import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
import joblib

train_df = pd.read_csv("data/bbc_news_train.csv")
test_df = pd.read_csv("data/bbc_news_tests.csv")

X_train, y_train = train_df["Text"], train_df["Category"]
X_test, y_test = test_df["Text"], test_df["Category"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

print("Training model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc * 100:.2f}%")

os.makedirs("results", exist_ok=True)
joblib.dump(pipeline, "results/topic_classifier.pkl")

print("Computing learning curves...")
X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])

train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X_all, y_all, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1),
val_std = val_scores.std(axis=1)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, train_mean, "o-", color="royalblue", label="Training score")
plt.plot(train_sizes, val_mean, "o-", color="darkorange", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="royalblue")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="darkorange")

plt.title("Learning Curves — Topic Classifier (TF-IDF + Logistic Regression)")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.ylim(0.5, 1.05)
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("results/learning_curves.png", dpi=150)
print("Learning curves saved successfully!")