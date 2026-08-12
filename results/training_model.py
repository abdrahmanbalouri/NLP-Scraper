import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve

DATA_DIR = "data"
RESULTS_DIR = "results"


def main():
    train = pd.read_csv(f"{DATA_DIR}/bbc_news_train.csv")[["Text", "Category"]]
    test = pd.read_csv(f"{DATA_DIR}/bbc_news_tests.csv")[["Text", "Category"]]

    X_train, y_train = train["Text"].values, train["Category"].values
    X_test, y_test = test["Text"].values, test["Category"].values

    vectorizer = TfidfVectorizer(stop_words="english")
    clf = LogisticRegression(max_iter=2000, C=10)

    print("Training the topic classifier ...")
    X_train_vec = vectorizer.fit_transform(X_train)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(vectorizer.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred))

    # learning curves: train vs cross-validation score
    sizes, train_scores, valid_scores = learning_curve(
        clf, X_train_vec, y_train, cv=5, train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
        scoring="accuracy",
    )
    plt.plot(sizes, train_scores.mean(axis=1), "o-", label="Training score")
    plt.plot(sizes, valid_scores.mean(axis=1), "o-", label="Cross-validation score")
    plt.title("Learning curves - topic classifier")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/learning_curves.png")
    print(f"Learning curves saved in {RESULTS_DIR}/learning_curves.png")

    pickle.dump(
        {"vectorizer": vectorizer, "classifier": clf, "categories": sorted(set(y_train))},
        open(f"{RESULTS_DIR}/topic_classifier.pkl", "wb"),
    )
    print(f"Model saved in {RESULTS_DIR}/topic_classifier.pkl")


if __name__ == "__main__":
    main()
