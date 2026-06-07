import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def load_and_clean_data(file_path):
    print(f"Loading {file_path}...")
    try:
        df = pd.read_csv(file_path, encoding='latin-1')

        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')

        df = df.rename(columns={'v1': 'label', 'v2': 'message'})

        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

        df = df.dropna()

        return df
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Please check the path.")
        return None


def evaluate_model(name, model, X_test_tfidf, y_test):
    print(f"\n======================================")
    print(f"--- {name} ---")
    start_time = time.time()

    y_pred = model.predict(X_test_tfidf)

    print(f"Prediction time: {time.time() - start_time:.4f} seconds.")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    print("\nConfusion Matrix:")

    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))


def main():
    data_path = r"D:\Downloads\archive (5)\spam.csv"

    df = load_and_clean_data(data_path)
    if df is None:
        return

    print(f"\nDataset loaded. Total messages: {len(df)}")
    print(f"Spam messages: {df['label'].sum()}")
    print(f"Ham messages: {len(df) - df['label'].sum()}")

    X = df['message']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nVectorizing text data using TF-IDF...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"Vocabulary size: {X_train_tfidf.shape[1]} words.")

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(random_state=42),
        "Support Vector Machine": LinearSVC(random_state=42, dual=False)
    }

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        evaluate_model(name, model, X_test_tfidf, y_test)


if __name__ == "__main__":
    main()