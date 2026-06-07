import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import time


def load_kaggle_movie_data(file_path):
    print(f"Loading {file_path}...")
    try:
        df = pd.read_csv(
            file_path,
            sep=' ::: ',
            engine='python',
            names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
        )
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Please check the path.")
        return None


def main():
    train_data_path = r"D:\Downloads\archive (3)\Genre Classification Dataset\train_data.txt"
    test_solution_path = r"D:\Downloads\archive (3)\Genre Classification Dataset\test_data_solution.txt"

    df_train = load_kaggle_movie_data(train_data_path)
    df_test = load_kaggle_movie_data(test_solution_path)

    if df_train is None or df_test is None:
        return

    X_train_text = df_train['DESCRIPTION']
    y_train = df_train['GENRE']

    X_test_text = df_test['DESCRIPTION']
    y_test = df_test['GENRE']

    print("\nVectorizing text data using TF-IDF...")
    start_time = time.time()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

    print(f"Vectorization complete in {time.time() - start_time:.2f} seconds.")
    print(f"Vocabulary size: {X_train_tfidf.shape[1]} words.")

    models = {
        "Naive Bayes (Multinomial)": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine (LinearSVC)": LinearSVC(random_state=42, dual=False)
    }

    for name, model in models.items():
        print(f"\n======================================")
        print(f"Training {name}...")
        start_time = time.time()

        model.fit(X_train_tfidf, y_train)
        print(f"Training finished in {time.time() - start_time:.2f} seconds.")

        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

        print("\nClassification Report (Summary):")
        print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()