import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def clean_and_engineer(df):
    df = df.copy()

    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_day_of_week'] = df['trans_date_trans_time'].dt.dayofweek

    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'])
        df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

    cols_to_drop = [
        'Unnamed: 0', 'cc_num', 'merchant', 'first', 'last',
        'street', 'city', 'state', 'zip', 'trans_num',
        'unix_time', 'trans_date_trans_time', 'dob'
    ]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    return df


def evaluate_model(name, model, X_test, y_test):
    print(f"\n--- {name} ---")
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def main():
    train_data_path = r"D:\Downloads\archive (4)\fraudTrain.csv"
    test_data_path = r"D:\Downloads\archive (4)\fraudTest.csv"

    print("Loading datasets...")
    try:
        df_train = pd.read_csv(train_data_path)
        df_test = pd.read_csv(test_data_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print("Cleaning and engineering features...")
    df_train = clean_and_engineer(df_train)
    df_test = clean_and_engineer(df_test)

    X_train = df_train.drop(columns=['is_fraud'])
    y_train = df_train['is_fraud']
    X_test = df_test.drop(columns=['is_fraud'])
    y_test = df_test['is_fraud']

    print("Encoding categorical variables...")
    categorical_cols = ['category', 'gender', 'job']

    encoders = {}
    for col in categorical_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))

            test_classes = X_test[col].astype(str)
            test_classes_mapped = test_classes.map(lambda s: '<unknown>' if s not in le.classes_ else s)
            le.classes_ = np.append(le.classes_, '<unknown>')
            X_test[col] = le.transform(test_classes_mapped)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=50, max_depth=10, random_state=42,
                                                n_jobs=-1)
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        evaluate_model(name, model, X_test_scaled, y_test)


if __name__ == "__main__":
    main()