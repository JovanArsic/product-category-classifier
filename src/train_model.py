import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "products.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "product_classifier.pkl")


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# očisti nazive kolona
df.columns = df.columns.str.strip()

print("Kolone u datasetu:", df.columns)

# =========================
# CLEAN DATA (KRITIČNO)
# =========================
df = df.dropna(subset=["Product Title", "Category Label"])

# dodatno čišćenje praznih stringova
df = df[df["Product Title"].str.strip() != ""]
df = df[df["Category Label"].str.strip() != ""]


# =========================
# FEATURES + LABEL
# =========================
X = df["Product Title"].astype(str)
y = df["Category Label"].astype(str)


# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# TF-IDF VECTORIZER
# =========================
vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=20000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =========================
# MODEL
# =========================
model = LinearSVC()
model.fit(X_train_vec, y_train)


# =========================
# EVALUATION
# =========================
preds = model.predict(X_test_vec)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, preds))


# =========================
# SAVE MODEL
# =========================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump(
    {
        "model": model,
        "vectorizer": vectorizer
    },
    MODEL_PATH
)

print("\nModel uspešno sačuvan na:", MODEL_PATH)