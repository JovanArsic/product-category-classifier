import joblib
import os

# putanja do root projekta
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# putanja do modela
model_path = os.path.join(BASE_DIR, "models", "product_classifier.pkl")

# učitavanje (ovo je DICT)
data = joblib.load(model_path)

model = data["model"]
vectorizer = data["vectorizer"]

print("Product Category Predictor")
print("Upiši naziv proizvoda (ili 'exit' za izlaz)\n")

while True:
    text = input("Product: ")

    if text.lower() == "exit":
        print("Kraj programa.")
        break

    # 🔥 OBAVEZNO transform pre predict
    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)[0]

    print("Kategorija:", prediction)
    print("-" * 40)