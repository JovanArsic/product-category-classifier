import joblib
import os

# putanja do root projekta (product-category-classifier)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# putanja do modela
model_path = os.path.join(BASE_DIR, "models", "product_classifier.pkl")

# učitavanje modela
model = joblib.load(model_path)

print("Product Category Predictor")
print("Upiši naziv proizvoda (ili 'exit' za izlaz)\n")

while True:
    text = input("Product: ")

    if text.lower() == "exit":
        print("Kraj programa.")
        break

    prediction = model.predict([text])[0]
    print("Kategorija:", prediction)
    print("-" * 40)