#  Klasifikacija proizvoda po kategorijama (ML projekat)

##  Opis projekta

Ovaj projekat predstavlja sistem mašinskog učenja koji automatski klasifikuje nazive proizvoda u odgovarajuće kategorije.

Na primer:
- "iPhone 7 32GB Gold" → Mobile Phones
- "Bosch Serie 4 KGV39VL31G" → Dishwashes

Cilj projekta je da se automatizuje proces kategorizacije proizvoda u e-commerce sistemima i smanji potreba za ručnim razvrstavanjem.

---

##  Problem koji rešava

U online prodavnicama svakodnevno se dodaju hiljade novih proizvoda.

Ručna klasifikacija:
- traje dugo
- sklona je greškama
- ne može da skalira

Ovaj projekat rešava taj problem pomoću NLP i ML tehnika.

---

## Dataset

Dataset sadrži preko 30.000 proizvoda sa sledećim kolonama:

- Product Title (naziv proizvoda)
- Category Label (ciljna kategorija)
- Merchant ID
- Product Code
- Number of Views
- Merchant Rating
- Listing Date

 Kao ulaz za model koristi se samo **Product Title**.

---

## Pristup rešenju

### Obrada teksta:
- pretvaranje u mala slova
- TF-IDF vektorizacija

### Model:
- Linear Support Vector Classifier (LinearSVC)

### Pipeline:
TF-IDF → LinearSVC

---

## Rezultati

- Tačnost modela: ~96%
- Dobro razlikuje većinu kategorija
- Manje greške se javljaju kod sličnih kategorija (npr. frižideri i zamrzivači)

---

## Pokretanje projekta

### 1. Instalacija zavisnosti
```bash
pip install -r requirements.txt
