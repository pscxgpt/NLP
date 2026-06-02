# ASHO & UAB Clinical Codification NLP Challenge
**Project Overview & Deep Learning Strategy**

---

## 1. The Core Objective

The challenge focuses on automating a highly manual but critical healthcare administrative task: translating medical text into standardized international codes (ICD-10). 

While ASHO's full automated coding software deals with complex, multi-page medical records, this specific project is cleanly scoped. Instead of building a model to predict full 7-character ICD-10 codes (e.g., `0DTJ4ZZ`), **your goal is to build a classifier that predicts only the first alphanumeric character of the ICD-10 code based on a short medical text snippet (a "literal").**

This first character represents the broad category or chapter of the classification system:
* **Letters (A-Z):** Represent Diagnoses (e.g., A and B for infectious diseases, C for neoplasms) and a few specialized procedures.
* **Numbers (0-9):** Represent Medical, Surgical, and Obstetrical Procedures.

---

## 2. The Provided Data

The project utilizes two main data structures to train the model:

* **Literals Dataset:** A collection of specific, real-world medical phrases (e.g., *"golpe contra muro"*, *"sangrado uterino anómalo"*, *"shock septico"*) paired with their ground truth ICD-10 codes.
* **Official ICD-10 Pairs:** A reference dictionary mapping official codes to their formal descriptions and classifying them as either Diagnoses (`D`) or Procedures (`P`).

---

## 3. The Baseline Implementation

The presentation outlines a starting point using standard machine learning libraries. The provided baseline is a Scikit-Learn `Pipeline` that extracts features using two parallel TF-IDF vectorizers:

1.  **Character-level Vectorizer:** Extracts n-grams of 3 to 5 characters, which is excellent for catching medical root words and misspellings.
2.  **Word-level Vectorizer:** Extracts single words and pairs of words.

These text features are then fed into a standard `LogisticRegression` classifier to make the prediction.

---

## 4. The Kaggle Evaluation

The project is structured competitively on Kaggle. Model performance is evaluated against two distinct test sets:

* **Public Leaderboard:** Calculated on a randomly selected 30% subset of the test data (2,000 literals). This provides real-time feedback during the tuning process.
* **Private Leaderboard:** Calculated on the remaining 70% (4,667 literals). This hidden dataset determines the final standings and ensures models haven't simply memorized the public test set.

---

## 5. Deep Learning Strategy & Next Steps

While the provided TF-IDF baseline is functional, the most practical way to dominate this 36-class prediction task is to transition to a deep learning pipeline. Moving away from theoretical baselines to a hands-on neural network implementation will yield significant performance gains.

### Architectural Blueprint (PyTorch)

Setting up a custom text classification pipeline in PyTorch allows for far more nuanced text representation than TF-IDF. 

* **Target Encoding:** Map the alphanumeric characters (`0-9`, `A-Z`) into integer indices (`0` to `35`).
* **Text Representation:** * *Intermediate:* Swap TF-IDF for trainable Word/Character Embeddings passed through a Convolutional Neural Network (CNN) or LSTM layer to capture sequential context.
    * *Advanced:* Fine-tune a lightweight, pre-trained NLP model (like a Spanish BERT variant, such as BETO). Transformer models are exceptionally good at understanding the semantic nuances of clinical jargon.
* **The Output Layer:** The final fully connected layer (`nn.Linear`) must be explicitly set to `out_features=36` to output the logits for each possible character.
* **Loss Function:** Standard `nn.CrossEntropyLoss()` is the optimal choice for evaluating the probability distribution across the 36 mutually exclusive classes.
