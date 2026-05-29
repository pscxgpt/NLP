# NLP-I Task: ICD-10 Codification
## Deep Learning Baseline
Ernest Valveny (Ernest.Valveny@uab.cat)
Lei Kang (lkang@cvc.uab.cat)
Q2/1009
11/05/2026

### Task Formulation
- **Task type**: single-label multiclass classification.
- **Input**: one clinical literal from leaderboard_data.csv.
- **Output**: one ICD category label, stored as y_category.
- **Target definition**:
  - y_category = Code.astype(str).str[0]
- **Example**:
  - Code = "I420" -> y_category = "I"
  - Code = "3E03329" -> y_category = "3"
- **All categories**:
  - [0, 1, 2, …, 9, a, b, …, z]
