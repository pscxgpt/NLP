NLP-I Task: ICD-10 Codification
Deep Learning Baseline
Ernest Valveny (Ernest.Valveny@uab.cat)
Lei Kang (lkang@cvc.uab.cat)
Q2/1009
11/05/2026

Task Formulation
● Task type: single-label multiclass classification.
● Input: one clinical literal from leaderboard_data.csv.
● Output: one ICD category label, stored as y_category.
● Target definition:
○ y_category = Code.astype(str).str[0]
● Example:
○ Code = "I420" -> y_category = "I"
○ Code = "3E03329" -> y_category = "3"
● All categories:
○ [0, 1, 2, …, 9, a, b, …, z]

Data Pre-processing
Training file: codification_data.csv.
Test file: leaderboard_data.csv.
Required training columns: Code, Literal
Required test columns: id, Literal, y_category
Text preprocessing is intentionally light:
text = str(text)
text = text.strip()
text = re.sub(r"\s+", " ", text)
I do not lowercase text, not remove accents, not remove punctuation.
Because: I will use a pretrained tokenizer that can handle Spanish biomedical text directly.

Data Preparation
Build label classes from the training set: (number of classes: 36, [0, 1, 2, …, 9, a, b, …, z])
labels = sorted(train_df["y_category"].unique())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
Train/validation split: train: validation == 80: 20, and split type: stratified, so class proportions are
approximately preserved.
from sklearn.model_selection import train_test_split
train_split, val_split = train_test_split(
train_df,
test_size=0.2,
random_state=args.seed,
stratify=train_df["label_id"],
)

Backbone Model
PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
Architecture: RoBERTa-base style encoder
pretrained with masked language modeling on Spanish biomedical and clinical corpora
(more than 1B tokens, including biomedical crawled text, clinical cases, Spanish clinical
notes/documents, SciELO, BARR2, Wikipedia life sciences, patents, EMEA, Medline, and
PubMed.)
● Backbone type: roberta
● Transformer layers: 12
● Hidden feature size: 768
● Attention heads: 12
● Feed-forward intermediate size: 3072
● Number of parameters: 126M

Fine-tuning
Aggregation: <CLS> or Mean
● CLS aggregation: features = hidden_states[:, 0, :]
For RoBERTa, the first token is <s>, which plays the same role as BERT's [CLS].
● Mean pooling:
mask = attention_mask.unsqueeze(-1) # ignores padding tokens
features = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
Classification Head
● A dropout layer followed by a linear layer:
Dropout(0.1)
Linear(768, 36)
Output shape:
● batch_size x num_classes
● Each row gets one predicted class.

Training Pipeline
|     | Hyperparameters | Value |
| --- | --------------- | ----- |
  1. Train on the training split.
|     | Max epochs | 50  |
| --- | ---------- | --- |
  2. Compute training loss (nn.CrossEntropyLoss(), torch.optim.AdamW).
|     | Patience  | 10  |
| --- | --------- | --- |
  3. Evaluate on the validation split.
|   4. Compute validation accuracy. | Batch size  | 128 |
| --------------------------------- | ----------- | --- |
  5. Save the best model based on validation accuracy. Learning rate 2e-5
  6. Early stopping if validation accuracy does not improve for 10 epochs. Weight decay 0.01
|     | Dropout  | 0.1 |
| --- | -------- | --- |

|                      |            | Method | Acc   |
| -------------------- | ---------- | ------ | ----- |
| Results              |            | <CLS>  | 0.565 |
| leaderboard_data.csv | 10/05/2026 | Mean   | 0.570 |
https://www.kaggle.com/competitions/uab-asho-ai-codification
