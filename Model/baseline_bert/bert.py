import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback  # AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
import time

start_time = time.time()

train_data = pd.read_excel(r"")  
val_data   = pd.read_excel(r"")
test_data  = pd.read_excel(r"")


if train_data["질병"].dtype == "object":
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_data["질병"] = le.fit_transform(train_data["질병"])
    val_data["질병"]   = le.transform(val_data["질병"])
    test_data["질병"]  = le.transform(test_data["질병"])
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
else:
    unique_labels = sorted(set(train_data["질병"].tolist()))
    label_mapping = {i: str(i) for i in unique_labels}

train_texts  = train_data["진료 기록"].tolist()
train_labels = train_data["질병"].tolist()

val_texts  = val_data["진료 기록"].tolist()
val_labels = val_data["질병"].tolist()

test_texts  = test_data["진료 기록"].tolist()
test_labels = test_data["질병"].tolist()


tokenizer = BertTokenizer.from_pretrained("") # madatnlp/km-bert, BM-K/KoSimCSE-roberta, klue/roberta-large


class ClinicalDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),       
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

train_dataset = ClinicalDataset(train_texts, train_labels, tokenizer)
val_dataset   = ClinicalDataset(val_texts, val_labels, tokenizer)


num_labels = len(set(train_labels))
model = BertForSequenceClassification.from_pretrained("", num_labels=num_labels) # madatnlp/km-bert, BM-K/KoSimCSE-roberta, klue/roberta-large



training_args = TrainingArguments(
    output_dir=r"",
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_dir=r"",
    metric_for_best_model = "accuracy",
    seed=42,
    learning_rate=0.00005
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    weighted_f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": accuracy, "weighted_f1": weighted_f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

    

trainer.train()



