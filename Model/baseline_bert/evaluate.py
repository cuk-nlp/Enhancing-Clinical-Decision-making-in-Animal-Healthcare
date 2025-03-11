import os
import torch
import time
import logging
import numpy as np
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification


logging.basicConfig(
    filename='your_log_path',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


model_path = r"your_checkpoint_path"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


real_data_path = r"data_path"
new_data = pd.read_excel(real_data_path)

disease_file = r"DB_path"
df_disease = pd.read_excel(disease_file)
df_disease.rename(columns={df_disease.columns[0]: "질병"}, inplace=True)
disease_col = df_disease.columns[0]  # "질병"

if "질병" in new_data.columns and new_data["질병"].dtype == "object":
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    new_data["질병"] = le.fit_transform(new_data["질병"])
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
else:
    label_mapping = {}

new_texts = new_data["진료 기록"].tolist()
new_labels = new_data["질병"].tolist() if "질병" in new_data.columns else None



def predict_topk(text, top_k=3, max_length=512):
    start_time = time.time()
    
    inputs = tokenizer(text, max_length=max_length, truncation=True,
                       padding='max_length', return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0)
    
    topk = torch.topk(probs, k=top_k)
    topk_indices = topk.indices.tolist()
    topk_probs = topk.values.tolist()
    
    results = [
        f"{label_mapping.get(idx, idx)} ({prob:.2f})"
        for idx, prob in zip(topk_indices, topk_probs)
    ]
    
    inference_time = round(time.time() - start_time, 2)
    return results, inference_time


def generate_additional_question(top_diseases, df_symptoms=df_disease, disease_col="질병"):
    if not top_diseases:
        return "추가 확인할 증상이 없습니다."
    all_symptoms = []
    for disease_str in top_diseases:
        disease_name = disease_str.split("(")[0].strip()
        sub_df = df_symptoms[df_symptoms[disease_col] == disease_name]
        if not sub_df.empty:
            row_symptoms = sub_df.iloc[:, 1:]
            matching_symptoms = row_symptoms.columns[row_symptoms.iloc[0] == 1].tolist()
            all_symptoms.extend(matching_symptoms)
    if not all_symptoms:
        return "추가 확인할 증상이 없습니다."
    
    symptom_counts = Counter(all_symptoms)
    most_common_symptom = symptom_counts.most_common(1)[0][0]
    all_symptoms_list = ', '.join(set(all_symptoms))
    return f"환자의 반려동물에게 다음 증상 중 하나라도 있나요? {all_symptoms_list}. 가장 많이 겹치는 증상은 '{most_common_symptom}'입니다."


top_k_values = [3, 5, 10]

results_list = []
memo_times = []  

for i, text in enumerate(new_texts):
    if new_labels is not None:
        true_diag = label_mapping.get(new_labels[i], "Unknown")
    else:
        true_diag = "N/A"
    
    # 1) k=3
    block1_start = time.time()
    pred_k3, time_k3 = predict_topk(text, top_k=3)
    q_k3 = generate_additional_question(pred_k3)
    block1_end = time.time()
    block1_time = round(block1_end - block1_start, 2)
    memo_times.append(block1_time)
    
    # 2) k=5
    pred_k5, time_k5 = predict_topk(text, top_k=5)
    q_k5 = generate_additional_question(pred_k5)
    
    # 3) k=10
    pred_k10, time_k10 = predict_topk(text, top_k=10)
    q_k10 = generate_additional_question(pred_k10)
    

    results_list.append({
        "record_text": text,
        "Diagnosis": true_diag,
        "Predicted_3": ", ".join(pred_k3),
        "Question_3": q_k3,
        "Predicted_5": ", ".join(pred_k5),
        "Question_5": q_k5,
        "Predicted_10": ", ".join(pred_k10),
        "Question_10": q_k10,
    })


total_time_k3 = round(sum(memo_times), 2)
avg_time_k3 = round(np.mean(memo_times), 2) if memo_times else 0

logging.info(f"Total inference time (k=3 only): {total_time_k3} sec")
logging.info(f"Average inference time (k=3 only): {avg_time_k3} sec")


logging.info("==== Inference & Additional Questions Results ====")
for idx, row in enumerate(results_list):
    txt_brief = row["record_text"][:30]
    msg = (f"[문장 #{idx}] {txt_brief}...\n"
           f" - Diagnosis: {row['Diagnosis']}\n"
           f" - Predicted_3: {row['Predicted_3']}\n"
           f" - Question_3: {row['Question_3']}\n"
           f" - Predicted_5: {row['Predicted_5']}\n"
           f" - Question_5: {row['Question_5']}\n"
           f" - Predicted_10: {row['Predicted_10']}\n"
           f" - Question_10: {row['Question_10']}\n")
    logging.info(msg)


logging.info(f"All inference done. (k=3 total time: {total_time_k3} sec)")

