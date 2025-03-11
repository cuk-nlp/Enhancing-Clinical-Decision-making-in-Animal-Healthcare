import os
os.environ["HF_HOME"] = "/home/user09/beaver/data/cache"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from collections import Counter
import re
import bitsandbytes as bnb
import time 
import random
import logging  
import numpy as np


logging.basicConfig(
    filename='your_log_path',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(1)


MODEL_NAME = "Model_path"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 사용 중인 디바이스: {device}")
logging.info(f"Device used: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    trust_remote_code=True
).to(device)

print("✅ bitsandbytes 4-bit 퀀타이제이션 완료!")
logging.info("bitsandbytes 4-bit quantization complete.")


memo_file = r""  
disease_file = r""  

df_memo = pd.read_excel(memo_file)
df_disease = pd.read_excel(disease_file)
df_disease.rename(columns={df_disease.columns[0]: "질병"}, inplace=True)

disease_col = df_disease.columns[0]  
symptom_cols = df_disease.columns[1:]
db_symptoms = list(symptom_cols)     
all_diseases = df_disease[disease_col].tolist()

logging.info("Disease-symptom data loaded.")


def predict_disease_from_memo(memo, top_k):
    """
    top_k에 맞춰 '정확히 {top_k}개만 예측하세요' 프롬프트를 구성하고,
    예측 결과(질병명 리스트) + 걸린 시간 반환
    """
    start_time = time.time()

    prompt_template = f"""
    """
    inputs = tokenizer(prompt_template, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    idx = response.find("**실제 출력**:")
    short_response = response[idx + len("**실제 출력**:"):].strip() if idx != -1 else response.strip()

    predicted = []
    for line in short_response.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            disease = line[2:].strip()
            if disease in all_diseases:
                predicted.append(disease)

    
    predicted = list(dict.fromkeys(predicted))
    predicted = predicted[:top_k]

    elapsed_time = round(time.time() - start_time, 2)
    return predicted, elapsed_time


def generate_additional_question(top_diseases):
    start_time = time.time()
    if not top_diseases:
        return "추가 확인할 증상이 없습니다.", 0
    all_symptoms = []
    for disease in top_diseases:
       
        sub_df = df_disease[df_disease[disease_col] == disease].iloc[:, 1:]
        if len(sub_df) > 0:  
            matching_symptoms = sub_df.columns[sub_df.iloc[0] == 1].tolist()
            all_symptoms.extend(matching_symptoms)
    if not all_symptoms:
        return "추가 확인할 증상이 없습니다.", 0
    symptom_counts = Counter(all_symptoms)
    most_common_symptom = symptom_counts.most_common(1)[0][0]
    all_symptoms_list = ', '.join(set(all_symptoms))
    question = f"환자의 반려동물에게 다음 증상 중 하나라도 있나요? {all_symptoms_list}. 가장 많이 겹치는 증상은 '{most_common_symptom}'입니다."
    elapsed_time = round(time.time() - start_time, 2)
    return question, elapsed_time


def accuracy_at_k(y_true_list, y_pred_list, k):
    correct_count = 0
    for true_labels, pred_labels in zip(y_true_list, y_pred_list):
        if any(t in pred_labels[:k] for t in true_labels):
            correct_count += 1
    return correct_count / len(y_true_list) if y_true_list else 0


df_results = pd.DataFrame({
    "Memo": df_memo["진료 기록"],
    "Diagnosis": df_memo["질병"].fillna("").apply(lambda x: x.split(",") if pd.notna(x) else [])
})

top_k_values = [3, 5, 10]
for k in top_k_values:
    df_results[f"Predicted_{k}"] = [[] for _ in range(len(df_results))]
    df_results[f"Question_{k}"]  = [""] * len(df_results)


global_start_time = time.time()
memo_times = []

for i, row in df_results.iterrows():
    memo = row["Memo"]
    if not isinstance(memo, str) or not memo.strip():
        continue

    # (1) k=3
    block1_start = time.time()
    pred_k3, _ = predict_disease_from_memo(memo, top_k=3)
    q_k3, _     = generate_additional_question(pred_k3)
    block1_end  = time.time()
    block1_time = round(block1_end - block1_start, 2)
    memo_times.append(block1_time)

    df_results.at[i, "Predicted_3"] = pred_k3
    df_results.at[i, "Question_3"]  = q_k3

    # (2) k=5
    pred_k5, _ = predict_disease_from_memo(memo, top_k=5)
    q_k5, _    = generate_additional_question(pred_k5)
    df_results.at[i, "Predicted_5"] = pred_k5
    df_results.at[i, "Question_5"]  = q_k5

    # (3) k=10
    pred_k10, _ = predict_disease_from_memo(memo, top_k=10)
    q_k10, _    = generate_additional_question(pred_k10)
    df_results.at[i, "Predicted_10"] = pred_k10
    df_results.at[i, "Question_10"]  = q_k10

global_total_time = round(sum(memo_times), 2)
avg_memo_time = round(np.mean(memo_times), 2) if memo_times else 0
# print(f"\n전체 메모 처리 총 실행 시간 (k=3만 포함): {global_total_time}초")
# print(f"평균 메모 실행 시간 (k=3만 포함): {avg_memo_time}초")
logging.info(f"Total processing time (k=3 only): {global_total_time} seconds.")
logging.info(f"Average processing time per memo (k=3 only): {avg_memo_time} seconds.")


y_true_list = df_results["Diagnosis"].tolist()
for k in top_k_values:
    y_pred_list = df_results[f"Predicted_{k}"].tolist()
    acc_k = accuracy_at_k(y_true_list, y_pred_list, k)
    msg = f"Accuracy@{k}: {acc_k:.4f}"
    # print(msg)
    logging.info(msg)



for i, row in df_results.iterrows():
    print(f"\n[Memo #{i}]")
    for k in top_k_values:
        q_col = row[f"Question_{k}"]
        print(f" - Question_{k}: {q_col}")


logging.info("All predictions and evaluations are done.")
