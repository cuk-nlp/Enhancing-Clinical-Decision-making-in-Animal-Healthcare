import os
# os.environ["HF_HOME"] = ""
import nltk
nltk.data.path.append(r"")  
from nltk.tokenize import word_tokenize, sent_tokenize
import requests
import concurrent.futures
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from collections import Counter
import re
import time 
import openai
import random
import numpy as np
import logging
from konlpy.tag import Okt  

API_URL = ""

API_KEY = ""

logging.basicConfig(filename='your_log_path', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

overall_start_time = time.time()


max_length = 40  
stride = 10   


memo_file = r"" 
disease_file = r""  

df_memo = pd.read_excel(memo_file)
print(":작은_파란색_다이아몬드: 질병-증상 데이터 로드 중...")
df_disease = pd.read_excel(disease_file)

df_disease.rename(columns={df_disease.columns[0]: "Diagnosis"}, inplace=True)
disease_col = df_disease.columns[0]  
symptom_cols = df_disease.columns[1:]  
db_symptoms = list(symptom_cols) 
print(":흰색_확인_표시: 질병-증상 데이터 로드 완료!")

okt = Okt()
def custom_tokenize(text):

    words = okt.morphs(text) 
    return words
def split_into_sliding_windows(text, max_length=max_length, stride=stride):

    tokens = custom_tokenize(text) 
    windows = []
    for start in range(0, len(tokens), stride):
        end = min(start + max_length, len(tokens))
        window_text = " ".join(tokens[start:end])  
        windows.append(window_text)
    return windows
def call_gpt4o(sentence):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt_template = f"""
    """
    prompt = prompt_template.format(db_symptoms=', '.join(db_symptoms), sentence=sentence)
    
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 128
    }
 
    response = requests.post(API_URL, json=payload, headers=headers)
    result = response.json()

    return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
def extract_symptoms(memo, max_workers=5):
    print(":작은_파란색_다이아몬드: LLM을 사용하여 증상 추출 중...")
    start_time = time.time() 

    sliding_windows = split_into_sliding_windows(memo.strip(), max_length, stride)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(call_gpt4o, sliding_windows))
    symptoms_list = set()
    for response_text in results:
        for line in response_text.split("\n"):
            line = line.strip()
            if line.startswith("- "):  
                symptom = line[2:].strip()
                if symptom in db_symptoms:
                    symptoms_list.add(symptom)
    symptoms_list = sorted(symptoms_list)

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    return symptoms_list, elapsed_time

def find_related_diseases(symptoms):
    start_time = time.time()
    matched_diseases = []
    for symptom in symptoms:
        if symptom in df_disease.columns:
            diseases = df_disease[df_disease[symptom] == 1][disease_col].tolist()
            matched_diseases.extend(diseases)
    elapsed_time = round(time.time() - start_time, 2)
    return matched_diseases, elapsed_time

def filter_diseases_by_common_occurrence(diseases, df_disease=df_disease, df_eval=None, top_n=None):
    start_time = time.time()
    if not isinstance(diseases, list) or not diseases:
        return [], 0
    if df_eval is None:
        try:
            df_eval = globals()['df_eval']
        except KeyError:
            raise ValueError("df_eval_error")
    
    df_disease.rename(columns={df_disease.columns[0]: "Diagnosis"}, inplace=True)
    df_disease.columns = df_disease.columns.str.strip()
    df_disease.set_index("Diagnosis", inplace=True)

    disease_counter = Counter(diseases)
    disease_freq = dict(disease_counter)
    

    disease_data_count = df_eval["Diagnosis"].value_counts().to_dict()
    

    disease_symptom_count = {
        disease: df_disease.loc[disease].sum() if disease in df_disease.index else 0
        for disease in disease_counter.keys()
    }

    sorted_diseases = sorted(
        disease_counter.keys(),
        key=lambda disease: (
            -disease_freq[disease],  
            -disease_data_count.get(disease, 0),  
            disease_symptom_count.get(disease, float('inf'))
        )
    )
    
    selected_diseases = sorted_diseases[:top_n]
    selected_diseases.sort(
        key=lambda d: (
            -disease_freq[d],
            -disease_data_count.get(d, 0),
            disease_symptom_count.get(d, float('inf'))
        )
    )
    
    df_disease.reset_index(inplace=True)
    elapsed_time = round(time.time() - start_time, 2)
    return selected_diseases, elapsed_time

def accuracy_at_k(y_true, y_pred):
    return np.mean([1 if true in pred else 0 for true, pred in zip(y_true, y_pred)])

def generate_additional_question(top_diseases):
    if not top_diseases:
        return "There are no additional symptoms to check.", 0
    start_time = time.time()
    relevant_rows = df_disease[df_disease[disease_col].isin(top_diseases)]
    symptom_counts = relevant_rows.iloc[:, 1:].sum(axis=0)
    if symptom_counts.sum() == 0:
        return "There are no additional symptoms to check.", 0
    most_common_symptom = symptom_counts.idxmax() if symptom_counts.any() else "없음"
    all_symptoms_list = ', '.join(symptom_counts.index[symptom_counts > 0])
    question = f"Does the patient have any of the following symptoms? {all_symptoms_list}. The most common overlapping symptom is '{most_common_symptom}'."
    elapsed_time = round(time.time() - start_time, 2)
    return question, elapsed_time

def process_memo(row):
    memo, diagnosis = row["record"], row["Diagnosis"]
    if pd.isna(memo) or pd.isna(diagnosis):
        return None
    total_start_time = time.time()
    
  
    predicted_symptoms, ex_time = extract_symptoms(memo, max_workers=5)
   
    initial_diseases, ex_ex_time = find_related_diseases(predicted_symptoms)
    
    total_time = round(time.time() - total_start_time, 2)
    
    return {
        "Memo": memo,
        "Diagnosis": diagnosis,
        "Extracted symptoms": ', '.join(predicted_symptoms) if predicted_symptoms else "",
        "Related diseases": ', '.join(initial_diseases) if initial_diseases else "",
        "total time (sec)": total_time
    }


memo_eval_results = []
for row in df_memo.to_dict("records"):
    result = process_memo(row)
    if result:
        memo_eval_results.append(result)
df_results = pd.DataFrame(memo_eval_results)


df_eval = df_results.copy()
average_memo_time = round(df_results["total time (sec)"].mean(), 2)
top_ns = [3, 5, 10]
accuracy_results = {}

time_top3 = None
for top_n in top_ns:
    branch_start = time.time()
    df_eval["Filtered Diseases"] = df_eval["Related diseases"].apply(
        lambda x: filter_diseases_by_common_occurrence(str(x).split(", "), df_eval=df_eval, top_n=top_n)[0] if pd.notna(x) else []
    )

    df_eval["Additional Question"] = df_eval["Filtered Diseases"].apply(
        lambda diseases: generate_additional_question(diseases)[0] if diseases else "There are no additional symptoms to check."
    )
    
    branch_time = time.time() - branch_start
    if top_n == 3:
        time_top3 = branch_time
    y_true = df_eval['Diagnosis'].tolist()
    y_pred = df_eval['Filtered Diseases'].tolist()
    accuracy = accuracy_at_k(y_true, y_pred)
    accuracy_results[top_n] = accuracy

    logging.info("Additional question (top_%d): %s", top_n, df_eval["Additional Question"])
    
if time_top3 is None:
    total_execution_time = round(time.time() - overall_start_time, 2)
else:
    total_execution_time = round(time_top3, 2)


for top_n in top_ns:
    logging.info("Accuracy@%d: %.4f", top_n, accuracy_results[top_n])
logging.info("average_memo_time (sec): %.2f", average_memo_time)
logging.info("total_execution_time (sec): %d", total_execution_time)
print("==== Accuracy log completion ====")
print("log completion")
