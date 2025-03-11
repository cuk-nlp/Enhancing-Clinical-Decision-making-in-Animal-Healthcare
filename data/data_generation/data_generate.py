import openai
import pandas as pd
import random
import time
import concurrent.futures


openai.api_key = ""


def extract_disease_symptoms(file_path, sheet_name=''):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    

    df = df.drop_duplicates(subset=[df.columns[0]])
    
    disease_column = df.columns[0]  
    symptom_columns = df.columns[1:]  
    
    disease_symptoms = {}
    
    for _, row in df.iterrows():
        disease = row[disease_column]  
        
        symptoms = symptom_columns[row[symptom_columns].fillna(0).astype(int) == 1].tolist()
        
        if len(symptoms) >= 1:
            disease_symptoms[disease] = symptoms
    
    
    return disease_symptoms

def generate_medical_record(disease, symptoms):
    try:
        num_selected = random.randint(1, len(symptoms))  
        selected_symptoms = random.sample(symptoms, num_selected)

        prompt = f""" """ # 프롬프트는 예시 데이터의 프라이버시 이슈로 인해 삭제

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.45
        )
        record = response["choices"][0]["message"]["content"].strip()
        
        return [disease, ", ".join(selected_symptoms), record]

    except Exception as e:

        return [disease, ", ".join(selected_symptoms), "error"]
    
def generate_medical_records(disease_symptoms, total_samples=1220, max_workers=5):

    generated_data = []
    start_time = time.time()
    disease_list = list(disease_symptoms.keys())

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for disease in disease_list:
            if len(generated_data) >= total_samples:
                break  
            
            futures = []
            disease_symptom = disease_symptoms[disease]


            for _ in range(5):
                if len(generated_data) >= total_samples:
                    break  
                
                future = executor.submit(generate_medical_record, disease, disease_symptom)
                futures.append(future)


            for future in concurrent.futures.as_completed(futures):
                generated_data.append(future.result())

                if len(generated_data) >= total_samples:
 
                    return generated_data
                

    return generated_data

file_path = r""

disease_symptoms = extract_disease_symptoms(file_path)
augmented_data = generate_medical_records(disease_symptoms, total_samples=1220, max_workers=5)


output_df = pd.DataFrame(augmented_data, columns=["Diagnosis", "Selected symptom", "record"])
output_file = r""
output_df.to_excel(output_file, index=False)
