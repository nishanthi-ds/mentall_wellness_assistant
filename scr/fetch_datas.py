import pandas as pd
import os
import numpy as np
import json


def csv_loader(file_path,all_datas ):

    df = pd.read_csv(file_path)

    possible_cols = [
            ("question", "answer"),
            ("Prompt", "Response"),
            ("Questions", "Answers")
        ]
        
    for q_col, a_col in possible_cols:
        if q_col in list(df.columns) and a_col in list(df.columns):
            for ques, ans in np.array(df[[q_col, a_col]]):
                all_datas.append([ques, ans])
    return all_datas


def json_loader(filename, file_path, all_datas):

    with open(file_path,"r", encoding="utf-8") as f:
                data = json.load(f)
    if 'MentalChat16K' in filename:
        for i in range(len(data)):
            all_datas.append([data[i]['input'], data[i]['output']]) 
    else:  
        for i in range(len(data['intents'])):
            qn, ans= data['intents'][0]['patterns'], data['intents'][0]['responses']
            qn = " or ".join(qn)
            ans = " or ".join(ans)
            all_datas.append([qn, ans])
    return all_datas

# Load datas
def get_data(data_folder='data'):
    all_datas = []
    
    for filename in os.listdir(data_folder):
        
        # --- Read all CSVs ---
        if filename.endswith(".csv"):
            file_path = os.path.join(data_folder, filename)
            
            all_datas= csv_loader(file_path,all_datas )
            

        # --- Read all json files ---
        elif filename.endswith(".json") or filename.endswith(".jsonl"):
            file_path = os.path.join(data_folder, filename)
            all_datas= json_loader(filename, file_path, all_datas)
            
        else:
            print("Invalid File")

    return all_datas
