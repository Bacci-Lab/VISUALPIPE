import numpy as np
import os
import pandas as pd
import pickle
import json

def create_H5_dataset(group, variable, variable_name):
    for name, value in zip(variable_name, variable):
        group.create_dataset(name, data=value)
        
def save_pickle(object, save_directory='', filename=''):
    with open(os.path.join(save_directory, filename + '.pkl'), 'wb') as outp:
        pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)

def compile_xlsx_file(df : pd.DataFrame, filepath, filename="compile_per_Tseries.xlsx", verbose=True) :
    path_compile = os.path.join(filepath, filename)
    isexsist = os.path.exists(path_compile)

    if not isexsist:
        if verbose :
            print(f"Excel file created: {path_compile}")
        df.to_excel(path_compile)
        return df
    else : 
        if verbose :
            print("Excel sheet already exists")
        df_existing = pd.read_excel(path_compile).set_index("Session_id")

        if df.index[0] not in df_existing.index:
            # Append new data
            df_combined = pd.concat([df_existing, df], ignore_index=False)

            # Save the combined data to Excel
            df_combined.to_excel(path_compile)
            
            if verbose :
                print("New row added successfully.")
            return df_combined
        else:
            if verbose :
                print("Row with the same session id already exists in the Excel file. Row not added.")
            return df_existing

def get_mouse_id(path, filename):
    excel_path = os.path.join(path, filename+'.xlsx')
    if os.path.exists(excel_path) :
        df = pd.read_excel(excel_path)
        return df["Code"].loc[0]
    else :
        print(f"No excel file for mouse metadata found : {excel_path}")
        return None

def get_metadata(path):
    metadata_path = os.path.join(path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as file:
            data = json.load(file)
            unique_id = data['date'] + '_' + data['time']
            global_protocol = data['protocol']
            experimenter = data['experimenter']
            subject_id = data['subject_ID']
            return unique_id, global_protocol, experimenter, subject_id
    else:
        raise Exception("No JSON metadata file exists in this directory")