import numpy as np
import os
import pandas as pd
import pickle
import json
import glob

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
        df_existing = pd.read_excel(path_compile)
        
        same_recording_list = [id for id in df_existing.index if df_existing.loc[id]["Session_id"] == df.index[0]]
        for el in same_recording_list :
            if df_existing.loc[el]["Output_id"] == df.iloc[0]["Output_id"] :
                # Remove row
                df_existing.drop(index=el, inplace=True)
                if verbose :
                    print("Row with the same session id and output id already exists in the Excel file. Removing old row.")
        
        # Set index to Session_id
        df_existing.set_index("Session_id", inplace=True)

        # Append new data
        df_combined = pd.concat([df_existing, df], ignore_index=False)
        
        # Save the combined data to Excel
        df_combined.to_excel(path_compile)
        
        if verbose :
            print("New row added successfully.")
        return df_combined

def get_mouse_id(path, filename):
    excel_path = os.path.join(path, filename+'.xlsx')
    if os.path.exists(excel_path) :
        try :
            df = pd.read_excel(excel_path, keep_default_na=False)
            mouse_code = df["Code"].loc[0]
        except :
            df = pd.read_excel(excel_path, header=1, keep_default_na=False)
            mouse_code = df["Code"].loc[0]
        return mouse_code
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
    
def create_output_folder(path, unique_id):
    version = []
    i = 1
    list_dir = glob.glob(os.path.join(path, unique_id + "_output_*"))

    for s in list_dir :
        splited_string = os.path.basename(s).split("_")
        version.append(int(splited_string[5]))
        
    while i in version :
        i +=1
    id_version = str(i)

    save_dir = os.path.join(path, unique_id + "_output_" + id_version)
    if not os.path.exists(save_dir) :
        os.makedirs(save_dir)
    else :
        raise Exception("Folder should not exist")
    
    save_fig_dir = os.path.join(save_dir, unique_id + "_figures")
    if not os.path.exists(save_fig_dir) :
        os.makedirs(save_fig_dir)

    return save_dir, save_fig_dir, id_version

def save_analysis_settings(data:dict, save_directory):
    text = ""
    for el in data :
        temp = f"{el} : {data[el]}\n"
        text += temp
    
    save_direction_text = os.path.join(save_directory , "analysis_settings.txt")
    with open(save_direction_text, 'a') as file:
        file.write(text + '\n')