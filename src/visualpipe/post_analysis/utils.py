import pandas as pd
import numpy as np
import glob
import os

def load_excel_sheet(excel_sheet_path:str, protocol_name:str, neuron_type:str=None, genotype:str=None) :
    """
    Load an excel sheet and filter it according to the specified protocol name.

    Parameters
    ----------
    excel_sheet_path : str
        Path to the excel sheet.
    protocol_name : str
        Name of the protocol to filter the sessions.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the selected sessions with the specified protocol name.
    """
    
    column_names = ["Session_id", "Output_id", "Protocol", "Mouse_id", "Genotype", "Neuron_type", "Analyze", "Session_path"]

    df = pd.read_excel(excel_sheet_path)
    df = df[df["Protocol"] == protocol_name]
    if neuron_type is not None :
        df = df[df["Neuron_type"] == neuron_type]
    if genotype is not None :
        df = df[df["Genotype"] == genotype]
    df = df[df["Analyze"] == 1]
    duplicates = df.duplicated(subset=['Session_id'], keep='first')

    if np.sum(duplicates) > 0 : 
        print(f"There is/are {np.sum(duplicates)} duplicated session(s) in the excel file. Please remove them or set 'Analyze' to 0 in the excel file. The first occurence will be kept automatically if nothing is specified.")
        print(f"    Duplicated session(s) : {df[duplicates]['Session_id'].unique()}")
        df = df[~duplicates] # Remove duplicates

    print(f"Mice included : {df.Mouse_id.unique()}")
    
    return df[column_names]

def load_data_session(path:str) :
    """
    Load the validity and trials data from the specified path.

    Parameters
    ----------
    path : str
        Path to the directory containing the .npz and .npy files.

    Returns
    -------
    validity : dict
        Dictionary containing the validity data loaded from the .npz file.
    trials : dict
        Dictionary containing the trials data loaded from the .npy file.
    stimuli_df : pandas.DataFrame
        DataFrame containing the visual stimuli information loaded from the .xlsx file.

    Raises
    ------
    FileNotFoundError
        If the expected .npz, .npy or .xlsx files are not found in the specified path.
    """
    # Load the npz file
    npz_files = glob.glob(os.path.join(path, "*protocol_validity_2.npz"))
    if len(npz_files) == 1:
        validity = np.load(npz_files[0], allow_pickle=True)
        validity = dict(validity)
    else:
        raise FileNotFoundError(f"Expected exactly one .npz file in {path}, found {len(npz_files)} files")      
    
    # Load .npy file
    npy_files = glob.glob(os.path.join(path, "*trials.npy"))
    if len(npy_files) == 1:
        trials = np.load(npy_files[0], allow_pickle=True).item()
    else:
        raise FileNotFoundError(f"Expected exactly one .npy file in {path}, found {len(npy_files)}")
    
    # Load xlsx file with visual stimuli info
    xlsx_files = glob.glob(os.path.join(path, "*visual_stim_info.xlsx"))
    if len(xlsx_files) == 1:
        stimuli_df = pd.read_excel(os.path.join(path, xlsx_files[0]), engine='openpyxl').set_index('id')
    else:
        raise FileNotFoundError(f"Expected exactly one .xlsx file in {path}, found {len(xlsx_files)} files")   

    return validity, trials, stimuli_df

def get_session_metadata(df:pd.DataFrame, k:int):
    """
    Get the metadata of a session from the dataframe and index.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the session information.
    k : int
        Index of the row in the dataframe to get the information from.

    Returns
    -------
    mouse_id : str
        Mouse ID.
    session_id : str
        Session ID.
    output_id : str
        Output ID.
    session_path : str
        Path to the session directory.
    """
    mouse_id = df["Mouse_id"].iloc[k]
    session_id = df["Session_id"].iloc[k]
    output_id = df["Output_id"].iloc[k]
    session_path = os.path.join(df["Session_path"].iloc[k], f"{session_id}_output_{output_id}")

    return mouse_id, session_id, output_id, session_path

def get_period_names(trace_type:str):
    """
    Get the period names for a given trace type.

    Parameters
    ----------
    trace_type : str
        Type of trace ('dFoF0-baseline' or 'z-scores').

    Returns
    -------
    period_names : list[str]
        List of period names corresponding to the trace type.
    """
    if trace_type == 'dFoF0-baseline':
        period_names = ['norm_averaged_baselines', 'norm_trial_averaged_ca_trace', 'norm_post_trial_averaged_ca_trace']
    elif trace_type == 'z-scores':
        period_names = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']

    return period_names

def sum_value_in_dict(dict:dict, key:str, value: int | float):
    """
    Add a value to a key in a dictionary.

    If the key does not exist in the dictionary, it is created with the value.
    If the key already exists, the value is summed to the existing one.

    Parameters
    ----------
    dict : dict
        The dictionary to update.
    key : str
        The key to update.
    value : int or float
        The value to add.
    """
    if key not in dict.keys():
        dict[key] = value
    else:
        dict[key] += value

def sum_list_in_dict(dict:dict, key:str, l:np.ndarray):
    """
    Add a list of values to a key in a dictionary.

    If the key does not exist in the dictionary, it is created with the list of values.
    If the key already exists, the list of values is summed element-wise to the existing one.

    Parameters
    ----------
    dict : dict
        The dictionary to update.
    key : str
        The key to update.
    l : np.ndarray
        The list of values to add.
    """
    if key not in dict.keys():
        dict[key] = np.array(l)
    else:
        # Sum element-wise the list of values to the existing one
        dict[key] += np.array(l)
