import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from itertools import chain
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

def split_stages(speed, behavior, speed_threshold:float, behav_threshold:float, real_time, min_states_window:dict, 
                 fs:float, method='std', speed_filter_kernel=0, behav_filter_kernel=0):
    """
    October 2023 - Bacci Lab - faezeh.rabbani97@gmail.com

    .......................................................................

    This function splits fluorescence activity to different time windows based
    on duration of running, speed of running and standard deviation of whisking
    final results contains maximum 4 movement states:
    1. Active movement: running with the speed of above 0.5 cm/s for more than 2 seconds
    2. Aroused stationary: aroused state without locomotion
    3. Rest: time periods which there are no whisking or movements
    \n.....................................................................
    
    :param speed: speed trace
    :param behavior: facemotion or pupil data
    :param speed_threshold: speed threshold in cm/s used to determine running periods
    :param behav_threshold: facemotion or pupil threshold to use to determine arousal periods without running
    :param real_time: real time stamps extracted from xml file
    :param min_states_window: dictionnary with minimum dt in s for each states
    :param fs: sample frequency of data
    :param method: string indicating which threshold to use between 'quantile' or 'std'
    :param speed_filter_kernel: kernel size for filtering speed trace
    :param behav_filter_kernel: kernel size for filtering behavioral trace

    :return Real_Time_states: dictionnary with windows of real timestamps for all states activity (s)
    :return states_window: dictionnary with windows of index for all states activity (frames)
    """

    undefined_state_idx = np.arange(0,len(speed))
    ids = np.arange(0, len(speed))
    exclude_start_session = round(1 * fs)
    min_inter_interval_size = round(np.min([min_states_window['run']/2, min_states_window['rest']/2, min_states_window['AS']/2]))

    if speed_filter_kernel > 0 :
        speed = gaussian_filter1d(speed, speed_filter_kernel)
    if behav_filter_kernel > 0 :
        behavior = gaussian_filter1d(behavior, behav_filter_kernel)
    
    scaler = MinMaxScaler()
    behavior = scaler.fit_transform(behavior.reshape(-1, 1)).reshape(-1)
    
    if method == 'std' :
        behav_threshold = behav_threshold*(np.std(behavior))
    elif method == "quantile" :
        behav_threshold = np.quantile(behavior, behav_threshold)
    
    ###------------------------------ Calculate active movement state -----------------------------###
    # default param: duration > 60 frames and speed > 0.5 s/m
    id_above_thr_speed = np.extract(speed[exclude_start_session:] >= speed_threshold, ids[exclude_start_session:])
    Aroused_Running_index, Aroused_Running_window, Real_Time_Aroused_Running = find_intervals(id_above_thr_speed, min_states_window['run'], real_time, round(0.3 * fs), min_inter_interval_size)
    
    Aroused_Running_index_check1 = []
    for i in Aroused_Running_index:
        if i[-1] < len(speed) - round(1 * fs) :
            ST = np.arange(i[0], i[-1] + round(1 * fs))
        else:
            ST = np.arange(i[0], len(speed))
        Aroused_Running_index_check1.append(ST)
    Aroused_Running_index_check = list(chain(*Aroused_Running_index_check1))

    ###------------------------------Calculate AS state-----------------------------###
    id_AS = np.extract(np.array(behavior[exclude_start_session:] >= behav_threshold) * np.array(speed[exclude_start_session:] < speed_threshold), ids[exclude_start_session:])
    Aroused_stationary_index_temp, _, _  = find_intervals(id_AS, np.min([round(0.3 * fs), min_states_window['AS']]), real_time)

    #delete idx belonging to running state
    delet_Running_IDX = []
    Aroused_stationary_index_check = list(chain(*Aroused_stationary_index_temp)) 
    for i in Aroused_stationary_index_check:
        if i in Aroused_Running_index_check:
            delet_Running_IDX.append(i)
    mask = np.isin(Aroused_stationary_index_check, delet_Running_IDX, invert=True)
    result = np.extract(mask, Aroused_stationary_index_check)

    Aroused_stationary_index, Aroused_stationary_window, Real_time_Aroused_stationary = find_intervals(result, min_states_window['AS'], real_time, round(0.3 * fs), min_inter_interval_size)

    Aroused_stationary_index_check1 = []
    for i in Aroused_stationary_index:
        if i[-1] < len(speed) - round(0.7 * fs) :
            ST = np.arange(i[0], i[-1] + round(0.7 * fs))
        else:
            ST = np.arange(i[0], len(speed))
        Aroused_stationary_index_check1.append(ST)
    Aroused_stationary_index_check = list(chain(*Aroused_stationary_index_check1))

    ###------------------------------Calculate Rest state-----------------------------###
    id_rest = np.extract(np.array(behavior < behav_threshold) * np.array(speed < speed_threshold), ids)
    Rest_index_temp, _, _  = find_intervals(id_rest, np.min([round(0.3 * fs), min_states_window['rest']]), real_time)

    delet_IDX = []
    rest_index_check = list(chain(*Rest_index_temp))
    for i in rest_index_check:
        if i in Aroused_Running_index_check or i in Aroused_stationary_index_check:
            delet_IDX.append(i)
    mask = np.isin(rest_index_check, delet_IDX, invert=True)
    result = np.extract(mask, rest_index_check)

    rest_index, rest_window, Real_Time_rest = find_intervals(result, min_states_window['rest'], real_time, 0, min_inter_interval_size)

    ###------------------------------Undefined states-----------------------------###
    to_delete__idx = []
    for i in undefined_state_idx:
        if i in list(chain(*Aroused_stationary_index)) or\
                i in list(chain(*Aroused_Running_index)) or\
                i in list(chain(*rest_index)):
            to_delete__idx.append(i)
    mask = np.isin(undefined_state_idx, to_delete__idx, invert=True)
    result = np.extract(mask, undefined_state_idx)
    _,  Undefined_state_window, Real_time_Undefined_state = find_intervals(result, 0, real_time)

    for window in Undefined_state_window :
        if window[0] != 0 :
            window[0] = window[0] - 1
        if window[-1] != len(real_time) - 1 :
            window[-1] = window[-1] + 1

    Real_Time_states = {'run' : Real_Time_Aroused_Running, 
                        'AS' : Real_time_Aroused_stationary,
                        'rest' : Real_Time_rest,
                        'undefined' : Real_time_Undefined_state}
    states_window = {'run' : Aroused_Running_window, 
                     'AS' : Aroused_stationary_window, 
                     'rest' : rest_window,
                     'undefined' : Undefined_state_window}

    return Real_Time_states, states_window

def split_stages_mixed(speed, pupil, facemotion, idx_lim_dark,
                       speed_threshold:float, pupil_threshold:float, facemotion_threshold:float, 
                       real_time, min_states_window:dict, fs:float, 
                       method_pupil='std', method_facemotion='std', 
                       speed_filter_kernel=0, pupil_filter_kernel=0, facemotion_filter_kernel=0):
    """
    .......................................................................

    This function splits fluorescence activity to different time windows based
    on duration of running, speed of running and standard deviation of whisking
    final results contains maximum 4 movement states:
    1. Active movement: running with the speed of above 0.5 cm/s for more than 2 seconds
    2. Aroused stationary: aroused state without locomotion
    3. Rest: time periods which there are no whisking or movements
    \n.....................................................................
    
    :param speed: speed trace
    :param pupil: pupil data
    :param facemotion: facemotion data
    :param idx_lim_dark: list of the limit indexes of dark periods
    :param speed_threshold: speed threshold in cm/s used to determine running periods
    :param pupil_threshold: pupil threshold to use to determine arousal periods without running
    :param facemotion_threshold: facemotion threshold to use to determine arousal periods without running
    :param real_time: real time stamps extracted from xml file
    :param min_states_window: dictionnary with minimum dt in s for each states
    :param fs: sample frequency of data
    :param method_pupil: string indicating which threshold to use between 'quantile' or 'std' for pupil
    :param method_facemotion: string indicating which threshold to use between 'quantile' or 'std' for facemotion
    :param speed_filter_kernel: kernel size for filtering speed trace
    :param pupil_filter_kernel: kernel size for filtering behavioral trace
    :param facemotion_filter_kernel: kernel size for filtering behavioral trace

    :return Real_Time_states: dictionnary with windows of real timestamps for all states activity (s)
    :return states_window: dictionnary with windows of index for all states activity (frames)
    """

    undefined_state_idx = np.arange(0,len(speed))
    ids = np.arange(0, len(speed))
    exclude_start_session = round(1 * fs)
    min_inter_interval_size = round(np.min([min_states_window['run']/2, min_states_window['rest']/2, min_states_window['AS']/2]))

    if speed_filter_kernel > 0 :
        speed = gaussian_filter1d(speed, speed_filter_kernel)
    if pupil_filter_kernel > 0 :
        pupil = gaussian_filter1d(pupil, pupil_filter_kernel)
    if facemotion_filter_kernel > 0 :
        facemotion = gaussian_filter1d(facemotion, facemotion_filter_kernel)
    
    scaler = MinMaxScaler()
    pupil = scaler.fit_transform(pupil.reshape(-1, 1)).reshape(-1)
    scaler = MinMaxScaler()
    facemotion = scaler.fit_transform(facemotion.reshape(-1, 1)).reshape(-1)
    
    pupil_without_dark, dark_mask = get_pupil_without_dark(pupil, idx_lim_dark, round(3 * fs))

    if method_pupil == 'std' :
        pupil_threshold = pupil_threshold*(np.std(pupil_without_dark))
    elif method_pupil == "quantile" :
        pupil_threshold = np.quantile(pupil_without_dark, pupil_threshold)

    if method_facemotion == 'std' :
        facemotion_threshold = facemotion_threshold*(np.std(facemotion))
    elif method_facemotion == "quantile" :
        facemotion_threshold = np.quantile(facemotion, facemotion_threshold)
    
    ###------------------------------ Calculate active movement state -----------------------------###
    # default param: duration > 60 frames and speed > 0.5 s/m
    id_above_thr_speed = np.extract(speed[exclude_start_session:] >= speed_threshold, ids[exclude_start_session:])
    Aroused_Running_index, Aroused_Running_window, Real_Time_Aroused_Running = find_intervals(id_above_thr_speed, min_states_window['run'], real_time, round(0.3 * fs), min_inter_interval_size)
    
    Aroused_Running_index_check1 = []
    for i in Aroused_Running_index:
        if i[-1] < len(speed) - round(1 * fs) :
            ST = np.arange(i[0], i[-1] + round(1 * fs))
        else:
            ST = np.arange(i[0], len(speed))
        Aroused_Running_index_check1.append(ST)
    Aroused_Running_index_check = list(chain(*Aroused_Running_index_check1))

    ###------------------------------Calculate AS state-----------------------------###
    pupil_modified = [pupil[i] if dark_mask[i] else 0 for i in range(pupil.shape[0])]
    facemotion_modified = [facemotion[i] if not dark_mask[i] else 0 for i in range(len(facemotion))]
    id_AS = np.extract((np.array(pupil_modified[exclude_start_session:] >= pupil_threshold) | np.array(facemotion_modified[exclude_start_session:] >= facemotion_threshold)) * np.array(speed[exclude_start_session:] < speed_threshold), ids[exclude_start_session:])
    Aroused_stationary_index_temp, _, _  = find_intervals(id_AS, np.min([round(0.3 * fs), min_states_window['AS']]), real_time)

    #delete idx belonging to running state
    delet_Running_IDX = []
    Aroused_stationary_index_check = list(chain(*Aroused_stationary_index_temp)) 
    for i in Aroused_stationary_index_check:
        if i in Aroused_Running_index_check:
            delet_Running_IDX.append(i)
    mask = np.isin(Aroused_stationary_index_check, delet_Running_IDX, invert=True)
    result = np.extract(mask, Aroused_stationary_index_check)

    Aroused_stationary_index, Aroused_stationary_window, Real_time_Aroused_stationary = find_intervals(result, min_states_window['AS'], real_time, round(0.3 * fs), min_inter_interval_size)

    Aroused_stationary_index_check1 = []
    for i in Aroused_stationary_index:
        if i[-1] < len(speed) - round(0.7 * fs) :
            ST = np.arange(i[0], i[-1] + round(0.7 * fs))
        else:
            ST = np.arange(i[0], len(speed))
        Aroused_stationary_index_check1.append(ST)
    Aroused_stationary_index_check = list(chain(*Aroused_stationary_index_check1))

    ###------------------------------Calculate Rest state-----------------------------###
    pupil_modified = [pupil[i] if dark_mask[i] else 1 for i in range(pupil.shape[0])]
    facemotion_modified = [facemotion[i] if not dark_mask[i] else 1 for i in range(len(facemotion))]
    id_rest = np.extract((np.array(pupil_modified < pupil_threshold) | np.array(facemotion_modified < facemotion_threshold)) * np.array(speed < speed_threshold), ids)
    Rest_index_temp, _, _  = find_intervals(id_rest, np.min([round(0.3 * fs), min_states_window['rest']]), real_time)
    
    delet_IDX = []
    rest_index_check = list(chain(*Rest_index_temp))
    for i in rest_index_check:
        if i in Aroused_Running_index_check or i in Aroused_stationary_index_check:
            delet_IDX.append(i)
    mask = np.isin(rest_index_check, delet_IDX, invert=True)
    result = np.extract(mask, rest_index_check)

    rest_index, rest_window, Real_Time_rest = find_intervals(result, min_states_window['rest'], real_time, 0, min_inter_interval_size)

    ###------------------------------Undefined states-----------------------------###
    to_delete__idx = []
    for i in undefined_state_idx:
        if i in list(chain(*Aroused_stationary_index)) or\
                i in list(chain(*Aroused_Running_index)) or\
                i in list(chain(*rest_index)):
            to_delete__idx.append(i)
    mask = np.isin(undefined_state_idx, to_delete__idx, invert=True)
    result = np.extract(mask, undefined_state_idx)
    _,  Undefined_state_window, Real_time_Undefined_state = find_intervals(result, 0, real_time)

    for window in Undefined_state_window :
        if window[0] != 0 :
            window[0] = window[0] - 1
        if window[-1] != len(real_time) - 1 :
            window[-1] = window[-1] + 1

    Real_Time_states = {'run' : Real_Time_Aroused_Running, 
                        'AS' : Real_time_Aroused_stationary,
                        'rest' : Real_Time_rest,
                        'undefined' : Real_time_Undefined_state}
    states_window = {'run' : Aroused_Running_window, 
                     'AS' : Aroused_stationary_window, 
                     'rest' : rest_window,
                     'undefined' : Undefined_state_window}

    return Real_Time_states, states_window

def split_stages_locomotion(speed, speed_threshold:float, real_time, min_states_window:dict, 
                 fs:float, speed_filter_kernel=0):
    """
    October 2023 - Bacci Lab - faezeh.rabbani97@gmail.com

    .......................................................................

    This function splits fluorescence activity to different time windows based
    on duration of running, speed of running and standard deviation of whisking
    final results contains maximum 4 movement states:
    1. Active movement: running with the speed of above 0.5 cm/s for more than 2 seconds
    2. Aroused stationary: aroused state without locomotion
    3. Rest: time periods which there are no whisking or movements
    \n.....................................................................
    
    :param speed: speed trace
    :param speed_threshold: speed threshold in cm/s used to determine running periods
    :param real_time: real time stamps extracted from xml file
    :param min_states_window: dictionnary with minimum dt in s for each states
    :param fs: sample frequency of data
    :param speed_filter_kernel: kernel size for filtering speed trace

    :return Real_Time_states: dictionnary with windows of real timestamps for all states activity (s)
    :return states_window: dictionnary with windows of index for all states activity (frames)
    """

    undefined_state_idx = np.arange(0,len(speed))
    ids = np.arange(0, len(speed))
    exclude_start_session = round(1 * fs)
    min_inter_interval_size = round(np.min([min_states_window['run']/2, min_states_window['rest']/2, min_states_window['AS']/2]))

    if speed_filter_kernel > 0 :
        speed = gaussian_filter1d(speed, speed_filter_kernel)
    
    ###------------------------------ Calculate active movement state -----------------------------###
    # default param: duration > 60 frames and speed > 0.5 s/m
    id_above_thr_speed = np.extract(speed[exclude_start_session:] >= speed_threshold, ids[exclude_start_session:])
    Aroused_Running_index, Aroused_Running_window, Real_Time_Aroused_Running = find_intervals(id_above_thr_speed, min_states_window['run'], real_time, round(0.3 * fs), min_inter_interval_size)
    
    Aroused_Running_index_check1 = []
    for i in Aroused_Running_index:
        if i[-1] < len(speed) - round(1 * fs) :
            ST = np.arange(i[0], i[-1] + round(1 * fs))
        else:
            ST = np.arange(i[0], len(speed))
        Aroused_Running_index_check1.append(ST)
    Aroused_Running_index_check = list(chain(*Aroused_Running_index_check1))

    ###------------------------------Calculate Rest state-----------------------------###
    id_rest = np.extract(np.array(speed < speed_threshold), ids)
    Rest_index_temp, _, _  = find_intervals(id_rest, np.min([round(0.3 * fs), min_states_window['rest']]), real_time)
    
    delet_IDX = []
    rest_index_check = list(chain(*Rest_index_temp))
    for i in rest_index_check:
        if i in Aroused_Running_index_check :
            delet_IDX.append(i)
    mask = np.isin(rest_index_check, delet_IDX, invert=True)
    result = np.extract(mask, rest_index_check)

    rest_index, rest_window, Real_Time_rest = find_intervals(result, min_states_window['rest'], real_time, 0, min_inter_interval_size)

    ###------------------------------Undefined states-----------------------------###
    to_delete__idx = []
    for i in undefined_state_idx:
        if i in list(chain(*Aroused_Running_index)) or\
                i in list(chain(*rest_index)):
            to_delete__idx.append(i)
    mask = np.isin(undefined_state_idx, to_delete__idx, invert=True)
    result = np.extract(mask, undefined_state_idx)
    _,  Undefined_state_window, Real_time_Undefined_state = find_intervals(result, 0, real_time)

    for window in Undefined_state_window :
        if window[0] != 0 :
            window[0] = window[0] - 1
        if window[-1] != len(real_time) - 1 :
            window[-1] = window[-1] + 1

    Real_Time_states = {'run' : Real_Time_Aroused_Running, 
                        'AS' : [],
                        'rest' : Real_Time_rest,
                        'undefined' : Real_time_Undefined_state}
    states_window = {'run' : Aroused_Running_window, 
                     'AS' : [],
                     'rest' : rest_window,
                     'undefined' : Undefined_state_window}

    return Real_Time_states, states_window

def stage_plot(speed, motion, pupil, dF, real_time, Real_Time_states:dict, states_window:dict, 
               save_dir:str, speed_th:float, behav_th:float, method='std', threshold_on="pupil",
               speed_filter_kernel=0, motion_filter_kernel=0,  pupil_filter_kernel=0, dFoF_filter_kernel=0, svg=False):
    
    marker_idx = []
    mean_dF = np.mean(dF, 0)

    if speed_filter_kernel > 0 :
        speed = gaussian_filter1d(speed, speed_filter_kernel)
    if motion_filter_kernel > 0 :
        motion = gaussian_filter1d(motion, motion_filter_kernel)
    if pupil_filter_kernel > 0 :
        pupil = gaussian_filter1d(pupil, pupil_filter_kernel)
    if dFoF_filter_kernel > 0 :
        mean_dF = gaussian_filter1d(mean_dF, dFoF_filter_kernel)

    # Apply Min-Max Scaling
    scaler_motion = MinMaxScaler()
    scaler_pupil = MinMaxScaler()
    normalized_motion = scaler_motion.fit_transform(motion.reshape(-1, 1)).reshape(-1)
    normalized_pupil = scaler_pupil.fit_transform(pupil.reshape(-1, 1)).reshape(-1)
    
    if method == 'std' :
        if threshold_on == 'facemotion' :
            behav_th = behav_th*(np.std(normalized_motion))
        elif threshold_on == "pupil" :
            behav_th = behav_th*(np.std(normalized_pupil))
    elif method == 'quantile' :
        if threshold_on == 'facemotion' :
            behav_th = np.quantile(normalized_motion, behav_th)
        elif threshold_on == "pupil" :
            behav_th = np.quantile(normalized_pupil, behav_th)

    colors, positions = ['darkred', 'lightgray'], [0, 1]
    cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(positions, colors)), N=3)

    colors_list =  [cmap(i) for i in range(3)] + ['lightsteelblue']
    c = '#483c32'
    alphas_list = [[0.2,1], [0,1], [0,1], [0,1]]

    states_names = ['Running', 'AS', 'Rest', 'Undefined']

    ##------------------------------------Plotting_states-----------------------------###

    # Set plot parameters
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    
    # Create plot
    fig, axs = plt.subplots(4, 1, figsize=(25, 7))

    axs[0].plot(real_time, normalized_pupil, color=c)
    axs[0].set_ylabel('Pupil')
    axs[0].margins(x=0)
    axs[0].set_xticks([])

    axs[1].plot(real_time, normalized_motion, color=c)
    axs[1].set_ylabel('Motion')
    axs[1].margins(x=0)
    axs[1].set_xticks([])

    axs[2].plot(real_time, speed, color=c)
    axs[2].set_ylabel('Speed (cm/s)')
    axs[2].margins(x=0)
    axs[2].set_xticks([])

    axs[3].plot(real_time, mean_dF, color=c)
    axs[3].set_ylabel(r'$\Delta$F/F')
    axs[3].margins(x=0)
    axs[3].set_xlabel('Time (s)')

    # Speed threshold
    tr = len(dF[0])*[speed_th]
    axs[2].plot(real_time, tr, color = '#555555', linestyle='--', label='running th')

    # Facemotion threshold
    if threshold_on == 'facemotion' :
        axs[1].axhline(behav_th, color = '#555555', linestyle='--', label='motion th')
        axs[1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})

    # Pupil threshold
    elif threshold_on == 'pupil':
        axs[0].axhline(behav_th, color = '#555555', linestyle='--', label='pupil th')
        axs[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
    
    axs[2].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})

    # Add color block to indicate states
    for k in range(4):
        for i in range(len(Real_Time_states['run'])):
            axs[k].axvspan(Real_Time_states['run'][i][0], Real_Time_states['run'][i][-1], color=colors_list[0], alpha=0.5)
        for i in range(len(Real_Time_states['AS'])):
            axs[k].axvspan(Real_Time_states['AS'][i][0], Real_Time_states['AS'][i][-1], color=colors_list[1], alpha=0.5)
        for i in range(len(Real_Time_states['rest'])):
            axs[k].axvspan(Real_Time_states['rest'][i][0], Real_Time_states['rest'][i][-1], color=colors_list[2], alpha=0.5)
    
    # Legend of the blocks
    Running = Line2D([0], [0], color=colors_list[0], linewidth=5)
    Aroused_stationary = Line2D([0], [0], color=colors_list[1], linewidth=5)
    rest = Line2D([0], [0], color=colors_list[2], linewidth=5)
    axs[3].legend([Running, Aroused_stationary, rest],
                 states_names[:-1],
                 loc='lower left', bbox_to_anchor=(1.0, 0.0), prop={'size': 9})
    
    if svg == True:
        save_direction_svg = os.path.join(save_dir, "activity_states_" + threshold_on +".svg")
        fig.savefig(save_direction_svg,format = 'svg')

    save_path = os.path.join(save_dir, "activity_states_" + threshold_on)
    fig.savefig(save_path)
    plt.close(fig)

     ###------------------------------Plotting_states2-----------------------------###
    for key in states_window.keys() :
        marker_idx.append(list(chain(*states_window[key])))
    line_seg_list = [np.array([]), np.array([]), np.array([]), np.array([])]
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False, "axes.spines.left": False}
    sns.set_theme(style="ticks", rc=custom_params)

    #------------------
    fig2, ax = plt.subplots(4, 1, figsize=(25, 7))
    
    zscored_pupil = zscore(pupil)
    xy_vals = np.transpose([real_time, zscored_pupil])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[0].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[0].set_ylim(min(zscored_pupil), max(zscored_pupil))
    ax[0].set_ylabel('Pupil z-score')
    ax[0].set_xticks([])
    ax[0].margins(x=0)

    #--------------------
    zscored_motion = zscore(motion)
    xy_vals = np.transpose([real_time, zscored_motion])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[1].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[1].set_ylim(min(zscored_motion), max(zscored_motion))
    ax[1].set_ylabel('Motion z-score')
    ax[1].set_xticks([])
    ax[1].margins(x=0)

    #------------------
    xy_vals = np.transpose([real_time, speed])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[2].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[2].set_ylim(min(speed), max(speed))
    ax[2].set_ylabel('Speed (cm/s)')
    ax[2].set_xticks([])
    ax[2].margins(x=0)

    #------------------
    xy_vals = np.transpose([real_time, mean_dF])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[3].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[3].set_ylim(min(mean_dF), max(mean_dF))
    ax[3].set_ylabel(r'$\Delta$F/F')
    ax[3].set_xlabel('Time (s)')
    ax[3].margins(x=0)

    #------------------
    Running = Line2D([0], [0], color=colors_list[0], linewidth=5)
    Aroused_stationary = Line2D([0], [0], color=colors_list[1], linewidth=5)
    rest = Line2D([0], [0], color=colors_list[2], linewidth=5)
    undefined = Line2D([0], [0], color=colors_list[3], linewidth=5)
    ax[1].legend([Running, Aroused_stationary, rest, undefined],
                 states_names,
                 loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 9})
    
    #------------------
    if svg == True:
        save_direction_svg = os.path.join(save_dir, "activity_states2_" + threshold_on + ".svg")
        fig2.savefig(save_direction_svg,format = 'svg')

    save_path = os.path.join(save_dir, "activity_states2_" + threshold_on)
    fig2.savefig(save_path)
    plt.close(fig2)

def stage_plot_mixed(speed, motion, pupil, dF, real_time, Real_Time_states:dict, states_window:dict,
                     idx_lim_dark:list, fs:float,
                     save_dir:str, speed_th:float, pupil_th:float, facemotion_th:float, 
                     pupil_method='std', facemotion_method="std",
                     speed_filter_kernel=0, motion_filter_kernel=0,  pupil_filter_kernel=0, dFoF_filter_kernel=0, svg=False):
    
    marker_idx = []
    mean_dF = np.mean(dF, 0)

    if speed_filter_kernel > 0 :
        speed = gaussian_filter1d(speed, speed_filter_kernel)
    if motion_filter_kernel > 0 :
        motion = gaussian_filter1d(motion, motion_filter_kernel)
    if pupil_filter_kernel > 0 :
        pupil = gaussian_filter1d(pupil, pupil_filter_kernel)
    if dFoF_filter_kernel > 0 :
        mean_dF = gaussian_filter1d(mean_dF, dFoF_filter_kernel)

    # Apply Min-Max Scaling
    scaler_motion = MinMaxScaler()
    scaler_pupil = MinMaxScaler()
    normalized_motion = scaler_motion.fit_transform(motion.reshape(-1, 1)).reshape(-1)
    normalized_pupil = scaler_pupil.fit_transform(pupil.reshape(-1, 1)).reshape(-1)
    
    pupil_without_dark, dark_mask = get_pupil_without_dark(normalized_pupil, idx_lim_dark, round(3*fs))

    if pupil_method == 'std' :
        pupil_th = pupil_th*(np.std(pupil_without_dark))
    elif pupil_method == 'quantile' :
        pupil_th = np.quantile(pupil_without_dark, pupil_th)
    
    if facemotion_method == 'std' :
        facemotion_th = facemotion_th*(np.std(normalized_motion))
    elif facemotion_method == 'quantile' :
        facemotion_th = np.quantile(normalized_motion, facemotion_th)

    idx_lim_light = get_light_intervals(dark_mask)

    colors, positions = ['darkred', 'lightgray'], [0, 1]
    cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(positions, colors)), N=3)

    colors_list =  [cmap(i) for i in range(3)] + ['lightsteelblue']
    c = '#483c32'
    alphas_list = [[0.2,1], [0,1], [0,1], [0,1]]

    states_names = ['Running', 'AS', 'Rest', 'Undefined']

    ##------------------------------------Plotting_states-----------------------------###

    # Set plot parameters
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    
    # Create plot
    fig, axs = plt.subplots(4, 1, figsize=(25, 7))

    axs[0].plot(real_time, normalized_pupil, color=c)
    axs[0].set_ylabel('Pupil')
    axs[0].margins(x=0)
    axs[0].set_xticks([])

    axs[1].plot(real_time, normalized_motion, color=c)
    axs[1].set_ylabel('Motion')
    axs[1].margins(x=0)
    axs[1].set_xticks([])

    axs[2].plot(real_time, speed, color=c)
    axs[2].set_ylabel('Speed (cm/s)')
    axs[2].margins(x=0)
    axs[2].set_xticks([])

    axs[3].plot(real_time, mean_dF, color=c)
    axs[3].set_ylabel(r'$\Delta$F/F')
    axs[3].margins(x=0)
    axs[3].set_xlabel('Time (s)')

    # Speed threshold
    tr = len(dF[0])*[speed_th]
    axs[2].plot(real_time, tr, color = '#555555', linestyle='--', label='running th')

    # Facemotion threshold
    for el in idx_lim_dark : 
        axs[1].plot(real_time[el[0]:el[1]], [facemotion_th]*len(real_time[el[0]:el[1]]), color = '#555555', linestyle='--', label='facemotion th')
        #axs[1].axhline(facemotion_th, xmin=el[0], xmax=el[1], color = '#555555', linestyle='--', label='motion th')

    # Pupil threshold
    for el in idx_lim_light :
        axs[0].plot(real_time[el[0]:el[1]], [pupil_th]*len(real_time[el[0]:el[1]]), color = '#555555', linestyle='--', label='pupil th')
        #axs[0].axhline(pupil_th, xmin=el[0], xmax=el[1], color = '#555555', linestyle='--', label='pupil th')
    
    axs[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
    axs[2].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})

    # Add color block to indicate states
    for k in range(4):
        for i in range(len(Real_Time_states['run'])):
            axs[k].axvspan(Real_Time_states['run'][i][0], Real_Time_states['run'][i][-1], color=colors_list[0], alpha=0.5)
        for i in range(len(Real_Time_states['AS'])):
            axs[k].axvspan(Real_Time_states['AS'][i][0], Real_Time_states['AS'][i][-1], color=colors_list[1], alpha=0.5)
        for i in range(len(Real_Time_states['rest'])):
            axs[k].axvspan(Real_Time_states['rest'][i][0], Real_Time_states['rest'][i][-1], color=colors_list[2], alpha=0.5)
    
    # Legend of the blocks
    Running = Line2D([0], [0], color=colors_list[0], linewidth=5)
    Aroused_stationary = Line2D([0], [0], color=colors_list[1], linewidth=5)
    rest = Line2D([0], [0], color=colors_list[2], linewidth=5)
    axs[3].legend([Running, Aroused_stationary, rest],
                 states_names[:-1],
                 loc='lower left', bbox_to_anchor=(1.0, 0.0), prop={'size': 9})
    
    if svg == True:
        save_direction_svg = os.path.join(save_dir, "activity_states_pupil.svg")
        fig.savefig(save_direction_svg,format = 'svg')

    save_path = os.path.join(save_dir, "activity_states_pupil")
    fig.savefig(save_path)
    plt.close(fig)

     ###------------------------------Plotting_states2-----------------------------###
    for key in states_window.keys() :
        marker_idx.append(list(chain(*states_window[key])))
    line_seg_list = [np.array([]), np.array([]), np.array([]), np.array([])]
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False, "axes.spines.left": False}
    sns.set_theme(style="ticks", rc=custom_params)

    #------------------
    fig2, ax = plt.subplots(4, 1, figsize=(25, 7))
    
    zscored_pupil = zscore(pupil)
    xy_vals = np.transpose([real_time, zscored_pupil])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[0].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[0].set_ylim(min(zscored_pupil), max(zscored_pupil))
    ax[0].set_ylabel('Pupil z-score')
    ax[0].set_xticks([])
    ax[0].margins(x=0)

    #--------------------
    zscored_motion = zscore(motion)
    xy_vals = np.transpose([real_time, zscored_motion])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[1].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[1].set_ylim(min(zscored_motion), max(zscored_motion))
    ax[1].set_ylabel('Motion z-score')
    ax[1].set_xticks([])
    ax[1].margins(x=0)

    #------------------
    xy_vals = np.transpose([real_time, speed])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[2].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[2].set_ylim(min(speed), max(speed))
    ax[2].set_ylabel('Speed (cm/s)')
    ax[2].set_xticks([])
    ax[2].margins(x=0)

    #------------------
    xy_vals = np.transpose([real_time, mean_dF])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[3].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[3].set_ylim(min(mean_dF), max(mean_dF))
    ax[3].set_ylabel(r'$\Delta$F/F')
    ax[3].set_xlabel('Time (s)')
    ax[3].margins(x=0)

    #------------------
    Running = Line2D([0], [0], color=colors_list[0], linewidth=5)
    Aroused_stationary = Line2D([0], [0], color=colors_list[1], linewidth=5)
    rest = Line2D([0], [0], color=colors_list[2], linewidth=5)
    undefined = Line2D([0], [0], color=colors_list[3], linewidth=5)
    ax[1].legend([Running, Aroused_stationary, rest, undefined],
                 states_names,
                 loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 9})
    
    #------------------
    if svg == True:
        save_direction_svg = os.path.join(save_dir, "activity_states2_pupil.svg")
        fig2.savefig(save_direction_svg,format = 'svg')

    save_path = os.path.join(save_dir, "activity_states2_pupil")
    fig2.savefig(save_path)
    plt.close(fig2)

def stage_plot_locomotion(speed, dF, real_time, Real_Time_states:dict, states_window:dict, 
                            save_dir:str, speed_th:float,
                            speed_filter_kernel=0, dFoF_filter_kernel=0, svg=False):
    
    marker_idx = []
    mean_dF = np.mean(dF, 0)

    if speed_filter_kernel > 0 :
        speed = gaussian_filter1d(speed, speed_filter_kernel)
    if dFoF_filter_kernel > 0 :
        mean_dF = gaussian_filter1d(mean_dF, dFoF_filter_kernel)

    colors, positions = ['darkred', 'lightgray'], [0, 1]
    cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(positions, colors)), N=3)

    colors_list =  [cmap(i) for i in range(3)] + ['lightsteelblue']
    c = '#483c32'
    alphas_list = [[0.2,1], [0,1], [0,1], [0,1]]

    states_names = ['Running', 'AS', 'Rest', 'Undefined']

    ##------------------------------------Plotting_states-----------------------------###

    # Set plot parameters
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    
    # Create plot
    fig, axs = plt.subplots(2, 1, figsize=(25, 7))

    axs[0].plot(real_time, speed, color=c)
    axs[0].set_ylabel('Speed (cm/s)')
    axs[0].margins(x=0)
    axs[0].set_xticks([])

    axs[1].plot(real_time, mean_dF, color=c)
    axs[1].set_ylabel(r'$\Delta$F/F')
    axs[1].margins(x=0)
    axs[1].set_xlabel('Time (s)')

    # Speed threshold
    tr = len(dF[0])*[speed_th]
    axs[0].plot(real_time, tr, color = '#555555', linestyle='--', label='running th')
    axs[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})

    # Add color block to indicate states
    for k in range(2):
        for i in range(len(Real_Time_states['run'])):
            axs[k].axvspan(Real_Time_states['run'][i][0], Real_Time_states['run'][i][-1], color=colors_list[0], alpha=0.5)
        for i in range(len(Real_Time_states['rest'])):
            axs[k].axvspan(Real_Time_states['rest'][i][0], Real_Time_states['rest'][i][-1], color=colors_list[2], alpha=0.5)
    
    # Legend of the blocks
    Running = Line2D([0], [0], color=colors_list[0], linewidth=5)
    rest = Line2D([0], [0], color=colors_list[2], linewidth=5)
    axs[1].legend([Running, rest],
                  np.array(states_names)[[0, 2]],
                  loc='lower left', bbox_to_anchor=(1.0, 0.0), prop={'size': 9})
    
    if svg == True:
        save_direction_svg = os.path.join(save_dir, "activity_states.svg")
        fig.savefig(save_direction_svg,format = 'svg')

    save_path = os.path.join(save_dir, "activity_states")
    fig.savefig(save_path)
    plt.close(fig)

     ###------------------------------Plotting_states2-----------------------------###
    for key in states_window.keys() :
        marker_idx.append(list(chain(*states_window[key])))
    line_seg_list = [np.array([]), np.array([]), np.array([]), np.array([])]
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False, "axes.spines.left": False}
    sns.set_theme(style="ticks", rc=custom_params)

    #------------------
    fig2, ax = plt.subplots(2, 1, figsize=(25, 7))

    xy_vals = np.transpose([real_time, speed])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[0].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[0].set_ylim(min(speed), max(speed))
    ax[0].set_ylabel('Speed (cm/s)')
    ax[0].set_xticks([])
    ax[0].margins(x=0)

    #------------------
    xy_vals = np.transpose([real_time, mean_dF])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[1].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[1].set_ylim(min(mean_dF), max(mean_dF))
    ax[1].set_ylabel(r'$\Delta$F/F')
    ax[1].set_xlabel('Time (s)')
    ax[1].margins(x=0)

    #------------------
    Running = Line2D([0], [0], color=colors_list[0], linewidth=5)
    rest = Line2D([0], [0], color=colors_list[2], linewidth=5)
    undefined = Line2D([0], [0], color=colors_list[3], linewidth=5)
    ax[1].legend([Running, rest, undefined],
                 np.array(states_names)[[0, 2, 3]],
                 loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 9})
    
    #------------------
    if svg == True:
        save_direction_svg = os.path.join(save_dir, "activity_states2.svg")
        fig2.savefig(save_direction_svg,format = 'svg')

    save_path = os.path.join(save_dir, "activity_states2")
    fig2.savefig(save_path)
    plt.close(fig2)

def find_intervals(x, min_window_size, time, exclude_start=0, min_inter_interval_size=0, include_end=0):
    """
    :param x: list or array of indexes of interest to divide in intervals
    :param int min_window_size: minimum number of frames for each state interval
    :param time: list array of time to divide in intervals
    :param int exclude: frames that should be removed at the beginning of window. Default is 0.
    :param int min_inter_interval_size: minimum number of frames separating two intervals. Default is 0.
    """
    time_bounds = []
    index_bounds = []
    index_intervals =[]
    window = []

    if exclude_start > min_window_size :
        exclude_start = 0
        print("Minimum window size is under 1s. No frames will be removed at the beginning of the window.")

    for i in range(len(x)):
        if (x[i] + 1) in x:
            window.append(x[i])
        elif i < len(x)-1 and (x[i+1] - x[i] - include_end) < min_inter_interval_size:
            window.extend(np.arange(x[i], x[i+1]))
        else :
            window.extend(np.arange(x[i], x[i]+1+include_end))
            if len(window) >= min_window_size:
                index_intervals.append(window[exclude_start:])
            window = []
    
    for interval in index_intervals:
        time_bounds_temp = []
        index_bounds_temp = []

        index_bounds_temp.append(interval[0])
        index_bounds_temp.append(interval[-1])
        time_bounds_temp.append(time[interval[0]])
        time_bounds_temp.append(time[interval[-1]])

        index_bounds.append(index_bounds_temp)
        time_bounds.append(time_bounds_temp)

    return index_intervals, index_bounds, time_bounds

def state_duration(state_real_time):
    general_time = 0
    for window in state_real_time:
        interval = window[-1] - window[0]
        general_time += interval
    return general_time

def get_pupil_without_dark(pupil, idx_lim_dark, exclude):
    mask = (np.ones(pupil.shape[0]) == 1)
    for l in idx_lim_dark:
        temp = (np.arange(pupil.shape[0]) < l[0]) | (np.arange(pupil.shape[0]) >= l[1] + exclude)
        mask = mask & temp
    return pupil[mask], mask

def get_light_intervals(mask):
    idx_lim_light = []
    temp = []
    for i in range(len(mask)) :
        if len(temp) == 0 :
            if mask[i] == True :
                temp.append(i)
        if len(temp) == 1 :
            if mask[i] == False :
                temp.append(i)
                idx_lim_light.append(temp)
                temp = []
    if len(temp) == 1 :
        temp.append(len(mask))
        idx_lim_light.append(temp)
    
    return idx_lim_light

def time_pie(real_time_states, total_duration, save_direction, svg=False, figname="Motion_time"):

    run_dt = state_duration(real_time_states['run'])
    as_dt = state_duration(real_time_states['AS'])
    rest_dt = state_duration(real_time_states['rest'])
    not_used_dt = total_duration - (as_dt + run_dt + rest_dt)

    sizes = [run_dt , as_dt, rest_dt, not_used_dt]
    labels = ["Running", "Aroused stationnary", "Resting", "Undefined"]
    
    colors, positions = ['darkred', 'lightgray'], [0, 1]
    cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(positions, colors)), N=3)
    colors = [cmap(i) for i in range(3)] + ['lightsteelblue']
    explode = (0.05, 0.05, 0.05, 0.05)
               
    fig = plt.figure(figsize=(10, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', labeldistance=None, explode=explode)
    plt.legend(labels=labels, loc='center left', bbox_to_anchor=(0.8, 0.3), frameon=False)
    plt.axis('equal')
    plt.title("Motion states duration")

    if svg == True:
        save_direction_svg = os.path.join(save_direction, figname + ".svg")
        fig.savefig(save_direction_svg)
    fig.savefig(os.path.join(save_direction, figname + ".png"))
    plt.close()

    return run_dt / total_duration, as_dt / total_duration, rest_dt / total_duration

def time_pie_locomotion(real_time_states, total_duration, save_direction, svg=False, figname="Motion_time"):

    run_dt = state_duration(real_time_states['run'])
    rest_dt = state_duration(real_time_states['rest'])
    not_used_dt = total_duration - (run_dt + rest_dt)

    sizes = [run_dt, rest_dt, not_used_dt]
    labels = ["Running", "Resting", "Undefined"]
    
    colors = ['darkred', 'lightgray', 'lightsteelblue']
    explode = (0.05, 0.05, 0.05)
               
    fig = plt.figure(figsize=(10, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', labeldistance=None, explode=explode)
    plt.legend(labels=labels, loc='center left', bbox_to_anchor=(0.8, 0.3), frameon=False)
    plt.axis('equal')
    plt.title("Motion states duration")

    if svg == True:
        save_direction_svg = os.path.join(save_direction, figname + ".svg")
        fig.savefig(save_direction_svg)
    fig.savefig(os.path.join(save_direction, figname + ".png"))
    plt.close()

    return run_dt / total_duration, rest_dt / total_duration

def sort_dict_el(d: dict):
    l_el, l_key = [], []
    for key in d.keys():
        for i in range(len(d[key])):
            l_el.append(d[key][i])
            l_key.append(key)

    return sorted(zip(l_el, l_key))