import sys
sys.path.append("./src")
import visualpipe.analysis.speed_around_stim as Speed_around_stim
import visualpipe.post_analysis.utils as utils
import os
import numpy as np
import matplotlib.pyplot as plt



excel_sheet_path = r"Y:\raw-imaging\Nathan\Nathan_sessions_visualpipe.xlsx"
save_path = r"Y:\raw-imaging\Nathan\PYR\Visualpipe_postanalysis\looming-sweeping-log\Analysis"
protocol_type = 'looming-sweeping-log'
sub_protocols = ['looming-stim-log-1.0', 'looming-stim-log-0.4', 'looming-stim-log-0.1', 'looming-stim-log-0.0']
groups_id = {'WT': 0, 'KO': 1}  # keys are group names, e.g 'WT': 0, 'KO': 1



df = utils.load_excel_sheet(excel_sheet_path, protocol_type)

import numpy as np
import visualpipe.analysis.speed_around_stim as Speed_around_stim
import visualpipe.post_analysis.utils as utils

def average_speed_stim(df, groups_id, sub_protocols, variable = 'Speed', pre_time=2.0, post_time=7.0):
    """
    For each genotype group:
        - Loops over sessions
        - Extracts per-protocol speed traces
        - Computes the mean speed trace across trials for this session
        - Stores session-averaged traces
        - Trims all session traces for a protocol to the minimum length for consistent plotting
    """
    group_results = {}

    for key in groups_id.keys():
        print(f"\n-------------------------- Processing {key} group --------------------------")
        df_filtered = df[df["Genotype"] == key]
        protocol_container = {}

        # Temporary storage of session traces before trimming
        temp_traces = {}

        for k in range(len(df_filtered)):
            mouse_id = df_filtered["Mouse_id"].iloc[k]
            session_id = df_filtered["Session_id"].iloc[k]
            session_path = df_filtered["Session_path"].iloc[k]
            output_id = df_filtered["Output_id"].iloc[k]
            analysis_id = f"{session_id}_output_{output_id}"

            print(f"\nSession id: {session_id}\n  Mouse id : {mouse_id}\n     Session path: {session_path}")

            _, traces = Speed_around_stim.process_session(session_path, analysis_id, variable, pre_time, post_time)

            # Loop over protocols
            for pname, trials_list in traces.items():
                if len(trials_list) == 0 or pname not in sub_protocols:
                    continue
                print(f"   Processing protocol: {pname}")
                # Convert trials to array and average across trials
                min_len_trial = min(len(trial['speed']) for trial in trials_list)
                all_speeds = np.array([trial['speed'][:min_len_trial] for trial in trials_list])
                session_mean = np.mean(all_speeds, axis=0)
                session_time = trials_list[0]['time'][:min_len_trial]

                if pname not in temp_traces:
                    temp_traces[pname] = {
                        'time_list': [session_time],
                        'speed_traces': [session_mean],
                        'sessions': [session_id]
                    }
                else:
                    temp_traces[pname]['time_list'].append(session_time)
                    temp_traces[pname]['speed_traces'].append(session_mean)
                    temp_traces[pname]['sessions'].append(session_id)

        # Trim all session traces to minimum length for this protocol
        for pname, pdata in temp_traces.items():
            min_len_session = min(len(tr) for tr in pdata['speed_traces'])
            trimmed_traces = np.array([tr[:min_len_session] for tr in pdata['speed_traces']])
            time_vector = pdata['time_list'][0][:min_len_session]  # take first session as reference

            protocol_container[pname] = {
                'time': time_vector,
                'speed_traces': trimmed_traces,
                'sessions': pdata['sessions']
            }

        group_results[key] = protocol_container

    return group_results


def plot_behavior(group_results, sub_protocols, save_path, variable = 'Speed'):
    """
    Plot average speed traces for each genotype group and protocol.
    """
    colors = {'WT': 'blue', 'KO': 'red'}
    for pname in sub_protocols:
        plt.figure(figsize=(10, 6))
        plt.suptitle(f"Average {variable} Traces - Protocol: {pname}", fontsize=16)
        for group_name, protocols in group_results.items():
            if pname in protocols:
                time_vector = protocols[pname]['time']
                all_traces = np.array(protocols[pname]['speed_traces'])
                mean_trace = np.mean(all_traces, axis=0)
                sem_trace = np.std(all_traces, axis=0) / np.sqrt(all_traces.shape[0])
                plt.plot(time_vector, mean_trace, color = colors[group_name], label=f"{group_name} (n={all_traces.shape[0]})")
                plt.fill_between(time_vector, mean_trace - sem_trace, mean_trace + sem_trace, color = colors[group_name], alpha=0.3)
                plt.xlabel("Time (s)")
                plt.ylabel("Speed (cm/s)" if variable=='Speed' else variable)
        plt.legend()
        plt.savefig(os.path.join(save_path, f"Average_{variable}_{pname}.png"))
        plt.show()

group_results = average_speed_stim(df, groups_id, sub_protocols, 'FaceMotion', pre_time=2.0, post_time=7.0)
plot_behavior(group_results, sub_protocols, save_path, 'FaceMotion')

