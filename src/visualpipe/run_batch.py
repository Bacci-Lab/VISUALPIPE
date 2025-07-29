"""
This module provides a function to run a batch process for running visual pipelines on a list of file paths.
"""
from pipeline import visual_pipe

def run_batch(filepath) :
    """
    Executes a batch process for running visual pipelines on a list of file paths.

    This function reads file paths from the specified text file and processes each 
    path using the visual_pipe function. It logs successful output directories and 
    keeps track of any failed runs due to exceptions.

    Args:
        filepath (str): Path to the text file containing a list of file paths to process.

    Prints:
        - A message indicating the end of the batch process.
        - A summary of the number of failed runs, if any.
        - A list of output folders for successful runs.
    """

    failed_runs = []
    outputs_folders = []

    with open(filepath, "r") as f:
        path_lists = [line.rstrip() for line in f]
    
    for path in path_lists:
        try :
            save_dir = visual_pipe(path)
            outputs_folders.append(save_dir)
        except Exception as e :
            print(f"Pipeline failed for {path}")
            print(e)
            failed_runs.append(path)

    print('#-------------------------------------------------------------------------#')
    print('End of batch')

    if len(failed_runs) > 0 :
        print(f"Pipeline failed for {len(failed_runs)} runs")
        print(f"Failed runs : {failed_runs}")
    else :
        print("Sessions' analysis completed successfully")
    
    print(f'Output folders : {outputs_folders}')

if __name__ == "__main__":

    # Change filepath of the text file containing the list of sessions to run
    filepath = 'C:/Users/mai-an.nguyen/Downloads/test.txt'
    
    run_batch(filepath)