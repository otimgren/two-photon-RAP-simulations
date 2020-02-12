# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:33:51 2020

@author: Oskari
"""

import numpy as np
import argparse
import json
import time
import os, sys



def generate_jobs_file(options_dict):
    """
    This is a function that generates a text file with the command line arguments
    to run jobs using "dead Simple Queue" (dSQ) on the Yale HPC cluster.
    
    input on command line:
    run_dir = path to directory in which the code is run
    options_fname = path to options file that specifies parameters for the jobs
    jobs_fname = path to output file that is used to with dSQ to generate a jobs array
    
    output:
    a text file containing the commands needed to submit jobs to cluster using dSQ
    """
    #Get some parameters from options_dict
    cluster_params = options_dict["cluster_params"]
    run_dir = options_dict["run_dir"]
    jobs_fname = options_dict["jobs_fname"]
    options_fname = options_dict["options_fname"]
    result_fname = options_dict["result_fname"]
    
    #Generate arrays for field parameters
    param_names = []
    params = []
    units = []
    for param_name, param_dict in options_dict["params"].items():
        param, unit = generate_param_array(param_dict)
        param_names.append(param_name)
        params.append(param)
        units.append(unit)
        
    #Generate a table of the field parameters to use for each job
    array_list = np.meshgrid(*params)
    flattened_list = []
    for array in array_list:
        flattened_list.append(array.flatten())
    field_param_table = np.vstack(flattened_list).T
    
    #Open the text file that is used for the jobsfile
    with open(run_dir + '/jobs_files/' + jobs_fname, 'w+') as f:
        #Loop over rows of the parameter table
        for row in field_param_table:
            #Extract the parameters for this job
            param_dict = {}
            for i, param_value in enumerate(row):
                param_name = param_names[i]
                param_dict[param_name] = param_value
                
                
            #Start printing into the jobs file 
            #Load the correct modules
            print("module load miniconda", file=f, end = '; ')
            print("source deactivate", file=f, end = '; ')
            print("source activate non_adiabatic", file=f, end = '; ')
            
            #Generate the string that executes the program and gives it parameters
            exec_str =  ("python " + cluster_params["prog"] + " "
                            + run_dir + " " + options_fname + " " 
                            + result_fname + " ")
            
            for param_name, param_value in param_dict.items():
                exec_str += "--{} {} ".format(param_name, param_value)
                
            print(exec_str, file=f)
    
    #Also initialize the results file
    with open(run_dir + '/results/' + result_fname, 'w+') as f:      
        #Print headers for the results
        headers = ['Probability']
        for param_name, unit in zip(param_names,units):
            headers.append(param_name +'/'+unit)
        headers_str = '\t\t'.join(headers)
        print(headers_str, file = f)


def generate_batchfile(options_dict):
    """
    Function that generates a batchfile basd on a given jobs file using dSQ
    """
    #Settings for dSQ
    cluster_params = options_dict["cluster_params"]
    memory_per_cpu = cluster_params["mem-per-cpu"]
    time = cluster_params["time"]
    mail_type = cluster_params["mail-type"]
    cpus_per_task = cluster_params["cpus-per-task"]
    
    #Setting paths
    run_dir = options_dict['run_dir']
    jobs_fname = options_dict['jobs_fname']
    jobs_path =run_dir + '/jobs_files/' + jobs_fname
    batchfile_path = run_dir + '/slurm/' + 'dsq-' +jobs_fname
    
    #Generate the string to execute
    exec_str = ('dsq --job-file ' + jobs_path + ' --mem-per-cpu ' + memory_per_cpu
                +' -t ' + time + ' --mail-type '+ mail_type + ' -o /dev/null --batch-file '
                + batchfile_path)
    print(exec_str)
    
    os.system(exec_str)
    
    #Write which partition to use to the file on line 2
    with open(batchfile_path, 'r') as f:
        lines = f.readlines()
    
    text = ('#SBATCH --partition '+options_dict["cluster_params"]["partition"])
    text += ('\n#SBATCH --cpus-per-task ' + cpus_per_task)
    if options_dict["cluster_params"]["requeue"]:
        text += '\n#SBATCH --requeue\n'
    lines.insert(1, text)
    
    with open(batchfile_path, 'w') as f:
        f.writelines(lines)
    
    #Return the path to the batchfile
    return batchfile_path

def generate_param_array(param_dict):
    """
    Function that generates an array of values for a scan over a field parameter,
    e.g. z-component of electric field
    
    input:
    param_dict =  dictionary that specifies if parameter is to be scanned, what
    value the parameter should take etc.
    
    return:
    an array that contains the parameter
    """
    #Check if the parameter is supposed to be scanned
    scan = param_dict["scan"]
    
    #Two cases: parameter is scanned or not scanned
    if scan:
        #If parameter is scanned, find the parameters for the scan
        p_ini = param_dict["min"]
        p_fin = param_dict["max"]
        N = param_dict["N"]
        param = np.linspace(p_ini, p_fin, N)
        
    else:
        #If not scanned, set value
        value = param_dict["value"]
        param = np.array(value)
        
    #If the unit is specified, also get that
    try:
        unit = param_dict["unit"]
    except ValueError:
        pass
    
    return param,unit


if __name__ == "__main__":
    #Get arguments from command line and parse them
    parser = argparse.ArgumentParser(description="A script for studying non-adiabatic transitions")
    parser.add_argument("run_dir", help = "Run directory")
    parser.add_argument("options_fname", help = "Filename of the options file")
    parser.add_argument("result_fname", help = "Filename for storing results")
    parser.add_argument("jobs_fname", help = "Filename for storing jobs")
    parser.add_argument("--jobs", help="If true, generate jobsfile"
                    , action = "store_true")
    parser.add_argument("--batch", help="If true, generate jobsfile and batchfile from that"
                    , action = "store_true")
    parser.add_argument("--submit", help="If true, generate batchfile and submit to cluster"
                        , action = "store_true")
    
    args = parser.parse_args()
    
    #Load options file
    with open(args.run_dir + '/options/' + args.options_fname) as options_file:
        options_dict = json.load(options_file)
    
    #Add run directory and options filename to options_dict
    options_dict["run_dir"] = args.run_dir
    options_dict["options_fname"] = args.options_fname
    options_dict["result_fname"] = args.result_fname + time.strftime('_%Y-%m-%d_%H-%M-%S') +'.txt'
    options_dict["jobs_fname"] = args.jobs_fname
    
    #Different uses of main.py. Can either make a job file, job file + batch file,
    #or make both files and submit
    if args.jobs:
        generate_jobs_file(options_dict)
        
    elif args.batch:
        generate_jobs_file(options_dict)
        generate_batchfile(options_dict)
    
    #Generate a jobs file and batch file, and submit to cluster
    elif args.submit:
        generate_jobs_file(options_dict)
        batchfile_path = generate_batchfile(options_dict)
        os.system("sbatch {}".format(batchfile_path))