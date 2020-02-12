# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:02:39 2020

@author: Oskari

Tests the code by running it for a single trajectory

"""
import numpy as np
import argparse
import json
import time
import os, sys
import dill
from main import *


if __name__ == "__main__":
    #Get arguments from command line and parse them
    parser = argparse.ArgumentParser(description="A script for studying non-adiabatic transitions")
    parser.add_argument("run_dir", help = "Run directory")
    parser.add_argument("options_fname", help = "Filename of the options file")
    parser.add_argument("result_fname", help = "Filename for storing results")
    parser.add_argument("jobs_fname", help = "Filename for storing jobs")
    
    args = parser.parse_args()
    
    #Load options file
    with open(args.run_dir + '/options/' + args.options_fname) as options_file:
        options_dict = json.load(options_file)
    
    #Add run directory and options filename to options_dict
    options_dict["run_dir"] = args.run_dir
    options_dict["options_fname"] = args.options_fname
    options_dict["result_fname"] = args.result_fname + time.strftime('_%Y-%m-%d_%H-%M-%S') +'.txt'
    options_dict["jobs_fname"] = args.jobs_fname
    
    #Make a jobs file
    jobs_files = generate_jobs_file(options_dict)
    
    #Generate the string that exectutes the state propagation program
    cluster_params = options_dict["cluster_params"]
    run_dir = options_dict["run_dir"]
    jobs_fname = options_dict["jobs_fname"]
    options_fname = options_dict["options_fname"]
    result_fname = options_dict["result_fname"]
    exec_str =  ("python "  + '"'+cluster_params["prog"] + '"'
                            +  ' "' + run_dir +  '"' + " " + options_fname + " " + result_fname
                            + ' --save_fields')
    print(exec_str)
    os.system(exec_str)
