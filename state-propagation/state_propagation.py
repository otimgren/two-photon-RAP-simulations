# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:38:22 2020

@author: Oskari
"""

import sys
sys.path.append('./molecular-state-classes-and-functions/')
sys.path.append('./utility/')
from functions import make_QN, make_hamiltonian, make_transform_matrix, find_closest_state
from EM_fields import E_field_ring, calculate_power_needed, microwave_field
from matrix_element_functions import calculate_microwave_ED_matrix_element_mixed_state_uncoupled
from functions import find_state_idx_from_state , make_H_mu, reorder_evecs, vector_to_state
from state_propagation_functions import simulate_RAP, molecule_position, get_Hamiltonian
from datetime import datetime
import pickle
import dill
import argparse
import numpy as np
import json
from scipy.linalg.lapack import zheevd
from scipy.linalg import block_diag

if __name__ == "__main__":
    #Get path info and electric field parameters from command line
    #Get arguments from command line and parse them
    parser = argparse.ArgumentParser(description="A script for studying non-adiabatic transitions")
    parser.add_argument("run_dir", help = "Run directory")
    parser.add_argument("options_fname", help = "Filename of the options file")
    parser.add_argument("result_fname", help = "Filename for storing results")
    parser.add_argument("--save_fields", help = "If true, save the E-and B-fields",
                                                 action = "store_true")
    parser.add_argument("--z0", help = "Starting position z of molecule [m]", type = float,
                        default = -0.1)
    parser.add_argument("--vz", help = "Z-velocity of molecule [m/s]", type = float,
                        default = 200.)
    parser.add_argument("--ring_z1", help = "Position of electrode 1 [m]", type = float,
                        default = -71.4375e-3)
    parser.add_argument("--ring_z2", help = "Position of electrode 2 [m]", type = float,
                        default = 85.725e-3)
    parser.add_argument("--ring_V1", help = "Voltage on electrode 1 [V]", type = float,
                        default = 3e3)
    parser.add_argument("--ring_V2", help = "Voltage on electrode 2 [V]", type = float,
                        default = 2.7e2)
    parser.add_argument("--Omega1", help = "Rabi rate for microwave 1 [Hz]", type = float,
                        default = 200e3)
    parser.add_argument("--Omega2", help = "Rabi rate for microwave 2 [Hz]", type = float,
                        default = 200e3)
    parser.add_argument("--Delta", help = "Detuning rate for 1 photon transitions at center of microwaves [Hz]",
                        type = float, default = 1000e3)
    parser.add_argument("--delta", help = "Detuning rate for 2 photon transition at center of microwaves [Hz]", 
                        type = float, default = 1000e3)
    parser.add_argument("--N_steps", help = "Number of timesteps to take", 
                        type = float, default = 10e3)
    
    args = parser.parse_args()
    
    #Get values from args
    z0 = args.z0
    vz = args.vz
    ring_z1 = args.ring_z1
    ring_z2 = args.ring_z2
    ring_V1 = args.ring_V1
    ring_V2 = args.ring_V2
    Omega1 = 2*np.pi*args.Omega1
    Omega2 = 2*np.pi*args.Omega2
    Delta = 2*np.pi*args.Delta
    delta = 2*np.pi*args.delta
    N_steps = int(args.N_steps)
    
    #Define the total time for which the molecule is simulated
    T = np.abs(2*z0/vz)
    
    #Load options file
    with open(args.run_dir + '/options/' + args.options_fname) as options_file:
        options_dict = json.load(options_file)
    
    #Add run directory and options filename to options_dict
    options_dict["run_dir"] = args.run_dir
    options_dict["options_fname"] = args.options_fname
    result_fname =  args.result_fname  
    save_fields = args.save_fields
    
    #Define the position of the molecule as a function of time
    r0 = np.array((0,0,z0))
    v = np.array((0,0,vz))
    r_t = lambda t: molecule_position(t, r0, v)
    
    #Define electric field as function of position
    E_r = lambda r: (E_field_ring(r, z0 = ring_z1, V = ring_V1) 
                            + E_field_ring(r, z0 = ring_z2, V = ring_V2))
    
    #Next define electric field as function of time
    E_t = lambda t: E_r(r_t(t))
    
    #Define the magnetic field
    B_earth = np.array((0,0,0))
    B_r = lambda r: B_earth
    B_t = lambda t: B_r(r_t(t))
    
    #Generate list of quantum numbers
    QN = make_QN(0,3,1/2,1/2)
    
    #Get hamiltonian as function of E- and B-field
    H_0_EB = get_Hamiltonian(options_dict)
    
    #Make H_0 as function of time
    H_0_t = lambda t: H_0_EB(E_t(t), B_t(t))
    dim = H_0_t(0).shape[0]
    
    #Fetch the initial state from file
    with open("./utility/initial_state0to3.pickle", 'rb') as f:
        initial_state_approx = pickle.load(f)
    
    #Fetch the intermediate state from file
    with open("./utility/intermediate_state0to3.pickle", "rb") as f:
        intermediate_state_approx = pickle.load(f)
        
    #Fetch the final state from file
    with open("./utility/final_state0to3.pickle", 'rb') as f:
        final_state_approx = pickle.load(f)
        
    #Find the eigenstate of the Hamiltonian that most closely corresponds to initial_state at T=0. This will be used as the 
    #actual initial state
    initial_state = find_closest_state(H_0_t(0), initial_state_approx, QN)
    initial_state_vec = initial_state.state_vector(QN)
    
    #Find the eigenstate of the Hamiltonian that most closely corresponds to final_state_approx at t=T. This will be used to 
    #determine what fraction of the molecules end up in the correct final state
    final_state = find_closest_state(H_0_t(T), final_state_approx, QN)
    final_state_vec = final_state.state_vector(QN)
    
    intermediate_state = find_closest_state(H_0_t(0), intermediate_state_approx, QN)
    intermediate_state_vec = intermediate_state.state_vector(QN)
    
    #Find the energies of ini, int and fin so that suitable microwave frequencies can be calculated
    H_z0 = H_0_EB(E_r(np.array((0,0,0))), B_r(np.array((0,0,0))))
    D_z0, V_z0 = np.linalg.eigh(H_z0)
    
    ini_index = find_state_idx_from_state(H_z0, initial_state_approx, QN)
    int_index = find_state_idx_from_state(H_z0, intermediate_state_approx, QN)
    fin_index = find_state_idx_from_state(H_z0, final_state_approx, QN)
    
    #Note: the energies are in 2*pi*[Hz]
    E_ini = D_z0[ini_index]
    E_int = D_z0[int_index]
    E_fin = D_z0[fin_index]
    
    #Define dipole moment of TlF
    D_TlF = 2*np.pi*4.2282 * 0.393430307 * 5.291772e-9/4.135667e-15 # [rad/s/(V/cm)]
    
    #Calculate the approximate power required for each set of microwaves to get a specified peak Rabi rate for the transitions
    ME1 = np.abs(calculate_microwave_ED_matrix_element_mixed_state_uncoupled(initial_state_approx, 
                                                                             intermediate_state_approx,
                                                                             reduced = False))
    
    ME2 = np.abs(calculate_microwave_ED_matrix_element_mixed_state_uncoupled(intermediate_state_approx, 
                                                                             final_state_approx, reduced = False))
    
    P1 = calculate_power_needed(Omega1, ME1)
    P2 = calculate_power_needed(Omega2, ME2)
    
    #Define the microwave electric field as a function of time
    E_mu1_t = lambda t: microwave_field(r_t(t), power = P1)
    E_mu2_t = lambda t: microwave_field(r_t(t), power = P2)
    
    #Define matrix for microwaves coupling J = 0 to 1
    J1_mu1 = 0
    J2_mu1 = 1
    omega_mu1 = E_int - E_ini + Delta
    H1 = make_H_mu(J1_mu1, J2_mu1, omega_mu1, QN)
    H_mu1 = lambda t: H1(0)*D_TlF*E_mu1_t(t)
    
    #Define matrix for microwaves coupling J = 1 to 2
    J1_mu2 = 1
    J2_mu2 = 2
    omega_mu2 = E_fin - E_int + delta - Delta
    H2 = make_H_mu(J1_mu2, J2_mu2, omega_mu2, QN)
    H_mu2 = lambda t: H2(0)*D_TlF*E_mu2_t(t)
    
    
    #Make the matrices used to transform to rotating frame
    U1, D1 = make_transform_matrix(J1_mu1, J2_mu1, omega_mu1, QN)
    U2, D2 = make_transform_matrix(J1_mu2, J2_mu2, omega_mu2+omega_mu1, QN)
    
    U = lambda t: U1(t) @ U2(t)
    D = lambda t: D1 + D2
    
    #Define number of timesteps and make a time-array
    t_array = np.linspace(0,T,N_steps)
    
    #Calculate timestep
    dt = T/N_steps
    
    #Set system to its initial state
    psi = initial_state_vec

    
    #Loop over timesteps to evolve system in time
    for i, t in enumerate(t_array):
        #Calculate the necessary Hamiltonians at this time
        H_0 = H_0_t(t)
        H_mu1_i = H_mu1(t)
        H_mu2_i = H_mu2(t)
        
        #Diagonalize H_0 and transform to that basis
        D_0, V_0, info_0 = zheevd(H_0)
        if info_0 !=0:
            print("zheevd didn't work for H_0")
            D_0, V_0 = np.linalg.eigh(H_0)
        #Make intermediate hamiltonian by transforming H to the basis where H_0 is diagonal
        H_I = V_0.conj().T @ H_0 @ V_0    
        
        #Sort the eigenvalues so they are in ascending order
        index = np.argsort(D_0)
        D_0 = D_0[index]
        V_0 = V_0[:,index]
        
        #Find the microwave coupling matrix:
        H_mu1_i = V_0.conj().T @ H_mu1_i @ V_0 * np.exp(1j*omega_mu1*t)
        H_mu1_i = block_diag(H_mu1_i[0:16,0:16], np.zeros((dim-16,dim-16)))
        H_mu1_i = np.triu(H_mu1_i) + np.triu(H_mu1_i).conj().T #+ np.diag(np.diag(H1))
        
        H_mu2_i = V_0.conj().T @ H_mu2_i @ V_0 * np.exp(1j*omega_mu2*t)
        H_mu2_i = block_diag(np.zeros((4,4)), H_mu2_i[4:36,4:36], np.zeros((dim-36,dim-36)))
        H_mu2_i = np.triu(H_mu2_i) + np.triu(H_mu2_i).conj().T #+ np.diag(np.diag(H1))
        
        
        #Make total hamiltonian
        H_I = H_I + H_mu1_i + H_mu2_i
        
        #Find transformation matrices for rotating basis
        U_t = U(t)
        D_t = D(t)
        
        #Transform H_I to the rotating basis
        H_R = U_t.conj().T @ H_I @ U_t + D_t
        
        #Diagonalize H_R
        D_R, V_R, info_R = zheevd(H_R)
        if info_R !=0:
            print("zheevd didn't work for H_R")
            D_R, V_R = np.linalg.eigh(H_R)
        
        #Propagate state vector in time
        psi = V_0 @ U(t+dt) @ V_R @ np.diag(np.exp(-1j*D_R*dt)) @ V_R.conj().T @ U_t.conj().T @ V_0.conj().T @ psi
        
        
    psi_fin = vector_to_state(psi, QN)
    
    #Calculate overlap between final target state and psi
    overlap = final_state_vec.conj().T@psi
    probability = np.abs(overlap)**2
    
    #Append results into file
    with open(args.run_dir + '/results/' + args.result_fname, 'a') as f:
        result_list = [probability, z0, vz, ring_z1, ring_z2, ring_V1, 
                       ring_V2, Omega1/2*np.pi, Omega2/2*np.pi, Delta/2*np.pi,
                       delta/2*np.pi, N_steps]
        results_list = ["{:.7e}".format(value) for value in result_list]
        results_str = "\t\t".join(results_list)
        print(results_str, file = f)

    