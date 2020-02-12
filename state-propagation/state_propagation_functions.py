# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:56:10 2020

@author: Oskari
"""

import sys
sys.path.append('../molecular-state-classes-and-functions/')
sys.path.append('../utility/')
from functions import make_QN, make_hamiltonian, make_transform_matrix, find_closest_state
from functions import find_state_idx_from_state , make_H_mu, reorder_evecs, vector_to_state
from matrix_element_functions import calculate_microwave_ED_matrix_element_mixed_state_uncoupled

from EM_fields import E_field_ring, calculate_power_needed, microwave_field
import numpy as np
import pickle
from scipy.linalg.lapack import zheevd
from scipy.linalg import block_diag
from tqdm.notebook import tqdm

def simulate_RAP(r0 = np.array((0,0,-100e-3)), v = np.array((0,0,200)),
                 ring_z1 = -71.4375e-3, ring_z2 = 85.725e-3,
                 ring_V1 = 1e3, ring_V2 = 0.9e2,
                 B_earth = np.array((0,0,0)),
                 Omega1 = 2*np.pi*200e3, Omega2 = 2*np.pi*200e3,
                 Delta = 2*np.pi*2e6, delta = 0,
                 N_steps = 100e3):
    """
    Function that runs the rapid adiabatic passage simulation for the given 
    parameters
    
    inputs:
    r0 = initial position of molecule [m]
    v = velocity of molecule (assumed to be constant) [m/s]
    ring_z1 = position of first ring electrode [m]
    ring_z2 = position of second ring electrode [m]
    ring_V1 = voltage on ring 1 [V]
    ring_V2 = voltage on ring 2 [V]
    B_earth = magnetic field of earth [G]
    
    """
    
    #Define the position of the molecule as a fucntion of time
    r_t = lambda t: molecule_position(t, r0, v)
    
    #Define the total time for which the molecule is simulated
    z0 = r0[2]
    vz = v[2]
    T = np.abs(2*z0/vz)
    
    #Define electric field as function of position
    E_r = lambda r: (E_field_ring(r, z0 = ring_z1, V = ring_V1) 
                            + E_field_ring(r, z0 = ring_z2, V = ring_V2))
    
    #Next define electric field as function of time
    E_t = lambda t: E_r(r_t(t))
    
    #Define the magnetic field
    B_r = lambda r: B_earth
    B_t = lambda t: B_r(r_t(t))
    
    #Make list of quantum numbers that defines the basis for the matrices
    QN = make_QN(0,3,1/2,1/2)
    
    #Get H_0 from file (H_0 should be in rad/s)
    H_0_EB = make_hamiltonian("../utility/TlF_X_state_hamiltonian0to3.py")
    
    #Make H_0 as function of time
    H_0_t = lambda t: H_0_EB(E_t(t), B_t(t))
    dim = H_0_t(0).shape[0]
    
    #Fetch the initial state from file
    with open("../utility/initial_state0to3.pickle", 'rb') as f:
        initial_state_approx = pickle.load(f)
    
    #Fetch the intermediate state from file
    with open("../utility/intermediate_state0to3.pickle", "rb") as f:
        intermediate_state_approx = pickle.load(f)
        
    #Fetch the final state from file
    with open("../utility/final_state0to3.pickle", 'rb') as f:
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
                                                                             intermediate_state_approx, reduced = False))
    
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
    for i, t in enumerate(tqdm(t_array)):
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
        
    return probability
    
    
def molecule_position(t, r0, v):
    """
    Function that returns position of molecule at a given time for given initial position and velocity.
    inputs:
    t = time in seconds
    r0 = position of molecule at t = 0 in meters
    v = velocity of molecule in meters per second
    
    returns:
    r = position of molecule in metres
    """
    r =  r0 + v*t
    
    return r

def get_Hamiltonian(options_dict): 
    """
    Function that gets the hamiltonian from a file specified in the options
    dictionary and returns it as a function of electric and magnetic fields
    """
    H_fname = options_dict["H_fname"]
    run_dir = options_dict["run_dir"]
    H = make_hamiltonian(run_dir+H_fname)
    
    return H