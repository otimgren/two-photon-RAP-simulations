# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:11:45 2020

@author: Oskari
"""

import numpy as np
import scipy
from scipy import constants


def E_field_lens(x, z0 = 0, V = 3e4, R = 1.75*0.0254/2, L = 0.60, l = 20e-3):
    """
    A function that gives the electric field due to to the electrostatic lens
    E_vec = electric field vector in V/cm
    x = position (m)
    z0 = center of lens (m)
    V = voltage on electrodes (V)
    R = radius of lens (m)
    L = length of lens (m)
    l = decay length of lens field (m)
    """
    
    #Calculate electric field vector (assumed to be azimuthal and perpendicular to r_vec)
    E_vec = 2*V/R**2 * np.array((-x[1], x[0], 0))
    
    #Scale the field by a tanh function so it falls off outside the lens
    E_vec = E_vec * (np.tanh((x[2]-z0+L/2)/l) - np.tanh((x[2]-z0-L/2)/l))/2
    
    return E_vec/100

#Define a function that gives the Ez component of the lens field as a function of position
def lens_Ez(x, lens_z0, lens_L):
    """
    Function that evaluates the z-component of the electric field produced by the lens based on position
    
    inputs:
    x = position (in meters) where Ex is evaluated (np.array) 
    
    returns:
    E = np.array that only has z-component (in V/cm)
    """
    
    #Determine radial position and the angle phi in cylindrical coordinates
    r = np.sqrt(x[0]**2+x[1]**2)
        
    #Calculate the value of the electric field
    #Calculate radial scaling
    c2 = 13673437.6
    c4 = 9.4893e+09
    radial_scaling = c2*r**2 + c4*r**4
    
    #Angular function
    angular = 2 * x[0]*x[1]/(r**2)
    
    #In z the field varies as a lorentzian
    sigma = 12.811614314258744/1000
    z1 = lens_z0 - lens_L/2
    z2 = lens_z0 + lens_L/2
    z_function = (np.exp(-(x[2]-z1)**2/(2*sigma**2)) - np.exp(-(x[2]-z2)**2/(2*sigma**2)))
    
    E_z = radial_scaling*angular*z_function
    
    return np.array((0,0,E_z))


def E_field_ring(x,z0 = 0, V = 2e4, R = 2.25*0.0254):
    """
    A function that calculates the axial electric field due to a ring electrode
    E_vec = electric field vector in V/m
    x = position (m)
    Q = charge on ring  (V m)
    R = radius of ring (m)
    """
    #Determine z-position
    z = x[2]
    
    #Calculate electric field
    #The electric field is scaled so that for R = 2.25*0.0254m, get a max field
    #of E = 100000 V/m for a voltage of 20 kV
    scaling_factor = (2.25*0.0254)**2/20e3 * (1e5) *3*np.sqrt(3)/2
    mag_E = scaling_factor*(z-z0)/((z-z0)**2 + R**2)**(3/2)*V
    
    
    #Return the electric field as an array which only has a z-component (approximation)
    return np.array((0,0,mag_E))/100

def polynomial(x,x0,c13,c12,c11,c10,c9,c8,c7,c6,c5,c4,c3,c2,c1,c0):
    
    """
    A polynomial used to evaluate the electric field due to ring electrodes
    """
    return (c13*(x-x0)**13 + c12*(x-x0)**12 +c11*(x-x0)**11 + c10*(x-x0)**10 + c9*(x-x0)**9 + c8*(x-x0)**8
            + c7*(x-x0)**7 +  c6*(x-x0)**6 + c5*(x-x0)**5 + c4*(x-x0)**4 
            + c3*(x-x0)**3 + c2*(x-x0)**2 + c1*(x-x0)**1 + c0)

def E_field_ring_poly(x,z0 = 0, E0 = 200):
    """
    A function that calculates the axial electric field due to a ring electrode
    based on a fit to a polynomial
    E_vec = electric field vector in V/m
    x = position (m)
    Q = charge on ring  (V m)
    R = radius of ring (m)
    """
    #Determine z-position (in m)
    z = x[2]
    
    #List of coefficients for the polynomial fit
    c_list = [-76781382727787.72, 7815959206080.488, 2220176435463.268, 
              -265411984194.187, -17654316750.0341, 3089976024.2313037, 
              -47904198.095862485, -8867780.441144042, 779721.141127246, 
              -51984.94905684154, 3292.120202273202, -36.55964704101279, 
              -15.652867720344146, 1.0]
    
    Ez = E0*polynomial(z, z0, *c_list)
    
    return np.array((0,0,Ez))
    



def microwave_field(x, z0 = 0, fwhm = 0.0254, power = 1):
    """
    Function that calculates the electric field at position x due to a 
    microwave horn with a Gaussian intensity profile defined by its width (fwhm) 
    and total power.
    
    inputs:
    x = position where electric field is to be evaluated (meters)
    z0 = position of the center of the microwave beam
    fwhm = full-width-half-maximum of the microwave intensity profile
    power = output power of microwaves in watts
    
    returns:
    E = magnitude of microwave electric field at x
    """
    
    #Convert FWHM to standard deviation
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    
    #Convert power to amplitude of the Gaussian
    I0 = power/(2*np.pi *sigma**2)
    
    #Get the value of z where the field needs to be evaluated
    z = x[2]
    
    #Calculate intensity at (0,0,z)
    I_z = I0 * np.exp(-1/2*((z-z0)/sigma)**2)
    
    #Calculate electric field from intensity (in V/m)
    c = constants.c
    epsilon_0 = constants.epsilon_0
    E = np.sqrt(2*I_z/(c*epsilon_0))
    
    #Return electric field in V/cm
    return E/100

def calculate_power_needed(Omega, ME, fwhm = 0.0254):
    """
    Function to calculate the microwave power required to get peak Rabi rate Omega
    for a transition with given matrix element when the microwaves have a
    Gaussian spatial profile
    """
    
    #Define dipole moment of TlF
    D_TlF = 2*np.pi * 4.2282 * 0.393430307 *5.291772e-9/4.135667e-15 # [rad/s /(V/cm)]
    
    #Calculate the microwave electric field required (in V/m)
    E =  Omega/(ME*D_TlF) * 100
    
    #Convert E to peak intensity
    c = constants.c
    epsilon_0 = constants.epsilon_0
    I = 1/2 * c * epsilon_0 * E**2
    
    #Convert FWHM to standard deviation
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    
    #Convert power to amplitude of the Gaussian
    P = I * (2*np.pi *sigma**2)
    
    return P
    