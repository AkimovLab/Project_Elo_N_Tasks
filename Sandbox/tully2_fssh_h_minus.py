#!/usr/bin/env python
# coding: utf-8

################# Generic setups #########################
import sys, cmath, math, os, h5py, time, warnings
import matplotlib.pyplot as plt   # plots
import numpy as np

from liblibra_core import *
import util.libutil as comn
from libra_py import units, data_conv
import libra_py.models.Tully as Tully
import libra_py.dynamics.tsh.recipes.fssh_h_minus as fssh_h_minus
from libra_py import dynamics_plotting
import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.dynamics.tsh.plot as tsh_dynamics_plot
import libra_py.data_savers as data_savers

import argparse

warnings.filterwarnings('ignore')

colors = {}
colors.update({"11": "#8b1a0e"})  # red       
colors.update({"12": "#FF4500"})  # orangered 
colors.update({"13": "#B22222"})  # firebrick 
colors.update({"14": "#DC143C"})  # crimson   
colors.update({"21": "#5e9c36"})  # green
colors.update({"22": "#006400"})  # darkgreen  
colors.update({"23": "#228B22"})  # forestgreen
colors.update({"24": "#808000"})  # olive      
colors.update({"31": "#8A2BE2"})  # blueviolet
colors.update({"32": "#00008B"})  # darkblue  
colors.update({"41": "#2F4F4F"})  # darkslategray

clrs_index = ["11", "21", "31", "41", "12", "22", "32", "13","23", "14", "24"]


############## Specific content #################################
## 1. ======== Select the Hamiltonian (model) ==================
model = "Tully2"

def compute_model(q, params, full_id):
    return Tully.Tully2(q, params, full_id)

model_params = { "model":1, "model0":1, "nstates":2, "A": 0.10, "B": 0.28, "C": 0.015, "D": 0.06, "E0": 0.05}


NSTATES = model_params["nstates"]
print(F"NSTATES = {NSTATES}")


## 2.=========== Select the initial conditions: nuclear and electronic ==============

parser = argparse.ArgumentParser(description='NAMD...')
parser.add_argument('--istate', type=int)
parser.add_argument('--q0', type=float)
parser.add_argument('--p0', type=float)
args = parser.parse_args()


###############################
istate = args.istate
q0 = args.q0
p0 = args.p0
###############################

#============== How nuclear DOFs are initialized =================
# In exact calculations:
# envelope = torch.exp(-torch.sum( 0.25*(q - q0)**2 / sigma**2 , dim=0, keepdim=False) )  # because it is wavefunction
# where   sigma = 1.0 / torch.sqrt( 2.0 * mass * omega )
# so,  exp(- 0.25 * (q-q0)**2 * 2.0* mass * omega  ) = exp( - 0.5 * (mass * omega) * (q-q0)**2 )
# In TSH sampling:
# "force_constant" - ( list of double ): force constants involved in the Harmonic
#                oscillator model: U = (1/2) * k * x^2, and omega = sqrt( k / m )
#                These parameters define the Harmonic oscillator ground state wavefunctions from which
#                the coordinates and momenta are sampled. [ units: a.u. = Ha/Bohr^2, default: [0.001] ]
#                The length should be consistent with the length of Q, P and M arguments
#                The ground state of the corresponding HO is given by:
#                 psi_0 = (alp/pi)^(1/4)  exp(-1/2 * alp * x^2), with alp = m * omega / hbar 
# So the overall correspondence is:
# k = m * omega**2               

icond_nucl = 3  # Both coords and momenta are sampled
m = 2000.0
omega = 0.004
k = m * omega**2
wq, wp = 1.0, 1.0
nucl_params = { "ndof":1, "q":[q0], "p":[p0], "mass":[m], "force_constant":[k], 
                "q_width":[wq],  "p_width":[wp], "init_type":icond_nucl }

#============= How electronic DOFs are initialized ==================
# Select a specific initial condition
istates = list(np.zeros(NSTATES))
istates[istate] = 1.0
elec_params = {"verbosity":-1, "init_dm_type":0, "ndia":NSTATES, "nadi":NSTATES, "rep":1, 
               "init_type":1, "istate":istate, "istates":istates }


## 3. ============ Choosing the Nonadiabatic Dynamics Methodology ===============
method = "FSSH_h_minus"
prf = F"NAMD-model_{model}-istate_{istate}-q0_{q0}-p0_{p0}-method_{method}"

dyn_params = { "nsteps":200, "ntraj":2000, "nstates":NSTATES,
                "dt":10.0, "num_electronic_substeps":1, "isNBRA":0, "is_nbra":0,
                "progress_frequency":0.1, "which_adi_states":range(NSTATES), "which_dia_states":range(NSTATES),
                "prefix":prf, "prefix2":prf,
                "ensemble":0, "quantum_dofs":[0], "thermostat_dofs":[], "constrained_dofs":[],
                "mem_output_level":3,
                "properties_to_save":[ "timestep", "time", 
                                       "states", "states_dia", 
                                       "se_pop_adi", "se_pop_dia",
                                       "sh_pop_adi", "sh_pop_dia", 
                                       "sh_pop_adi_TR", "sh_pop_dia_TR", 
                                       "mash_pop_adi", "mash_pop_dia",
                                       "Epot_ave", "Ekin_ave", "Etot_ave",
                                       "coherence_adi", "coherence_dia"
                                     ],
                "icond":0
              }
fssh_h_minus.load(dyn_params)

## 4. =================== Running the calculations ================
print(F"Computing {prf}")    
rnd = Random()
res = tsh_dynamics.generic_recipe(dyn_params, compute_model, model_params, elec_params, nucl_params, rnd)


## 5. =============== Plotting the results ========================
pref = prf

plot_params = { "prefix":pref, "filename":"mem_data.hdf", "output_level":3,
                "which_trajectories":[0], "which_dofs":[0], "which_adi_states":list(range(NSTATES)), 
                "which_dia_states":list(range(NSTATES)), 
                "frameon":True, "linewidth":3, "dpi":300,
                "axes_label_fontsize":(8,8), "legend_fontsize":8, "axes_fontsize":(8,8), "title_fontsize":8,
                "which_energies":["potential", "kinetic", "total"],
                "save_figures":1, "do_show":0,
                "what_to_plot":["energies", "energy_fluctuations", 
                                "se_pop_adi", "se_pop_dia", 
                                "sh_pop_adi", "sh_pop_dia", 
                                "sh_pop_adi_TR", "sh_pop_dia_TR", 
                                "mash_pop_adi", "mash_pop_dia", 
                                "coherence_adi", "coherence_dia" ] 
              }
tsh_dynamics_plot.plot_dynamics(plot_params)

