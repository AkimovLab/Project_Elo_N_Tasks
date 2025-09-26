import os, sys
import argparse
import torch
import torch.fft
import libra_py.dynamics.exact_torch.compute as compute
import libra_py.models.Tully_for_exact_pytorch as Tully
import matplotlib.pyplot as plt

plt.rc('axes', titlesize=24)      # fontsize of the axes title
plt.rc('axes', labelsize=24)      # fontsize of the x and y labels
plt.rc('legend', fontsize=24)     # legend fontsize
plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
plt.rc('ytick', labelsize=24)    # fontsize of the tick labels

plt.rc('figure.subplot', left=0.2)
plt.rc('figure.subplot', right=0.95)
plt.rc('figure.subplot', bottom=0.13)
plt.rc('figure.subplot', top=0.88)


parser = argparse.ArgumentParser(description='EXACT...')
parser.add_argument('--istate', type=int)
parser.add_argument('--q0', type=float)
parser.add_argument('--p0', type=float)
args = parser.parse_args()


###############################
istate = args.istate
q0 = args.q0
p0 = args.p0
###############################

prf = F"EXACT-model_Tully2-istate_{istate}-q0_{q0}-p0_{p0}"

os.system(F"rm -r {prf}")
os.system(F"mkdir {prf}")

# Set up parameters
params = {
    "prefix": F"{prf}/data",
    "grid_size": [512*2],       # 1D grid
    "q_min": [-30.0],
    "q_max": [30.0],
    "save_every_n_steps": 1,
    "dt": 10.0,
    "nsteps": 200,
    "mass": [2000.0],         # realistic nuclear mass unit

    "Nstates": 2,
    "representation": "adiabatic",
    "initial_state_index": istate,   # Start in diabatic state 0
    
    "potential_fn": Tully.tully_model2_dual_avoided_crossing,
    "potential_fn_params": {
        "A": 0.1,
        "B": 0.28,
        "C": 0.015,
        "D": 0.06,
        "E0": 0.05
    },
    "psi0_fn" : compute.gaussian_wavepacket,
    "psi0_fn_params": {
        "mass": [2000.0],
        "omega": [0.004],
        "q0": [q0],            # initial position left of crossing
        "p0": [p0]             # initial momentum toward crossing
    },
    "method": "split-operator"
}



# Instantiate solver
solver = compute.exact_tdse_solver_multistate(params)
solver.solve()

import commons
commons.plot_snapshots(prf)
commons.plot_populations(prf)
commons.plot_PES(prf)


"""
f = torch.load(F"{prf}/data.pt")
t = torch.tensor(f["time"])/41.0  # to convert to fs
rho_dia = torch.tensor(f["rho_dia_all"])
rho_adi = torch.tensor(f["rho_adi_all"])

plt.plot(t[:], rho_dia[:,0,0], label="Dia pop 0", color="blue")
plt.plot(t[:], rho_dia[:,1,1], label="Dia pop 1", color="black")

plt.plot(t[:], rho_adi[:,0,0], label="Adi pop 0", color="red")
plt.plot(t[:], rho_adi[:,1,1], label="Adi pop 1", color="green")

plt.plot(t[:], rho_adi[:,0,1].real, label="rho_01, real", color="cyan")
plt.plot(t[:], rho_adi[:,0,1].imag, label="rho_01, imag", color="purple")

plt.legend()
plt.savefig(F"{prf}/populations.png")
"""

