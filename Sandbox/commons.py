import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma, gamma

from scipy.stats import gaussian_kde

plt.rc('axes', titlesize=24)      # fontsize of the axes title
plt.rc('axes', labelsize=24)      # fontsize of the x and y labels
plt.rc('legend', fontsize=24)     # legend fontsize
plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
plt.rc('ytick', labelsize=24)    # fontsize of the tick labels

plt.rc('figure.subplot', left=0.2)
plt.rc('figure.subplot', right=0.95)
plt.rc('figure.subplot', bottom=0.13)
plt.rc('figure.subplot', top=0.88)


def plot_populations(prefix, outfile="populations.png"):
    """
    Plot diabatic and adiabatic populations (and coherence) vs time.

    Parameters
    ----------
    prefix : str
        Path to directory containing data.pt
    outfile : str, optional
        Name of output file (saved in the same directory as prefix).
    tscale : float, optional
        Factor to convert time units (default: divide by 41.0 to get fs).
    """
    # Load data
    f = torch.load(os.path.join(prefix, "data.pt"))
    t = f["time"].clone().detach() / 41.0
    rho_dia = f["rho_dia_all"].clone().detach()
    rho_adi = f["rho_adi_all"].clone().detach()

    # Plot
    plt.figure(figsize=(6, 6))

    # Diabatic populations
    plt.plot(t, rho_dia[:, 0, 0], label=r"$\rho_{0}^{dia}$", color="blue")
    plt.plot(t, rho_dia[:, 1, 1], label=r"$\rho_{1}^{dia}$", color="black")

    # Adiabatic populations
    plt.plot(t, rho_adi[:, 0, 0], label=r"$\rho_{0}^{adi}$", color="red")
    plt.plot(t, rho_adi[:, 1, 1], label=r"$\rho_{1}^{adi}$", color="green")

    # Coherence (real and imag parts)
    plt.plot(t, rho_adi[:, 0, 1].real, label=r"$Re[\rho_{01}^{adi}]$", color="cyan")
    plt.plot(t, rho_adi[:, 0, 1].imag, label=r"$Im [\rho_{01}^{adi}]$", color="purple")

    plt.xlabel("Time (fs)")
    plt.ylabel("Population / Coherence")
    plt.legend()
    plt.tight_layout()

    # Save plot
    outpath = os.path.join(prefix, outfile)
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved plot to {outpath}")


def plot_snapshots(prefix, outfile="density.png"):
    """
    Plot initial and final adiabatic densities from trajectory data.

    Parameters
    ----------
    prefix : str
        Path to directory containing data.pt
    outfile : str, optional
        Filename for saving the plot (saved inside `prefix` by default)
    """
    # Load data
    f = torch.load(os.path.join(prefix, "data.pt"))
    print(f.keys())

    # Coordinate grid
    x = np.linspace(f["q_min"][0], f["q_max"][0], f["grid_size"][0])
    nsteps = f["nsteps"]
    print("nsteps =", nsteps)
    print("psi_r_adi_all shape:", f["psi_r_adi_all"].shape)

    # Initial and final densities
    psi = f["psi_r_adi_all"]  # shape: [nsteps, Ngrid, nstates]
    rho0_init = torch.abs(psi[0, :, 0])**2
    rho1_init = torch.abs(psi[0, :, 1])**2
    rho0_finl = torch.abs(psi[-1, :, 0])**2
    rho1_finl = torch.abs(psi[-1, :, 1])**2

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(x, rho0_init, color="red", label=r"$\psi_0(0)$")
    plt.plot(x, rho1_init, color="blue", label=r"$\psi_1(0)$")
    plt.plot(x, rho0_finl, "--", color="red", label=r"$\psi_0(T)$")
    plt.plot(x, rho1_finl, "--", color="blue", label=r"$\psi_1(T)$")

    plt.xlabel("x")
    plt.ylabel(r"$|\psi(x)|^2$")
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(prefix, outfile)
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved plot to {outpath}")


def plot_PES(prefix, outfile="PES.png"):
    """
    Plot diabatic and adiabatic surfaces.

    Parameters
    ----------
    prefix : str
        Path to directory containing data.pt
    outfile : str, optional
        Filename for saving the plot (saved inside `prefix` by default)
    """
    # Load data
    f = torch.load(os.path.join(prefix, "data.pt"))
    print(f.keys())

    # Coordinate grid
    x = np.linspace(f["q_min"][0], f["q_max"][0], f["grid_size"][0])
    nsteps = f["nsteps"]
    print("nsteps =", nsteps)
    print("psi_r_adi_all shape:", f["psi_r_adi_all"].shape)

    # Initial and final densities
    V = f["V"]  # shape: [Ngrid, nstates, nstates]
    E = f["E"]  # shape: [Ngrid, nstates]
    print(V.shape)
    print(E.shape)
    
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(x, V[:, 0, 0], color="red", label=r"$V_{00}$")
    plt.plot(x, V[:, 1, 1], color="blue", label=r"$V_{11}$")
    plt.plot(x, V[:, 0, 1], color="black", label=r"$V_{01}$")
    plt.plot(x, E[:, 0], ls="--", color="red", label=r"$E_0$")
    plt.plot(x, E[:, 1], ls="--", color="blue", label=r"$E_1$")

    plt.xlabel("x, a.u.")
    plt.ylabel(r"Energy, a.u.")
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(prefix, outfile)
    #plt.show()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved plot to {outpath}")
    

def kozachenko_leonenko_entropy(data, k=3):
    """
    Estimate differential entropy using the Kozachenko-Leonenko estimator.
    
    Note: it can be negative!
    Example: a narrow Gaussian has more negative differential entropy than a wider one.
    
    Parameters:
        data: ndarray of shape (n_samples, n_dimensions)
        k: int, number of nearest neighbors (default=1)
    
    Returns:
        Estimated entropy (float)
    """
    n, d = data.shape
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data)
    distances, _ = nbrs.kneighbors(data)
    
    # Exclude the zero distance to itself at index 0
    epsilons = np.maximum(distances[:, k], 1e-15)
    #epsilons = distances[:, k]
    
    # Constant: volume of unit ball in d dimensions
    cd = (np.pi ** (d / 2)) / gamma(1 + d / 2)
    
    entropy = (
        digamma(n) - digamma(k) +
        np.log(cd) +
        d * np.mean(np.log(epsilons))
    )
    return entropy


def compute_informations(z, bins=100, sigma=0.01):
    """
    Computes the histogram of the data and computes the KDE and Shannon entropy
    but does not plot anything
    """
    
    # Compute the histogram
    counts, bin_edges = np.histogram(z, bins, density=True)
    
    # Normalize
    counts = counts/sum(counts)

    # Compute bin centers for plotting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    dx = bin_edges[1] - bin_edges[0]
    #print(F"dx = {dx}")

    H = sum( -counts[counts>0.0] * np.log(counts[counts>0.0]) ) * dx
    print(F"H (from hist) = {H/ np.log(2)} bits")

    # Assume z is your 1D data
    kde = gaussian_kde(z, sigma)

    # Evaluate on a grid
    x = bin_centers
    p = kde(x)

    # Avoid log(0) by clipping
    p = np.clip(p, 1e-12, None)
    
    # Normalize
    p = p / sum(p)
    H_kde = -np.sum(p * np.log(p)) * dx
    print(f"H (from KDE) = {H_kde / np.log(2)} bits")
    
    npts = len(z)
    z = np.array(z).reshape(npts, 1)
    #print(z)
    H_KL = kozachenko_leonenko_entropy(z)
    print(f"Kozachenko-Leonenko entropy ≈ {H_KL / np.log(2)} bits")
    
    # Multi-line text with a single bbox
    value1 = H/ np.log(2)
    value2 = H_kde/ np.log(2)
    value3 = H_KL/ np.log(2)
    text_str = f"H (hist) = {value1:.3f} bits\nH (KDE) = {value2:.3f} bits\nH (KL) = {value3:.3f} bits"
        
    return value1, value2, value3, x, counts, p


def show_hist(_plt, z, bins=100, sigma=0.01, xbox=0.75, ybox=0.1):
    """
    Computes the histogram of the data and computes the KDE
    and Shannon entropy
    """
    
    # Compute the histogram
    counts, bin_edges = np.histogram(z, bins, density=True)
    
    # Normalize
    counts = counts/sum(counts)

    # Compute bin centers for plotting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    dx = bin_edges[1] - bin_edges[0]
    #print(F"dx = {dx}")

    H = sum( -counts[counts>0.0] * np.log(counts[counts>0.0]) ) * dx
    print(F"H (from hist) = {H/ np.log(2)} bits")

    # Assume z is your 1D data
    kde = gaussian_kde(z, sigma)

    # Evaluate on a grid
    x = bin_centers
    p = kde(x)

    # Avoid log(0) by clipping
    p = np.clip(p, 1e-12, None)
    
    # Normalize
    p = p / sum(p)
    H_kde = -np.sum(p * np.log(p)) * dx
    print(f"H (from KDE) = {H_kde / np.log(2)} bits")
    
    npts = len(z)
    z = np.array(z).reshape(npts, 1)
    #print(z)
    H_KL = kozachenko_leonenko_entropy(z)
    print(f"Kozachenko-Leonenko entropy ≈ {H_KL / np.log(2)} bits")

    _plt.plot(bin_centers, counts, label="PDF", color="blue", lw=5)
    _plt.plot(x, p, label=F"KDE", color="black", lw=5)
    
    # Multi-line text with a single bbox
    value1 = H/ np.log(2)
    value2 = H_kde/ np.log(2)
    value3 = H_KL/ np.log(2)
    text_str = f"H (hist) = {value1:.3f} bits\nH (KDE) = {value2:.3f} bits\nH (KL) = {value3:.3f} bits"
    
    _plt.text(xbox, ybox, text_str, 
        color='black', fontsize=40, ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'))
    _plt.legend()
    
    return value1, value2, value3
