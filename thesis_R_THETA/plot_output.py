#!/usr/bin/env python3
"""
Plot diagnostics for the thesis_R_THETA Vlasov-Poisson simulation.

Reads HDF5 output files produced by the PDI decl_hdf5 plugin and generates:
  - Electrostatic potential φ(r,θ)  in polar view
  - Charge density ρ(r,θ) integrated over velocity
  - Phase-space plots: f(r,vr), f(θ,vθ)
  - Velocity-space distribution f(vr,vθ)
  - Radial and poloidal profiles of φ and ρ
  - Time evolution of total density, kinetic energy, max|φ|

Usage
-----
  # Plot a single iteration (6-panel snapshot):
  python plot_output.py output/GYSELALIBXX_00005.h5

  # Plot all iterations + time evolution:
  python plot_output.py output/ --all

  # Plot init state mesh info:
  python plot_output.py output/ --initstate

  # Choose species (0=electron, 1=ion):
  python plot_output.py output/GYSELALIBXX_00005.h5 --species 1
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ---------------------------------------------------------------------------
# HDF5 reading helpers  (uses h5dump CLI to avoid h5py dependency)
# ---------------------------------------------------------------------------

def read_dataset(path: Path, name: str) -> np.ndarray:
    """Read a single HDF5 dataset via ``h5dump``."""
    text = subprocess.check_output(
        ["h5dump", "-y", "-d", name, str(path)], text=True
    )
    shape_match = re.search(
        r"DATASPACE\s+SIMPLE\s+\{\s+\(\s*([^)]*)\s*\)", text
    )
    if "DATA {" not in text:
        raise RuntimeError(f"Could not read dataset {name!r} from {path}")

    shape = () if shape_match is None else tuple(
        int(v.strip()) for v in shape_match.group(1).split(",")
    )
    data_block = text.split("DATA {", 1)[1].rsplit("}", 2)[0]
    values = np.fromstring(data_block, sep=",")
    return values.reshape(shape) if shape else values


def try_read_dataset(path: Path, name: str):
    """Return the dataset or None if it doesn't exist."""
    try:
        return read_dataset(path, name)
    except (subprocess.CalledProcessError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_initstate(output_dir: Path) -> dict:
    """Load meshes and metadata from the init-state file."""
    init_file = output_dir / "GYSELALIBXX_initstate.h5"
    if not init_file.exists():
        print(f"Warning: {init_file} not found – mesh info unavailable.")
        return {}

    info = {}
    for key in ["MeshR", "MeshTheta", "MeshVr", "MeshVtheta",
                 "Nr_spline_cells", "Ntheta_spline_cells",
                 "Nvr_spline_cells", "Nvtheta_spline_cells",
                 "Nkinspecies", "fdistribu_charges", "fdistribu_masses",
                 "deltat", "nbiter", "time_step_diag"]:
        val = try_read_dataset(init_file, key)
        if val is not None:
            info[key] = val
    return info


def load_iteration(filepath: Path) -> dict:
    """Load data from an iteration HDF5 file."""
    data = {}
    for key in ["time_saved", "fdistribu", "electrostatic_potential"]:
        val = try_read_dataset(filepath, key)
        if val is not None:
            data[key] = val
    return data


def find_iteration_files(output_dir: Path) -> list:
    """Find and sort all iteration HDF5 files (excluding initstate)."""
    files = sorted(output_dir.glob("GYSELALIBXX_*.h5"))
    return [f for f in files if "initstate" not in f.name]


SPECIES_NAMES = {0: "electron", 1: "ion"}


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def compute_density(fdistribu, mesh_vr, mesh_vtheta, species_idx=0):
    """
    Integrate f over velocity space to get density ρ(r,θ).

    fdistribu shape: (Nsp, Nr, Ntheta, Nvr, Nvtheta)
    Returns: density(Nr, Ntheta)
    """
    f_sp = fdistribu[species_idx]  # (Nr, Ntheta, Nvr, Nvtheta)
    return np.trapz(np.trapz(f_sp, mesh_vtheta, axis=3), mesh_vr, axis=2)


def compute_energy(fdistribu, mesh_vr, mesh_vtheta, species_idx=0):
    """
    Compute kinetic energy density  ∫ (vr² + vθ²)/2 · f dvr dvθ.
    """
    f_sp = fdistribu[species_idx]
    vr2 = mesh_vr[:, None] ** 2
    vt2 = mesh_vtheta[None, :] ** 2
    v2 = vr2 + vt2
    integrand = 0.5 * f_sp * v2[None, None, :, :]
    return np.trapz(np.trapz(integrand, mesh_vtheta, axis=3), mesh_vr, axis=2)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def polar_plot(ax, mesh_r, mesh_theta, field, title, cmap="RdBu_r",
               symmetric=False):
    """Plot a 2D field on (r,θ) grid in Cartesian-projected polar coords."""
    R, T = np.meshgrid(mesh_r, mesh_theta, indexing="ij")
    X = R * np.cos(T)
    Y = R * np.sin(T)

    if symmetric:
        vmax = np.max(np.abs(field))
        if vmax == 0:
            vmax = 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.pcolormesh(X, Y, field, cmap=cmap, norm=norm, shading="auto")
    else:
        im = ax.pcolormesh(X, Y, field, cmap=cmap, shading="auto")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, shrink=0.8)


def phase_space_plot(ax, mesh_x, mesh_v, field, xlabel, vlabel, title,
                     cmap="inferno"):
    """Plot a 2D phase-space slice f(x, v)."""
    XX, VV = np.meshgrid(mesh_x, mesh_v, indexing="ij")
    im = ax.pcolormesh(XX, VV, field, cmap=cmap, shading="auto")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(vlabel)
    plt.colorbar(im, ax=ax, shrink=0.8)


# ---------------------------------------------------------------------------
# Main snapshot plot  (6 panels)
# ---------------------------------------------------------------------------

def plot_snapshot(filepath: Path, info: dict, output_path: Path = None,
                  species_idx: int = 0):
    """
    Generate a 6-panel diagnostic plot:
      Row 1: φ(r,θ)         | ρ(r,θ)
      Row 2: f(r, vr)       | f(θ, vθ)
      Row 3: f(vr, vθ)      | radial profiles φ(r) and ρ(r)
    """
    data = load_iteration(filepath)
    if not data:
        print(f"No data found in {filepath}")
        return

    mesh_r = info.get("MeshR", np.linspace(0, 1, 32))
    mesh_theta = info.get("MeshTheta", np.linspace(0, 2 * np.pi, 64))
    mesh_vr = info.get("MeshVr", np.linspace(-6, 6, 32))
    mesh_vtheta = info.get("MeshVtheta", np.linspace(-6, 6, 32))

    time = data.get("time_saved", np.array(0.0))
    if isinstance(time, np.ndarray):
        time = time.item() if time.size == 1 else float(time.flat[0])

    sp_name = SPECIES_NAMES.get(species_idx, f"sp{species_idx}")
    phi = data.get("electrostatic_potential")
    fdistribu = data.get("fdistribu")

    fig, axes = plt.subplots(3, 2, figsize=(14, 18), constrained_layout=True)
    fig.suptitle(
        f"Thesis R-Θ Vlasov–Poisson  —  t = {time:.4f}  —  {sp_name}",
        fontsize=14, fontweight="bold"
    )

    # ── Row 1, Col 1: Electrostatic potential φ(r,θ) ──────────────────
    if phi is not None:
        polar_plot(axes[0, 0], mesh_r, mesh_theta, phi,
                   r"Electrostatic potential  $\varphi(r,\theta)$",
                   cmap="RdBu_r", symmetric=True)
    else:
        axes[0, 0].text(0.5, 0.5, "No potential data", transform=axes[0, 0].transAxes,
                         ha="center", va="center")

    # ── Row 1, Col 2: Density ρ(r,θ)  ─────────────────────────────────
    if fdistribu is not None:
        density = compute_density(fdistribu, mesh_vr, mesh_vtheta, species_idx)
        polar_plot(axes[0, 1], mesh_r, mesh_theta, density,
                   r"Density  $\rho(r,\theta) = \int f \, dv_r \, dv_\theta$",
                   cmap="viridis")
    else:
        axes[0, 1].text(0.5, 0.5, "No fdistribu data", transform=axes[0, 1].transAxes,
                         ha="center", va="center")
        density = None

    # ── Row 2, Col 1: Phase space f(r, vr) at θ=0, vθ=mid ────────────
    if fdistribu is not None:
        f_sp = fdistribu[species_idx]  # (Nr, Ntheta, Nvr, Nvtheta)
        itheta = 0
        ivtheta_mid = f_sp.shape[3] // 2
        f_r_vr = f_sp[:, itheta, :, ivtheta_mid]  # (Nr, Nvr)
        phase_space_plot(
            axes[1, 0], mesh_r, mesh_vr, f_r_vr,
            xlabel="r", vlabel=r"$v_r$",
            title=(f"Phase space  $f(r, v_r)$  at "
                   f"θ={mesh_theta[itheta]:.2f}, "
                   f"$v_\\theta$={mesh_vtheta[ivtheta_mid]:.2f}"),
            cmap="inferno"
        )
    else:
        axes[1, 0].set_visible(False)

    # ── Row 2, Col 2: Phase space f(θ, vθ) at r=mid, vr=mid ──────────
    if fdistribu is not None:
        ir_mid = f_sp.shape[0] // 2
        ivr_mid = f_sp.shape[2] // 2
        f_theta_vtheta = f_sp[ir_mid, :, ivr_mid, :]  # (Ntheta, Nvtheta)
        phase_space_plot(
            axes[1, 1], mesh_theta, mesh_vtheta, f_theta_vtheta,
            xlabel=r"$\theta$", vlabel=r"$v_\theta$",
            title=(f"Phase space  $f(\\theta, v_\\theta)$  at "
                   f"r={mesh_r[ir_mid]:.2f}, "
                   f"$v_r$={mesh_vr[ivr_mid]:.2f}"),
            cmap="inferno"
        )
    else:
        axes[1, 1].set_visible(False)

    # ── Row 3, Col 1: Velocity space f(vr, vθ) at (r_mid, θ=0) ───────
    if fdistribu is not None:
        ir_mid = f_sp.shape[0] // 2
        itheta = 0
        f_vr_vtheta = f_sp[ir_mid, itheta, :, :]  # (Nvr, Nvtheta)
        VR, VT = np.meshgrid(mesh_vr, mesh_vtheta, indexing="ij")
        im = axes[2, 0].pcolormesh(VR, VT, f_vr_vtheta, cmap="inferno",
                                    shading="auto")
        axes[2, 0].set_title(
            f"Velocity space  $f(v_r, v_\\theta)$  at "
            f"r={mesh_r[ir_mid]:.2f}, θ={mesh_theta[itheta]:.2f}",
            fontsize=11, fontweight="bold"
        )
        axes[2, 0].set_xlabel(r"$v_r$")
        axes[2, 0].set_ylabel(r"$v_\theta$")
        axes[2, 0].set_aspect("equal")
        plt.colorbar(im, ax=axes[2, 0], shrink=0.8)
    else:
        axes[2, 0].set_visible(False)

    # ── Row 3, Col 2: Radial profiles at θ=0 ──────────────────────────
    ax_profile = axes[2, 1]
    itheta = 0
    plotted = False
    if phi is not None:
        ax_profile.plot(mesh_r, phi[:, itheta], "-", color="#2563EB",
                        linewidth=2, label=r"$\varphi(r)$")
        plotted = True
    if density is not None:
        ax2 = ax_profile.twinx()
        ax2.plot(mesh_r, density[:, itheta], "-", color="#DC2626",
                 linewidth=2, label=r"$\rho(r)$")
        ax2.set_ylabel(r"$\rho$", color="#DC2626")
        ax2.tick_params(axis="y", labelcolor="#DC2626")
        ax2.legend(loc="upper left")
        plotted = True
    if plotted:
        ax_profile.set_title(
            f"Radial profiles at θ = {mesh_theta[itheta]:.2f}",
            fontsize=11, fontweight="bold"
        )
        ax_profile.set_xlabel("r")
        ax_profile.set_ylabel(r"$\varphi$", color="#2563EB")
        ax_profile.tick_params(axis="y", labelcolor="#2563EB")
        ax_profile.legend(loc="upper right")
        ax_profile.grid(True, alpha=0.3)
    else:
        ax_profile.set_visible(False)

    out = output_path or filepath.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dedicated fdistribu multi-view plot
# ---------------------------------------------------------------------------

def plot_fdistribu_detail(filepath: Path, info: dict, output_path: Path = None,
                          species_idx: int = 0):
    """
    6-panel detailed view of the distribution function:
      Row 1:  f(r, vr) at θ=0      |  f(r, vr) at θ=π
      Row 2:  f(θ, vθ) at r=0.25   |  f(θ, vθ) at r=0.75
      Row 3:  f(vr, vθ) at r=mid,θ=0  |  f(vr, vθ) at r=mid,θ=π
    """
    data = load_iteration(filepath)
    if not data or "fdistribu" not in data:
        print(f"No fdistribu in {filepath}")
        return

    mesh_r = info.get("MeshR", np.linspace(0, 1, 32))
    mesh_theta = info.get("MeshTheta", np.linspace(0, 2 * np.pi, 64))
    mesh_vr = info.get("MeshVr", np.linspace(-6, 6, 32))
    mesh_vtheta = info.get("MeshVtheta", np.linspace(-6, 6, 32))

    time = data.get("time_saved", np.array(0.0))
    if isinstance(time, np.ndarray):
        time = time.item() if time.size == 1 else float(time.flat[0])

    f_sp = data["fdistribu"][species_idx]  # (Nr, Ntheta, Nvr, Nvtheta)
    sp_name = SPECIES_NAMES.get(species_idx, f"sp{species_idx}")

    # Pick indices for different spatial locations
    Nr, Ntheta, Nvr, Nvtheta = f_sp.shape
    itheta_0 = 0
    itheta_pi = Ntheta // 2
    ir_quarter = Nr // 4
    ir_3quarter = 3 * Nr // 4
    ir_mid = Nr // 2
    ivr_mid = Nvr // 2
    ivtheta_mid = Nvtheta // 2

    fig, axes = plt.subplots(3, 2, figsize=(14, 18), constrained_layout=True)
    fig.suptitle(
        f"Distribution function detail  —  {sp_name}  —  t = {time:.4f}",
        fontsize=14, fontweight="bold"
    )

    # Row 1: f(r, vr) at two poloidal angles
    for col, itheta in enumerate([itheta_0, itheta_pi]):
        f_slice = f_sp[:, itheta, :, ivtheta_mid]
        phase_space_plot(
            axes[0, col], mesh_r, mesh_vr, f_slice,
            xlabel="r", vlabel=r"$v_r$",
            title=(f"$f(r, v_r)$ at θ={mesh_theta[itheta]:.2f}, "
                   f"$v_\\theta$={mesh_vtheta[ivtheta_mid]:.2f}")
        )

    # Row 2: f(θ, vθ) at two radii
    for col, ir in enumerate([ir_quarter, ir_3quarter]):
        f_slice = f_sp[ir, :, ivr_mid, :]
        phase_space_plot(
            axes[1, col], mesh_theta, mesh_vtheta, f_slice,
            xlabel=r"$\theta$", vlabel=r"$v_\theta$",
            title=(f"$f(\\theta, v_\\theta)$ at r={mesh_r[ir]:.2f}, "
                   f"$v_r$={mesh_vr[ivr_mid]:.2f}")
        )

    # Row 3: f(vr, vθ) at two locations
    for col, itheta in enumerate([itheta_0, itheta_pi]):
        f_slice = f_sp[ir_mid, itheta, :, :]
        VR, VT = np.meshgrid(mesh_vr, mesh_vtheta, indexing="ij")
        im = axes[2, col].pcolormesh(VR, VT, f_slice, cmap="inferno",
                                      shading="auto")
        axes[2, col].set_title(
            f"$f(v_r, v_\\theta)$ at r={mesh_r[ir_mid]:.2f}, "
            f"θ={mesh_theta[itheta]:.2f}",
            fontsize=11, fontweight="bold"
        )
        axes[2, col].set_xlabel(r"$v_r$")
        axes[2, col].set_ylabel(r"$v_\theta$")
        axes[2, col].set_aspect("equal")
        plt.colorbar(im, ax=axes[2, col], shrink=0.8)

    out = output_path or filepath.with_name(
        filepath.stem + "_fdistribu.png"
    )
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dedicated potential plot
# ---------------------------------------------------------------------------

def plot_potential_detail(filepath: Path, info: dict, output_path: Path = None):
    """
    4-panel detailed view of the electrostatic potential:
      Row 1:  φ(r,θ) polar view    |  φ(r,θ) heatmap (r vs θ)
      Row 2:  φ(r) at several θ    |  φ(θ) at several r
    """
    data = load_iteration(filepath)
    if not data or "electrostatic_potential" not in data:
        print(f"No electrostatic_potential in {filepath}")
        return

    mesh_r = info.get("MeshR", np.linspace(0, 1, 32))
    mesh_theta = info.get("MeshTheta", np.linspace(0, 2 * np.pi, 64))

    time = data.get("time_saved", np.array(0.0))
    if isinstance(time, np.ndarray):
        time = time.item() if time.size == 1 else float(time.flat[0])

    phi = data["electrostatic_potential"]  # (Nr, Ntheta)
    Nr, Ntheta = phi.shape

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(
        f"Electrostatic potential  —  t = {time:.4f}",
        fontsize=14, fontweight="bold"
    )

    # Panel 1: polar view
    polar_plot(axes[0, 0], mesh_r, mesh_theta, phi,
               r"$\varphi(r,\theta)$ — polar view",
               cmap="RdBu_r", symmetric=True)

    # Panel 2: heatmap in (r, θ) space
    R_grid, T_grid = np.meshgrid(mesh_r, mesh_theta, indexing="ij")
    im = axes[0, 1].pcolormesh(T_grid, R_grid, phi, cmap="RdBu_r",
                                shading="auto")
    axes[0, 1].set_title(r"$\varphi(r,\theta)$ — grid view",
                          fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel(r"$\theta$")
    axes[0, 1].set_ylabel("r")
    plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

    # Panel 3: radial profiles at several θ
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    theta_indices = np.linspace(0, Ntheta - 1, 5, dtype=int)
    for i, itheta in enumerate(theta_indices):
        axes[1, 0].plot(mesh_r, phi[:, itheta], "-", color=colors[i],
                        linewidth=1.5,
                        label=f"θ = {mesh_theta[itheta]:.2f}")
    axes[1, 0].set_title("Radial profiles φ(r)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("r")
    axes[1, 0].set_ylabel(r"$\varphi$")
    axes[1, 0].legend(fontsize=8, loc="best")
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: poloidal profiles at several r
    colors = plt.cm.plasma(np.linspace(0, 1, 5))
    r_indices = np.linspace(1, Nr - 1, 5, dtype=int)  # skip r=0
    for i, ir in enumerate(r_indices):
        axes[1, 1].plot(mesh_theta, phi[ir, :], "-", color=colors[i],
                        linewidth=1.5,
                        label=f"r = {mesh_r[ir]:.3f}")
    axes[1, 1].set_title("Poloidal profiles φ(θ)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel(r"$\theta$")
    axes[1, 1].set_ylabel(r"$\varphi$")
    axes[1, 1].legend(fontsize=8, loc="best")
    axes[1, 1].grid(True, alpha=0.3)

    out = output_path or filepath.with_name(filepath.stem + "_potential.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Time evolution
# ---------------------------------------------------------------------------

def plot_time_evolution(output_dir: Path, info: dict, output_path: Path = None,
                        species_idx: int = 0):
    """Plot time traces: total density, total kinetic energy, max |φ|."""
    files = find_iteration_files(output_dir)
    if not files:
        print("No iteration files found.")
        return

    mesh_vr = info.get("MeshVr")
    mesh_vtheta = info.get("MeshVtheta")

    times, total_density, total_energy, max_phi = [], [], [], []

    for f in files:
        data = load_iteration(f)
        if not data:
            continue
        t = data.get("time_saved", np.array(0.0))
        if isinstance(t, np.ndarray):
            t = t.item() if t.size == 1 else float(t.flat[0])
        times.append(t)

        phi = data.get("electrostatic_potential")
        if phi is not None:
            max_phi.append(np.max(np.abs(phi)))

        fdistribu = data.get("fdistribu")
        if fdistribu is not None and mesh_vr is not None:
            rho = compute_density(fdistribu, mesh_vr, mesh_vtheta, species_idx)
            total_density.append(np.sum(rho))
            ek = compute_energy(fdistribu, mesh_vr, mesh_vtheta, species_idx)
            total_energy.append(np.sum(ek))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle("Time Evolution", fontsize=14, fontweight="bold")

    if total_density:
        axes[0].plot(times[:len(total_density)], total_density, "o-",
                     color="#2563EB", linewidth=2, markersize=4)
        axes[0].set_title("Total Density")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel(r"$\sum \rho(r,\theta)$")
        axes[0].grid(True, alpha=0.3)

    if total_energy:
        axes[1].plot(times[:len(total_energy)], total_energy, "s-",
                     color="#DC2626", linewidth=2, markersize=4)
        axes[1].set_title("Total Kinetic Energy")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel(r"$\sum E_k$")
        axes[1].grid(True, alpha=0.3)

    if max_phi:
        axes[2].semilogy(times[:len(max_phi)], max_phi, "^-",
                         color="#059669", linewidth=2, markersize=4)
        axes[2].set_title(r"$\max|\varphi|$")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel(r"$\max|\varphi|$")
        axes[2].grid(True, alpha=0.3)

    out = output_path or (output_dir / "time_evolution.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Init state mesh plot
# ---------------------------------------------------------------------------

def plot_initstate(output_dir: Path, info: dict):
    """Plot mesh configuration from the init state file."""
    if not info:
        print("No init state data available.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("Mesh Configuration", fontsize=14, fontweight="bold")

    panels = [
        ("MeshR",      "Radial mesh",     "r",             "#2563EB"),
        ("MeshTheta",  "Poloidal mesh",   "θ",             "#DC2626"),
        ("MeshVr",     "Velocity-r mesh", r"$v_r$",        "#059669"),
        ("MeshVtheta", "Velocity-θ mesh", r"$v_\theta$",   "#D97706"),
    ]
    for ax, (key, title, ylabel, color) in zip(axes.flat, panels):
        mesh = info.get(key)
        if mesh is not None:
            ax.plot(mesh, "o-", color=color, markersize=3)
            ax.set_title(f"{title} ({len(mesh)} points)")
            ax.set_xlabel("Index")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    out = output_dir / "mesh_info.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot thesis_R_THETA Vlasov–Poisson simulation output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files", nargs="+", type=Path,
        help="HDF5 file(s) or output directory"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Plot all iteration files + time evolution"
    )
    parser.add_argument(
        "--species", type=int, default=0,
        help="Species index (0=electron, 1=ion; default: 0)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output PNG path (single-file mode)"
    )
    parser.add_argument(
        "--initstate", action="store_true",
        help="Plot mesh configuration from initstate"
    )
    parser.add_argument(
        "--potential", action="store_true",
        help="Detailed 4-panel potential plot"
    )
    parser.add_argument(
        "--fdistribu", action="store_true",
        help="Detailed 6-panel distribution function plot"
    )
    args = parser.parse_args()

    # Determine output directory and load init state
    first = args.files[0]
    output_dir = first if first.is_dir() else first.parent
    info = load_initstate(output_dir)

    # Print simulation info
    if info:
        dt = info.get("deltat", np.array("?"))
        niter = info.get("nbiter", np.array("?"))
        nsp = info.get("Nkinspecies", np.array("?"))
        print(f"Simulation: dt={dt}, nbiter={niter}, Nspecies={nsp}")
        for key in ["MeshR", "MeshTheta", "MeshVr", "MeshVtheta"]:
            if key in info:
                print(f"  {key}: {len(info[key])} points, "
                      f"range [{info[key][0]:.4f}, {info[key][-1]:.4f}]")

    # --- Dispatch ---
    if args.initstate:
        plot_initstate(output_dir, info)
        return

    if args.all or first.is_dir():
        files = find_iteration_files(output_dir)
        if not files:
            print(f"No iteration files in {output_dir}")
            sys.exit(1)
        print(f"\nFound {len(files)} iteration file(s)")
        for f in files:
            print(f"  Plotting {f.name} ...")
            plot_snapshot(f, info, species_idx=args.species)
            if args.potential:
                plot_potential_detail(f, info)
            if args.fdistribu:
                plot_fdistribu_detail(f, info, species_idx=args.species)
        print("\nPlotting time evolution ...")
        plot_time_evolution(output_dir, info, species_idx=args.species)

    else:
        for f in args.files:
            if not f.exists():
                print(f"Not found: {f}")
                continue
            if "initstate" in f.name:
                plot_initstate(output_dir, info)
            else:
                plot_snapshot(f, info, output_path=args.output,
                              species_idx=args.species)
                if args.potential:
                    plot_potential_detail(f, info)
                if args.fdistribu:
                    plot_fdistribu_detail(f, info, species_idx=args.species)


if __name__ == "__main__":
    main()
