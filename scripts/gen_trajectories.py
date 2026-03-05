#!/usr/bin/env python3
"""
Generate ODE trajectories from data/molecules.json (Algorithm 3).
Oracle denoiser D(y_t, t) = y_1; Euler integration; mulberry32(42) for noise.
Exports to data/trajectories/<molecule_name>.json
"""

import json
import math
import os
import sys


def mulberry32(seed):
    """Port of JS mulberry32 for pixel-identical trajectories."""
    state = seed & 0xFFFFFFFF

    def next_():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = (state ^ (state >> 15)) & 0xFFFFFFFF
        t = (t * (state | 1)) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF) ^ t
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return next_


def gaussian_pair(rng):
    """Box-Muller: two N(0,1) samples from two U(0,1)."""
    u1 = rng()
    u2 = rng()
    r = math.sqrt(-2 * math.log(u1 + 1e-10))
    return (r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2))


def compute_trajectory(mol, n_steps=60, seed=42):
    """
    Run Algorithm 3 with oracle denoiser D(y_t, t) = y_1.
    Returns {y0, y1, traj, n_atoms, n_types, n_steps}.
    """
    types = mol["types"]
    coords = mol["coords"]
    n_atoms = len(types)
    n_types = max(types) + 1
    n_dim = n_types + 3  # K + 3 per atom

    rng = mulberry32(seed)

    # y1: target [one-hot ; x,y,z]
    y1 = []
    for i in range(n_atoms):
        one_hot = [0.0] * n_types
        one_hot[types[i]] = 1.0
        y1.append(one_hot + list(coords[i]))

    # y0: N(0, I) via Box-Muller
    y0 = []
    for i in range(n_atoms):
        atom = []
        for d in range(0, n_dim, 2):
            pair = gaussian_pair(rng)
            atom.append(pair[0])
            if d + 1 < n_dim:
                atom.append(pair[1])
        y0.append(atom[:n_dim])

    # Euler integration
    traj = [list(a[:] for a in y0)]
    dt = 1.0 / n_steps

    for m in range(n_steps):
        t_val = m / n_steps
        prev = traj[m]
        next_state = []

        for i in range(n_atoms):
            atom = []
            for d in range(n_dim):
                vel = (y1[i][d] - prev[i][d]) / (1 - t_val + 1e-8)
                atom.append(prev[i][d] + dt * vel)
            next_state.append(atom)
        traj.append(next_state)

    return {
        "y0": y0,
        "y1": y1,
        "traj": traj,
        "n_atoms": n_atoms,
        "n_types": n_types,
        "n_steps": n_steps,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    mol_path = os.path.join(data_dir, "molecules.json")
    traj_dir = os.path.join(data_dir, "trajectories")

    if not os.path.exists(mol_path):
        print(f"Error: {mol_path} not found. Run gen_molecules.py first.", file=sys.stderr)
        sys.exit(1)

    with open(mol_path) as f:
        molecules = json.load(f)

    os.makedirs(traj_dir, exist_ok=True)

    for mol in molecules:
        name = mol["name"]
        data = compute_trajectory(mol)
        out_path = os.path.join(traj_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
