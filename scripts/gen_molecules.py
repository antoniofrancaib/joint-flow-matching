#!/usr/bin/env python3
"""
Generate molecule configs and export to data/molecules.json.
Molecule format: types, coords, bonds, labels, name.
"""

import json
import os

# Default labels for types 0, 1, 2 (C, N, O)
DEFAULT_LABELS = ["C", "N", "O"]


def make_molecule(name, types, coords, bonds, labels=None):
    """Build a molecule config dict."""
    if labels is None:
        n_types = max(types) + 1 if types else 0
        labels = DEFAULT_LABELS[:n_types]
    return {
        "name": name,
        "types": types,
        "coords": coords,
        "bonds": bonds,
        "labels": labels,
    }


# 5-atom chain: n=5, K=3, types [C, C, N, O, O]
TOY_CHAIN = make_molecule(
    name="toy_chain",
    types=[0, 0, 1, 2, 2],
    coords=[
        [0.0, 0.0, 0.0],
        [1.0, 0.2, 0.0],
        [2.0, -0.2, 0.1],
        [3.0, 0.1, -0.1],
        [4.0, 0.0, 0.0],
    ],
    bonds=[[0, 1], [1, 2], [2, 3], [3, 4]],
    labels=["C", "N", "O"],
)


def main():
    molecules = [TOY_CHAIN]
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "molecules.json")

    with open(out_path, "w") as f:
        json.dump(molecules, f, indent=2)

    print(f"Wrote {out_path} with {len(molecules)} molecule(s)")


if __name__ == "__main__":
    main()
