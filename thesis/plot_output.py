#!/usr/bin/env python3
"""Plot a quick sanity view of a Gyselalibxx PDI/HDF5 output file."""

import argparse
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_dataset(path: Path, name: str) -> np.ndarray:
    text = subprocess.check_output(
        ["h5dump", "-y", "-d", name, str(path)], text=True
    )
    shape_match = re.search(r"DATASPACE\s+SIMPLE\s+\{\s+\(\s*([^)]*)\s*\)", text)
    if "DATA {" not in text:
        raise RuntimeError(f"Could not read dataset {name!r} from {path}")

    shape = () if shape_match is None else tuple(
        int(value.strip()) for value in shape_match.group(1).split(",")
    )
    data = text.split("DATA {", 1)[1].rsplit("}", 2)[0]
    values = np.fromstring(data, sep=",")
    return values.reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="HDF5 file to plot")
    parser.add_argument("--output", type=Path, help="PNG output path")
    args = parser.parse_args()

    density_name = "density" if "_initstate" not in args.file.name else "density_eq"
    potential_name = (
        "electrical_potential"
        if "_initstate" not in args.file.name
        else "electrical_potential_eq"
    )
    density = read_dataset(args.file, density_name)
    potential = read_dataset(args.file, potential_name)

    if "_initstate" in args.file.name:
        x = read_dataset(args.file, "x_coords")
        y = read_dataset(args.file, "y_coords")
    else:
        initial = args.file.with_name("GYSELALIBXX_initstate.h5")
        x = read_dataset(initial, "x_coords")
        y = read_dataset(initial, "y_coords")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (density, potential),
        (density_name.replace("_", " ").title(), potential_name.replace("_", " ").title()),
    ):
        image = axis.pcolormesh(x, y, values, shading="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        figure.colorbar(image, ax=axis)

    if "_initstate" not in args.file.name:
        time = read_dataset(args.file, "time").item()
        figure.suptitle(f"Czarny simulation, t = {time:g}")
    else:
        figure.suptitle("Czarny simulation initial state")

    output = args.output or args.file.with_suffix(".png")
    figure.savefig(output, dpi=150)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
