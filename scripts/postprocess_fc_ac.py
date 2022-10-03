import os

import h5py
import numpy as np


def get_key(h5_name: str) -> int:
    return int(h5_name.split("-")[-1].split(".")[0])


def process_single_rms_field(
    input_path: str, output_path: str, verbose: bool = False
) -> None:
    h5_files = [p for p in os.listdir(input_path) if p.endswith(".h5")]
    h5_files = sorted(h5_files, key=get_key)

    with h5py.File(output_path, "x") as out:
        data_grp = out.create_group("data")
        for i, h5_file in enumerate(h5_files):
            with h5py.File(os.path.join(input_path, h5_file), "r") as f:
                if verbose:
                    print(h5_file)
                solve_steps = np.sort(np.array([int(key) for key in f["data"]]))
                if i == 0:
                    out["mesh"] = f["solution/device/mesh"]
                    for step in solve_steps:
                        data_grp[str(step)] = f["data"][str(step)]
                else:
                    step = solve_steps[-1]
                    data_grp[str(step + i)] = f["data"][str(step)]

    return output_path


def process_many_rms_fields(
    input_dir: str, output_dir: str, verbose: bool = True
) -> None:
    input_paths = []
    for p in os.listdir(input_dir):
        try:
            _ = int(p)
            input_paths.append(p)
        except ValueError:
            pass
    input_paths = sorted(input_paths, key=int)
    os.makedirs(output_dir, exist_ok=False)
    for path in input_paths:
        output_path = os.path.join(output_dir, f"run-{path:02}.h5")
        process_single_rms_field(
            os.path.join(input_dir, path), output_path, verbose=verbose
        )


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, help="Input directory.")
    parser.add_argument("-o", "--output", type=str, help="Output directory.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose.")

    args = parser.parse_args()

    process_many_rms_fields(args.input, args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
