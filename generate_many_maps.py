import argparse
import pathlib
import numpy as np
from src.maze_generator import create_maze


def parse_args():
    parser = argparse.ArgumentParser(description="Generate many maps")
    parser.add_argument(
        "--num-coarse-row-cells",
        type=int,
        default=7,
        help="Number of coarse row cells",
    )
    parser.add_argument(
        "--num-coarse-col-cells",
        type=int,
        default=7,
        help="Number of coarse column cells",
    )
    parser.add_argument(
        "--num-fine-row-cells", type=int, default=10, help="Number of fine row cells"
    )
    parser.add_argument(
        "--num-fine-col-cells", type=int, default=10, help="Number of fine column cells"
    )
    parser.add_argument(
        "--coarse-path-min-coverage",
        type=float,
        default=0.3,
        help="Coarse path min coverage (min: 0, max: 1)",
    )
    parser.add_argument(
        "--coarse-path-max-coverage",
        type=float,
        default=0.7,
        help="Coarse path min coverage (min: 0, max: 1)",
    )
    parser.add_argument(
        "--max-coarse-map-trial",
        type=int,
        default=-1,
        help="Max coarse map trial. -1 for no limit",
    )
    parser.add_argument(
        "--coarse-cell-fill-factor",
        type=float,
        default=0.5,
        help="How densly to fill the coarse cells (min: 0, max: 1)",
    )
    parser.add_argument(
        "--min-path-width",
        type=int,
        default=2,
        help="Min path width",
    )
    parser.add_argument(
        "--num-maps",
        type=int,
        default=1,
        help="Number of maps to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Show time performance",
    )
    return parser.parse_args()


def main(args):
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("Generating maps...")
    for i in range(args.num_maps):
        maze = create_maze(
            args.num_coarse_row_cells,
            args.num_coarse_col_cells,
            args.num_fine_row_cells,
            args.num_fine_col_cells,
            args.coarse_path_min_coverage,
            args.coarse_path_max_coverage,
            args.max_coarse_map_trial,
            args.coarse_cell_fill_factor,
            args.min_path_width,
            args.verbose,
        )
        np.savetxt(f"{args.output_dir}/maze_{i}.csv", maze, fmt="%i", delimiter=",")
        print(f"Generated maze_{i}.csv")


if __name__ == "__main__":
    args = parse_args()
    main(args)
