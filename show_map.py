import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Show map")
    parser.add_argument(
        "--map-file",
        type=str,
        default="results/maze_0.csv",
        help="Map file",
    )
    return parser.parse_args()


def main(args):
    map = np.loadtxt(args.map_file, delimiter=",")
    print(map)
    plt.imshow(map, cmap="gray")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
