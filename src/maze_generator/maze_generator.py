import random
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import time


def create_maze(
    num_coarse_row_cells: int,
    num_coarse_col_cells: int,
    num_fine_row_cells: int,
    num_fine_col_cells: int,
    coarse_path_min_coverage: float,
    max_coarse_map_trial: int,
    wall_attach_steps: int,
    min_path_width: int,
    verbose: bool,
) -> np.ndarray:

    coarse_maze_start = time.time()
    coarse_maze = create_coarse_maze(
        num_coarse_row_cells,
        num_coarse_col_cells,
        coarse_path_min_coverage,
        max_coarse_map_trial,
    )
    coarse_maze_end = time.time()
    if verbose:
        print(f"Coarse maze created in {coarse_maze_end - coarse_maze_start} seconds.")

    fine_maze_start = time.time()
    maze = attach_walls_to_coarse_maze(
        coarse_maze,
        num_fine_row_cells,
        num_fine_col_cells,
        wall_attach_steps,
        min_path_width,
    )
    fine_maze_end = time.time()
    if verbose:
        print(f"Fine maze created in {fine_maze_end - fine_maze_start} seconds.")

    sparse_maze = convert_to_sparse_maze(maze)

    return sparse_maze.astype(np.uint8)


def create_coarse_maze(
    num_coarse_row_cells: int,
    num_coarse_col_cells: int,
    coarse_path_min_coverage: float,
    max_coarse_map_trial: int,
):
    coarse_maze = create_random_coarse_maze(num_coarse_col_cells, num_coarse_row_cells)
    path_coverage = calculate_path_coverage(coarse_maze)

    coarse_map_trial_count = 0

    no_limit = max_coarse_map_trial == -1

    while path_coverage < coarse_path_min_coverage and (
        no_limit or coarse_map_trial_count < max_coarse_map_trial
    ):
        coarse_maze = create_random_coarse_maze(
            num_coarse_col_cells, num_coarse_row_cells
        )
        path_coverage = calculate_path_coverage(coarse_maze)
        coarse_map_trial_count += 1

    if path_coverage < coarse_path_min_coverage:
        raise Exception("Failed to create a coarse maze.")

    return coarse_maze


def calculate_path_coverage(maze: np.ndarray) -> float:
    return np.mean(1 - maze)


def create_random_coarse_maze(
    num_coarse_row_cells: int, num_coarse_col_cells: int
) -> np.ndarray:
    maze = np.ones((num_coarse_row_cells, num_coarse_col_cells), dtype=np.uint8)
    # backtracking maze is for checking cells that are at least once visted.
    backtracking_maze = np.ones(
        (num_coarse_row_cells, num_coarse_col_cells), dtype=np.uint8
    )

    # always the left top cell is the start
    entry_row = 1
    entry_col = 0
    maze[entry_row, entry_col] = 0
    backtracking_maze[entry_row, entry_col] = 0
    cursor = (entry_row, entry_col)
    backtracking_stack = [cursor]

    # then there is only one possible step
    maze[entry_row, entry_col + 1] = 0
    backtracking_maze[entry_row, entry_col + 1] = 0
    cursor = (entry_row, entry_col + 1)

    ### CUROSOR DOES NOT POINT A CELL IN THE BACKTRACKING STACK
    while True:
        # TODO: prevernt infinite loop
        next_cell = select_next_random_cell(backtracking_maze, cursor[0], cursor[1])
        # reset the backtrack point after using it to prevent infinite loop

        if next_cell is not None:
            maze[next_cell[0], next_cell[1]] = 0
            backtracking_maze[next_cell[0], next_cell[1]] = 0
            backtracking_stack.append(cursor)
            cursor = next_cell

            if is_on_the_map_boundary(maze, cursor[0], cursor[1]):
                # cursor is on the edge of the map
                # this will be the exit point
                break
        else:
            # DFS fail, backtrack

            if len(backtracking_stack) == 0:
                raise Exception("Failed to create a coarse maze. DFS fail.")

            # only unvisit the maze cell, not the backtracking maze cell.
            maze[cursor[0], cursor[1]] = 1
            cursor = backtracking_stack.pop()

    return maze


def select_next_random_cell(
    maze: np.ndarray,
    current_row: int,
    current_col: int,
) -> tuple[int, int]:
    """Select a random cell that is not visited, not adjacent to the visited cells, and not at the boundary"""

    def is_cell_valid(row: int, col: int) -> bool:
        return (
            is_in_the_map(maze, row, col)
            and is_not_visited(maze, row, col)
            and is_not_adjacent_to_visited(maze, current_row, current_col, row, col)
        )

    valid_next_cells: list[(int, int)] = []
    candiates = [
        (current_row - 1, current_col),
        (current_row + 1, current_col),
        (current_row, current_col - 1),
        (current_row, current_col + 1),
    ]

    # check 4 adjacent cells
    for row, col in candiates:
        if is_cell_valid(row, col):
            valid_next_cells.append((row, col))

    if len(valid_next_cells) == 0:
        return None

    return random.choice(valid_next_cells)


def is_in_the_map(maze: np.ndarray, row: int, col: int) -> bool:
    n_maze_row = maze.shape[0]
    n_maze_col = maze.shape[1]
    return 0 <= row < n_maze_row and 0 <= col < n_maze_col


def is_on_the_map_boundary(maze: np.ndarray, row: int, col: int) -> bool:
    n_maze_row = maze.shape[0]
    n_maze_col = maze.shape[1]
    return 0 == row or row == n_maze_row - 1 or 0 == col or col == n_maze_col - 1


def is_not_visited(maze: np.ndarray, row: int, col: int) -> bool:
    return maze[row, col] == 1


def is_not_adjacent_to_visited(
    maze: np.ndarray, current_row: int, current_col: int, next_row: int, next_col: int
) -> bool:
    if not is_in_the_map(maze, next_row, next_col):
        return False

    adjacent_cells_except_current = []
    for cell in [
        (next_row - 1, next_col),
        (next_row + 1, next_col),
        (next_row, next_col + 1),
        (next_row, next_col - 1),
    ]:
        if not is_in_the_map(maze, cell[0], cell[1]):
            continue

        if cell != (current_row, current_col):
            adjacent_cells_except_current.append(cell)

    for cell in adjacent_cells_except_current:
        if maze[cell[0], cell[1]] == 0:
            return False

    return True


def attach_walls_to_coarse_maze(
    coarse_maze: np.ndarray,
    num_fine_row_cells: int,
    num_fine_col_cells: int,
    wall_attach_steps: int,
    min_path_width: int,
) -> np.ndarray:
    maze = convert_to_fine_maze(coarse_maze, num_fine_row_cells, num_fine_col_cells)
    maze = cross_connect_components(maze)
    maze = iteratively_attach_walls(maze, wall_attach_steps, min_path_width)
    return maze


def convert_to_fine_maze(
    coarse_maze: np.ndarray, num_fine_row_cells: int, num_fine_col_cells: int
):
    sub_cell = np.ones((num_fine_row_cells, num_fine_col_cells), dtype=np.uint8)
    maze = np.kron(coarse_maze, sub_cell)
    return maze


def cross_connect_components(maze: np.ndarray) -> np.ndarray:
    """Connect components in the maze with a path"""

    kernal = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    cross_connected_maze = ndimage.convolve(maze, kernal, mode="constant", cval=0)
    cross_connected_maze[cross_connected_maze > 0] = 1
    return cross_connected_maze


def iteratively_attach_walls(
    maze: np.ndarray, steps: int, min_path_width: int
) -> np.ndarray:

    labeled_maze, num_features = ndimage.label(maze)

    maze_A, maze_B = split_components(maze, labeled_maze)

    iter_count = 0
    while iter_count < steps:
        maze_A, maze_B = add_random_wall_point(
            maze_A, maze_B, min_path_width=min_path_width
        )
        iter_count += 1

    new_maze = maze_A + maze_B

    return new_maze


def split_components(
    maze: np.ndarray, labeled_maze: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # ensure paths are 0 and the other compoents are >0
    binary_maze = maze * (labeled_maze + 1)
    labels = np.unique(binary_maze)
    label_A = np.median(labels)
    label_B = np.max(labels)

    maze_A = np.zeros_like(binary_maze)
    maze_A[binary_maze == label_A] = 1
    maze_B = np.zeros_like(binary_maze)
    maze_B[binary_maze == label_B] = 1

    return (maze_A, maze_B)


def select_near_boundary_points(maze: np.ndarray) -> list[(int, int)]:

    cross_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    kernel_sum = np.sum(cross_kernel)

    conved_maze = ndimage.convolve(maze, cross_kernel, mode="constant", cval=1)

    near_boundary_points = np.argwhere((conved_maze < kernel_sum) & (conved_maze > 0))

    # exclude the map boundary
    near_boundary_points = filter(
        lambda x: not is_on_the_map_boundary(maze, x[0], x[1]), near_boundary_points
    )
    # exclude the walls
    near_boundary_points = filter(lambda x: maze[x[0], x[1]] == 0, near_boundary_points)

    return list(near_boundary_points)


def select_random_near_boundary_point(component_maze: np.ndarray) -> tuple[int, int]:
    near_boundary_points = select_near_boundary_points(component_maze)
    if len(near_boundary_points) == 0:
        return None
    return random.choice(near_boundary_points)


def calculate_min_dist_to_point(maze: np.ndarray, row: int, col: int) -> int:
    boundary_points = select_near_boundary_points(maze)
    distances = [abs(row - point[0]) + abs(col - point[1]) for point in boundary_points]
    return min(distances)


def add_random_wall_point(
    maze_A: np.ndarray, maze_B: np.ndarray, min_path_width: float
) -> tuple[np.ndarray, np.ndarray]:
    maze_to_add_point = random.choice([maze_A, maze_B])
    add_point_to_maze_A = maze_to_add_point is maze_A
    other_maze = maze_A if maze_to_add_point is maze_B else maze_B

    random_near_boudary_point = select_random_near_boundary_point(maze_to_add_point)

    if random_near_boudary_point is None:
        return (maze_A, maze_B)

    min_dist_to_point = calculate_min_dist_to_point(
        other_maze, random_near_boudary_point[0], random_near_boudary_point[1]
    )

    if min_dist_to_point < min_path_width:
        return (maze_A, maze_B)

    copied_maze_A = maze_A.copy()
    copied_maze_B = maze_B.copy()
    if add_point_to_maze_A:
        copied_maze_A[random_near_boudary_point[0], random_near_boudary_point[1]] = 1
    else:
        copied_maze_B[random_near_boudary_point[0], random_near_boudary_point[1]] = 1

    return copied_maze_A, copied_maze_B


def convert_to_sparse_maze(maze: np.ndarray) -> np.ndarray:
    negative_maze = np.ones_like(maze) - maze
    near_boundary_points = select_near_boundary_points(negative_maze)
    sparse_maze = np.zeros_like(maze)

    for point in near_boundary_points:
        sparse_maze[point[0], point[1]] = 1

    return sparse_maze
