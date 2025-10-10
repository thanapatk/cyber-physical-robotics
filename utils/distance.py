def manhattan_distance(start_tile: tuple[int, int], end_tile: tuple[int, int]) -> int:
    return abs(end_tile[0] - start_tile[0]) + abs(end_tile[1] - start_tile[1])
