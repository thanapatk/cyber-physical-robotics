from core.enums import Direction


def get_pos_to_add(direction: Direction) -> tuple[int, int]:
    return {
        Direction.NORTH: (0, -1),
        Direction.SOUTH: (0, 1),
        Direction.EAST: (1, 0),
        Direction.WEST: (-1, 0),
    }[direction]
