import logging
from collections import defaultdict

import numpy as np

from core.enums import TeamEnum
from core.robot import BaseRobot
from core.tile import BaseTile, DepositTile


class Board:
    def __init__(
        self,
        board_size: tuple[int, int] = (20, 20),
        deposit_pos: tuple[tuple[int, int], tuple[int, int]] | None = None,
        total_gold_count: int = 40,
    ) -> None:
        if board_size[0] <= 0 or board_size[1] <= 0:
            raise ValueError("board_size cannot be less than 0")

        self.tiles = np.array(
            [[BaseTile() for _ in range(board_size[1])]
             for _ in range(board_size[0])]
        )
        self.robot_locations: dict[tuple[int, int],
                                   set[BaseRobot]] = defaultdict(set)
        self.board_size = board_size

        # Setup the deposit tiles
        if deposit_pos is not None:
            if not (
                self.is_valid_position(deposit_pos[0])
                and self.is_valid_position(deposit_pos[1])
            ):
                raise ValueError("Deposite pos is not valid")

            self.red_deposit_pos, self.blue_deposit_pos = deposit_pos
        else:
            self.red_deposit_pos = self.get_random_tile_pos()
            self.blue_deposit_pos = self.get_random_tile_pos()
            while self.blue_deposit_pos == self.red_deposit_pos:
                self.blue_deposit_pos = self.get_random_tile_pos()

        self._set_tile(self.red_deposit_pos, DepositTile(team=TeamEnum.RED))
        self._set_tile(self.blue_deposit_pos, DepositTile(team=TeamEnum.BLUE))

        # Randomly add gold bars across the field
        while total_gold_count != 0:
            pos = self.get_random_tile_pos()

            if type(tile := self.get_tile(pos)) is BaseTile:
                tile.add()
                total_gold_count -= 1

    def get_random_tile_pos(self) -> tuple[int, int]:
        return (
            np.random.randint(0, self.tiles.shape[1]),
            np.random.randint(0, self.tiles.shape[0]),
        )

    def _set_tile(self, pos: tuple[int, int], value: BaseTile):
        self.tiles[pos[1], pos[0]] = value

    def get_tile(self, pos: tuple[int, int]) -> BaseTile:
        # Positions are treated as (x, y) corresponding to (column, row)
        return self.tiles[pos[1], pos[0]]

    def get_deposit_tile(self, team: TeamEnum) -> DepositTile:
        pos = self.red_deposit_pos if team == TeamEnum.RED else self.blue_deposit_pos
        return self.get_tile(pos)  # pyright: ignore

    def add_robot(self, robot: BaseRobot, pos: tuple[int, int]):
        self.robot_locations[pos].add(robot)
        robot.pos = pos

    def move_robot(self, robot: BaseRobot, new_pos: tuple[int, int]):
        self.robot_locations[robot.pos].remove(robot)

        # Cleanup any empty key
        if not self.robot_locations[robot.pos]:
            del self.robot_locations[robot.pos]

        self.add_robot(robot, new_pos)

    def get_robots_at(self, pos: tuple[int, int]) -> set[BaseRobot]:
        if not self.is_valid_position(pos):
            raise IndexError("Position is not valid")

        return self.robot_locations.get(pos, set())

    def is_valid_position(self, pos: tuple[int, int] | np.ndarray) -> bool:
        return 0 <= pos[1] < self.tiles.shape[0] and 0 <= pos[0] < self.tiles.shape[1]
