from abc import ABC, abstractmethod
from collections import deque
from typing import Deque

import numpy as np
from pydantic import BaseModel

from core.actions import Action
from core.enums import Direction, TeamEnum
from core.message_handler import Message


class Observation(BaseModel):
    pos: tuple[int, int]
    gold_count: int
    same_team_robot_count: int


class TeamInfo(BaseModel):
    team: TeamEnum
    deposit_pos: tuple[int, int]

    def __eq__(self, other):
        return self.team == other.team


class BaseRobot(ABC):
    def __init__(
        self,
        robot_id: int,
        team_info: TeamInfo,
        pos: tuple[int, int],
        direction: Direction,
    ) -> None:
        self.robot_id = robot_id
        self.team_info = team_info
        self.pos = pos
        self.direction = direction
        self.partner_id: int | None = None

        self.incomming_messages: Deque[Message] = deque()
        self.outgoing_messages: Deque[tuple[int | None, Message]] = deque()

    def _get_observation_pos(self):
        base_i = (1, 2)
        base_j = (range(-1, 2), range(-2, 3))

        if self.direction.value < 2:  # NORTH, SOUTH
            output = [
                (j, i * (-1 if self.direction == Direction.SOUTH else 1))
                for i, all_j in zip(base_i, base_j)
                for j in all_j
            ]
        else:  # EAST, WEST
            output = [
                (i * (-1 if self.direction == Direction.WEST else 1), j)
                for i, all_j in zip(base_i, base_j)
                for j in all_j
            ]

        return np.array(self.pos) + np.array(output)

    def _is_same_team(self, robot: "BaseRobot") -> bool:
        return self.team_info == robot.team_info

    def observe(self, board) -> list[Observation]:
        all_observable_pos = self._get_observation_pos()

        output = []
        for pos in filter(board.is_valid_position, all_observable_pos):
            pos = tuple(pos)

            gold_count = board.get_tile(pos).gold_count
            robot_count = 0
            for robot in board.get_robots_at(pos):
                if self._is_same_team(robot):
                    robot_count += 1

            output.append(
                Observation(
                    pos=pos,
                    gold_count=gold_count,
                    same_team_robot_count=robot_count,
                )
            )

        return output

    @abstractmethod
    def decide_action(self, step: int, observations: list[Observation]) -> Action:
        pass
