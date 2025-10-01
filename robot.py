from collections import deque
from enum import Enum
from typing import Deque

import numpy as np
from pydantic import BaseModel

from core.actions import Action, MoveAction, TurnAction
from core.enums import Direction
from core.message_handler import Message
from core.robot import BaseRobot, Observation, TeamInfo


class RobotState(Enum):
    EXPLORING = 0
    # COORDINATING = 1
    # EXECUTING = 2


class SensedTile(BaseModel):
    step: int
    gold_count: int
    # same_team_robot_count: int


class ObservationsMessage(Message):
    value: list[Observation]


class Robot(BaseRobot):
    def __init__(
        self,
        robot_id: int,
        team_info: TeamInfo,
        pos: tuple[int, int],
        direction: Direction,
        board_size: tuple[int, int],
    ) -> None:
        super().__init__(robot_id, team_info, pos, direction)

        self.state = RobotState.EXPLORING
        self.sensed_map: dict[tuple[int, int], SensedTile] = dict()

        self.target_pos: tuple[int, int] | None = None
        self.saved_actions: Deque[Action] = deque()

        self.board_size = board_size
        self.yy, self.xx = np.mgrid[0 : self.board_size[1], 0 : self.board_size[0]]

    def generate_cost_matrix(self):
        x, y = self.pos

        dx = np.abs(self.xx - x)
        dy = np.abs(self.yy - y)

        manhattan_distance = dx + dy

        turn_cost = np.ones_like(manhattan_distance)

        if self.direction == Direction.NORTH:
            turn_cost[y:, x] = 0

            x_indices = np.where(np.arange(turn_cost.shape[1]) != x)
            y_indices = np.where(np.arange(turn_cost.shape[0]) > y)
        elif self.direction == Direction.SOUTH:
            turn_cost[: y + 1, x] = 0

            x_indices = np.where(np.arange(turn_cost.shape[1]) != x)
            y_indices = np.where(np.arange(turn_cost.shape[0]) < y)
        elif self.direction == Direction.EAST:
            turn_cost[y, x:] = 0

            x_indices = np.where(np.arange(turn_cost.shape[1]) < x)
            y_indices = np.where(np.arange(turn_cost.shape[0]) != y)
        else:  # self.direction == Direction.WEST:
            turn_cost[y, : x + 1] = 0

            x_indices = np.where(np.arange(turn_cost.shape[1]) > x)
            y_indices = np.where(np.arange(turn_cost.shape[0]) != y)

        turn_cost[np.ix_(*y_indices, *x_indices)] = 2

        return manhattan_distance + turn_cost

    def get_coldness_map(self, step: int):
        coldness_map = np.full(self.board_size, 2000)

        for pos, sensed_tile in self.sensed_map.items():
            coldness_map[pos[1], pos[0]] = step - sensed_tile.step

        return coldness_map

    def decide_exploration_target(self, step: int) -> tuple[int, int]:
        cost_matrix = self.generate_cost_matrix()
        coldness_map = self.get_coldness_map(step)

        COLDNESS_WEIGHT = 1.5
        exploration_score_map = (coldness_map * COLDNESS_WEIGHT) - cost_matrix

        # Prevent selecting the current tile
        exploration_score_map[self.pos[1], self.pos[0]] = -999

        target_loc = np.unravel_index(np.argmax(exploration_score_map), self.board_size)

        return int(target_loc[1]), int(target_loc[0])

    def pathfind(self, target: tuple[int, int]) -> list[Action]:
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]

        actions = []

        is_helping_x = (dx > 0 and self.direction == Direction.EAST) or (
            dx < 0 and self.direction == Direction.WEST
        )
        is_helping_y = (dy > 0 and self.direction == Direction.SOUTH) or (
            dy < 0 and self.direction == Direction.NORTH
        )

        if is_helping_x:
            actions.extend([MoveAction(robot_id=self.robot_id)] * abs(dx))
            if dy != 0:
                actions.append(
                    TurnAction(
                        robot_id=self.robot_id,
                        new_direction=Direction.SOUTH if dy > 0 else Direction.NORTH,
                    )
                )
                actions.extend([MoveAction(robot_id=self.robot_id)] * abs(dy))
        elif is_helping_y:
            actions.extend([MoveAction(robot_id=self.robot_id)] * abs(dy))
            if dx != 0:
                actions.append(
                    TurnAction(
                        robot_id=self.robot_id,
                        new_direction=Direction.EAST if dx > 0 else Direction.WEST,
                    )
                )
                actions.extend([MoveAction(robot_id=self.robot_id)] * abs(dx))
        else:
            if dx != 0:
                actions.append(
                    TurnAction(
                        robot_id=self.robot_id,
                        new_direction=Direction.EAST if dx > 0 else Direction.WEST,
                    )
                )
                actions.extend([MoveAction(robot_id=self.robot_id)] * abs(dx))
            if dy != 0:
                actions.append(
                    TurnAction(
                        robot_id=self.robot_id,
                        new_direction=Direction.SOUTH if dy > 0 else Direction.NORTH,
                    )
                )
                actions.extend([MoveAction(robot_id=self.robot_id)] * abs(dy))

        return actions

    def _update_sensed_tile(self, step: int, observation: Observation):
        pos = observation.pos

        # Only create new `SensedTile` when not found
        sensed_tile = self.sensed_map.get(pos, SensedTile(step=-1, gold_count=0))

        # Skip if the observation is outdated
        if sensed_tile.step > step:
            return

        sensed_tile.step = step
        sensed_tile.gold_count = observation.gold_count

        self.sensed_map[pos] = sensed_tile

    def decide_action(self, step: int, observations: list[Observation]) -> Action:
        # Update the `sensed_map`
        while len(self.incomming_messages) != 0:
            message = self.incomming_messages.popleft()

            if type(message) is ObservationsMessage:
                step = message.step
                for observation in message.value:
                    self._update_sensed_tile(message.step, observation)

        for observation in observations:
            self._update_sensed_tile(step, observation)

        # Broadcast the observation
        self.outgoing_messages.append(
            (
                None,
                ObservationsMessage(
                    sender_id=self.robot_id, step=step, value=observations
                ),
            )
        )

        # if len(self.saved_actions) != 0:
        #     return self.saved_actions.popleft()

        if self.state == RobotState.EXPLORING:
            target = self.decide_exploration_target(step)

            if target != self.target_pos:
                self.target_pos = target
                self.saved_actions.clear()
                self.saved_actions.extend(self.pathfind(self.target_pos))

            return self.saved_actions.popleft()

        return Action(robot_id=self.robot_id)
