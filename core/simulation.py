import logging
from collections import defaultdict
from typing import Sequence

import numpy as np

from core.actions import Action, MoveAction, PickupAction, TurnAction
from core.board import Board, DepositTile
from core.enums import TeamEnum
from core.message_handler import Message, MessageHandler
from core.robot import BaseRobot
from utils.direction import get_pos_to_add


class SimulationController:
    def __init__(self, robots: Sequence[BaseRobot], board: Board) -> None:
        self.robots: Sequence[BaseRobot] = robots
        self.board: Board = board
        self.step_count: int = 0
        self.logger = logging.getLogger(__name__)

        for robot in self.robots:
            self.board.add_robot(robot=robot, pos=robot.pos)

        self.message_handler = MessageHandler()

    def step(self):
        self.handle_outgoing_messages()

        actions = self.collect_actions()
        messages = self.collect_messages()

        self.handle_incoming_messages(messages)

        validated_actions = self.resolve_conflicts(actions)

        self.logger.debug(
            f"Simulation step {self.step_count}:\n\tAll actions: {actions}\n\tValid actions: {actions}"
        )

        self.execute_actions(validated_actions)
        self.handle_gold_deposit()

        self.step_count += 1

    def collect_actions(self) -> list[Action]:
        actions = []
        for robot in self.robots:
            observations = robot.observe(self.board)
            actions.append(robot.decide_action(self.step_count, observations))

        return actions

    def _get_new_pos(self, action: MoveAction) -> tuple[int, int]:
        robot = self.robots[action.robot_id]
        pos_to_add = get_pos_to_add(robot.direction)
        return (robot.pos[0] + pos_to_add[0], robot.pos[1] + pos_to_add[1])

    def _is_valid_move_action(self, action: MoveAction) -> bool:
        return self.board.is_valid_position(self._get_new_pos(action))

    def pickup_gold(self, robot_1: BaseRobot, robot_2: BaseRobot):
        if robot_1.partner_id or robot_2.partner_id:
            raise ValueError("One or more robots is already carrying gold")

        robot_1.partner_id = robot_2.robot_id
        robot_2.partner_id = robot_1.robot_id

        self.board.get_tile(robot_1.pos).take()

    def drop_gold(self, robot_1: BaseRobot, robot_2: BaseRobot):
        if (
            robot_1.partner_id != robot_2.robot_id
            or robot_2.partner_id != robot_1.robot_id
        ):
            raise ValueError("The robots are not a carrying pair")

        pos = robot_1.pos
        tile = self.board.get_tile(pos)

        robot_1.partner_id = None
        robot_2.partner_id = None
        tile.add()

    def validate_pickup_actions(
        self, actions_map: dict[tuple[int, int], list[PickupAction]]
    ) -> list[PickupAction]:
        valid_actions = []

        for pos, actions in actions_map.items():
            tile = self.board.get_tile(pos)
            gold_count = tile.gold_count

            red_team_actions = []
            blue_team_actions = []

            for action in actions:
                robot = self.robots[action.robot_id]
                if robot.team_info.team == TeamEnum.RED:
                    red_team_actions.append(action)
                else:
                    blue_team_actions.append(action)

            if (
                len(red_team_actions) == 2
                and len(blue_team_actions) == 2
                and gold_count >= 2
            ):
                valid_actions.extend(actions)
            elif len(red_team_actions) == 2 and gold_count >= 1:
                valid_actions.extend(red_team_actions)
            elif len(blue_team_actions) == 2 and gold_count >= 1:
                valid_actions.extend(blue_team_actions)

        return valid_actions

    def validate_paired_actions(
        self, paired_actions: dict[tuple[int, int], list[Action]]
    ) -> list[Action]:
        valid_actions = []
        for (robot_id_1, robot_id_2), actions in paired_actions.items():
            action_1, action_2 = actions

            any_pickup_action = PickupAction in map(type, actions)
            is_same_action = type(action_1) is type(action_2)
            is_move_actions = (
                type(action_1) is MoveAction and type(action_2) is MoveAction
            )
            is_same_direction = (
                self.robots[robot_id_1].direction == self.robots[robot_id_2].direction
            )

            if any_pickup_action:
                self.logger.debug(
                    f"One or more of ({robot_id_1}, {robot_id_2}) pair trys to pick up gold"
                )
                continue
            elif not is_same_action or (is_move_actions and not is_same_direction):
                self.drop_gold(
                    self.robots[robot_id_1], self.robots[robot_id_2]
                )  # Drop the gold but allow the actions

            valid_actions.extend(
                [
                    action
                    for action in actions
                    if type(action) != MoveAction or self._is_valid_move_action(action)
                ]
            )

        return valid_actions

    def resolve_conflicts(self, actions: list[Action]) -> list[Action]:
        paired_actions = defaultdict(list)  # (robot_id_1, robot_id_2): list[Action]
        valid_move_actions = []
        pickup_actions = defaultdict(list)  # (x, y): list[Action]
        other_actions = []

        for action in actions:
            if (partner_id := self.robots[action.robot_id].partner_id) is not None:
                paired_id = sorted([action.robot_id, partner_id])
                paired_actions[tuple(paired_id)].append(action)
            elif type(action) is MoveAction:
                if self._is_valid_move_action(action):
                    valid_move_actions.append(action)
            elif type(action) is PickupAction:
                pickup_actions[action.pos].append(action)
            else:
                other_actions.append(action)

        valid_pickup_actions = self.validate_pickup_actions(pickup_actions)
        valid_paired_actions = self.validate_paired_actions(paired_actions)

        return (
            other_actions
            + valid_move_actions
            + valid_pickup_actions
            + valid_paired_actions
        )

    def execute_actions(self, actions: list[Action]) -> None:
        pickup_actions = defaultdict(list)

        for action in actions:
            robot = self.robots[action.robot_id]

            if type(action) is TurnAction:
                robot.direction = action.new_direction
            elif type(action) is MoveAction:
                new_pos = self._get_new_pos(action)
                self.board.move_robot(robot=robot, new_pos=new_pos)
            elif type(action) is PickupAction:
                pickup_actions[action.pos].append(action)

                if len(pickup_actions[action.pos]) == 2:
                    action_1, action_2 = pickup_actions[action.pos]
                    self.pickup_gold(
                        robot_1=self.robots[action_1.robot_id],
                        robot_2=self.robots[action_2.robot_id],
                    )
                    del pickup_actions[action.pos]

    def handle_gold_deposit(self):
        processed_robots = set()
        for robot in self.robots:
            if robot.partner_id is None or robot.robot_id in processed_robots:
                continue

            partner = self.robots[robot.partner_id]
            if robot.pos != partner.pos:
                raise RuntimeError(
                    "The robot pair's location are not synced. Something went wrong."
                )

            tile = self.board.get_tile(robot.pos)
            if type(tile) is DepositTile and tile.team == robot.team_info.team:
                robot.partner_id = None
                partner.partner_id = None
                tile.deposit()

                processed_robots.add(robot.robot_id)
                processed_robots.add(partner.robot_id)

    def get_simulation_state(self) -> np.ndarray:
        output = np.ndarray(self.board.tiles.shape, dtype=object)

        for j in range(output.shape[0]):
            for i in range(output.shape[1]):
                pos = (i, j)
                output[j, i] = (
                    self.board.get_robots_at(pos),
                    self.board.get_tile(pos),
                )

        return output

    def collect_messages(self) -> list[tuple[int | None, Message]]:
        output = []
        for robot in self.robots:
            output.extend(robot.outgoing_messages)
            robot.outgoing_messages.clear()
        return output

    def handle_incoming_messages(self, messages: list[tuple[int | None, Message]]):
        for receiver_id, message in messages:
            if receiver_id:
                self.message_handler.direct_message(receiver_id, message)
            else:
                self.message_handler.broadcast(message)

    def handle_outgoing_messages(self):
        for receiver_id, message in self.message_handler.get_messages(self.step_count):
            self.robots[receiver_id].incomming_messages.append(message)
