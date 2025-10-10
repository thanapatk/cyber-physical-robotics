import random
from collections import deque
from enum import Enum, auto
from typing import Deque

import numpy as np
from pydantic import BaseModel

from core.actions import Action, MoveAction, PickupAction, TurnAction, WaitAction
from core.enums import Direction
from core.message_handler import Message
from core.robot import BaseRobot, Observation, TeamInfo
from paxos import *
from utils.distance import manhattan_distance


class RobotState(Enum):
    EXPLORING = auto()
    PROPOSING = auto()
    EXECUTING = auto()
    AWAITING_PARTNER = auto()
    DELIVERING = auto()


class SensedTile(BaseModel):
    step: int
    gold_count: int
    same_team_count: int


class ObservationsMessage(Message):
    value: list[Observation]


class TurnMessage(Message):
    value: Direction


class MissionAbortMessage(Message):
    value: None = None


class MissionCompleteMessage(Message):
    value: None = None


class GoldConsumedMessage(Message):
    """Broadcast when gold is picked up from a tile."""

    value: tuple[int, int]  # position where gold was consumed


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

        self.paxos_handler = PaxosHandler(self.robot_id, team_size=10)
        self.current_mission: FullMission | None = None
        self.agreed_direction: Direction | None = None

        self.timeout_counter: int | None = None

        self.proposal_start_step: int | None = None

        self.failed_proposal_count: int = 0
        self.backoff_until_step: int = 0

        # --- DIAGNOSTICS ---
        self.paxos_debug_log: list[str] = []

    def log_paxos(self, step: int, msg: str):
        """Log Paxos events for debugging."""
        self.paxos_debug_log.append(f"Step {step} | Robot {self.robot_id}: {msg}")
        # if len(self.paxos_debug_log) > 100:  # Keep last 100 entries
        #     self.paxos_debug_log.pop(0)

    def print_paxos_log(self):
        """Print the Paxos debug log."""
        if self.paxos_debug_log:
            print(f"\n=== PAXOS LOG FOR ROBOT {self.robot_id} ===")
            for entry in self.paxos_debug_log:  # [-50:]:  # Last 20 entries
                print(entry)

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
        else:  # WEST
            turn_cost[y, : x + 1] = 0
            x_indices = np.where(np.arange(turn_cost.shape[1]) > x)
            y_indices = np.where(np.arange(turn_cost.shape[0]) != y)

        turn_cost[np.ix_(*y_indices, *x_indices)] = 2
        return manhattan_distance + turn_cost

    def decide_exploration_target(self, step: int) -> tuple[int, int]:
        cost_matrix = self.generate_cost_matrix()

        coldness_map = np.full(self.board_size, step, dtype=float)
        gold_bonus_map = np.zeros(self.board_size, dtype=float)
        robot_density_map = np.zeros(self.board_size, dtype=float)
        for pos, sensed_tile in self.sensed_map.items():
            coldness_map[pos[1], pos[0]] = step - sensed_tile.step
            gold_bonus_map[pos[1], pos[0]] = sensed_tile.gold_count
            robot_density_map[pos[1], pos[0]] = sensed_tile.same_team_count

        COLDNESS_WEIGHT = 5
        DISTANCE_WEIGHT = 50
        GOLD_BONUS_WEIGHT = 10
        ROBOT_REPULSION_WEIGHT = 150

        exploration_score_map = (
            (coldness_map * COLDNESS_WEIGHT)
            + gold_bonus_map * GOLD_BONUS_WEIGHT
            - cost_matrix * DISTANCE_WEIGHT
            - robot_density_map * ROBOT_REPULSION_WEIGHT
        )

        exploration_score_map[self.pos[1], self.pos[0]] = -np.inf

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
        sensed_tile = self.sensed_map.get(
            pos, SensedTile(step=-1, gold_count=0, same_team_count=0)
        )

        if sensed_tile.step > step:
            return

        sensed_tile.step = step
        sensed_tile.gold_count = observation.gold_count
        sensed_tile.same_team_count = observation.same_team_robot_count
        self.sensed_map[pos] = sensed_tile

    def find_best_mission_from_map(self, step: int) -> Mission | None:
        best_mission = None
        lowest_total_cost = 999

        for pos, sensed_tile in self.sensed_map.items():
            if sensed_tile.gold_count == 0 or step - sensed_tile.step > 100:
                continue

            cost_robot_to_gold = manhattan_distance(self.pos, pos)
            cost_gold_to_deposit = manhattan_distance(pos, self.team_info.deposit_pos)
            total_mission_cost = cost_robot_to_gold + cost_gold_to_deposit

            if total_mission_cost < lowest_total_cost:
                lowest_total_cost = total_mission_cost
                best_mission = Mission(cost=cost_gold_to_deposit, target_tile=pos)

        return best_mission

    def process_messages(self, step: int):
        while len(self.incomming_messages) != 0:
            message = self.incomming_messages.popleft()

            if type(message) is ObservationsMessage:
                for observation in message.value:
                    self._update_sensed_tile(message.step, observation)
            elif type(message) is GoldConsumedMessage:
                pos = message.value
                if pos in self.sensed_map:
                    self.sensed_map[pos].gold_count -= 1
                    self.sensed_map[pos].step = message.step
            elif type(message) is PrepareRequest:
                # --- DIAGNOSTIC ---
                self.log_paxos(
                    step,
                    f"Received PrepareRequest from {message.sender_id} with ID {message.paxos_id}",
                )

                prepare_response = self.paxos_handler.handle_prepare_request(
                    message=message, step=step, current_tile=self.pos
                )
                if prepare_response:
                    sender_id, response = prepare_response
                    self.log_paxos(step, f"Sending PrepareResponse to {sender_id}")
                    self.outgoing_messages.append((sender_id, response))
            elif type(message) is PrepareResponse:
                # --- FIX: Only count promises for the current active proposal ---
                if (
                    not self.paxos_handler.is_proposing
                    or message.paxos_id != self.paxos_handler.proposal_id
                ):
                    self.log_paxos(
                        step,
                        f"Ignoring PrepareResponse: not proposing or wrong proposal ID",
                    )
                    continue

                # --- DIAGNOSTIC ---
                self.log_paxos(
                    step,
                    f"Received PrepareResponse from {message.sender_id}, now have {len(self.paxos_handler.promises_recieved) + 1} promises",
                )

                accept_request = self.paxos_handler.handle_promise_response(
                    message=message, step=step
                )
                if accept_request:
                    self.log_paxos(step, "Have majority! Sending AcceptRequest")
                    self.outgoing_messages.append((None, accept_request))
            elif type(message) is AcceptRequest:
                # --- DIAGNOSTIC ---
                self.log_paxos(step, f"Received AcceptRequest from {message.sender_id}")

                accept_response = self.paxos_handler.handle_accept_request(
                    message=message, step=step
                )
                if accept_response:
                    self.log_paxos(step, "Accepting! Sending AcceptResponse")
                    self.outgoing_messages.append((None, accept_response))
            elif type(message) is AcceptResponse:
                self.log_paxos(
                    step, f"Received AcceptResponse from {message.sender_id}"
                )
                self.paxos_handler.handle_accept_response(message=message)
                # --- DIAGNOSTIC: Log acceptance tally ---
                if self.paxos_handler.acceptance_tally:
                    total_accepts = sum(self.paxos_handler.acceptance_tally.values())
                    self.log_paxos(
                        step,
                        f"Total accepts: {total_accepts}, consensus: {self.paxos_handler.consensus_reached}",
                    )
            elif type(message) is TurnMessage:
                self.agreed_direction = message.value
            elif (
                type(message) is MissionCompleteMessage
                or type(message) is MissionAbortMessage
            ):
                self.current_mission = None
                self.state = RobotState.EXPLORING
                self.saved_actions.clear()
                self.agreed_direction = None
                self.paxos_handler.reset_proposer_state()
                self.paxos_handler.reset_acceptor_state()

    def decide_action(self, step: int, observations: list[Observation]) -> Action:
        self.process_messages(step=step)

        self.observations = observations

        for observation in observations:
            self._update_sensed_tile(step, observation)

        self.outgoing_messages.append(
            (
                None,
                ObservationsMessage(
                    sender_id=self.robot_id, step=step, value=observations
                ),
            )
        )

        if self.paxos_handler.consensus_reached:
            newly_decided_mission = self.paxos_handler.final_value

            is_committed = self.state in {
                RobotState.AWAITING_PARTNER,
                RobotState.DELIVERING,
            }

            if not is_committed:
                if newly_decided_mission and self.robot_id in [
                    newly_decided_mission.leader_id,
                    newly_decided_mission.follower_id,
                ]:
                    if self.current_mission != newly_decided_mission:
                        self.state = RobotState.EXECUTING
                        self.saved_actions.clear()
                    self.current_mission = newly_decided_mission

                else:
                    self.current_mission = None
                    if self.state == RobotState.EXECUTING:
                        self.state = RobotState.EXPLORING

            if self.paxos_handler.is_proposing:
                self.paxos_handler.reset_proposer_state()

            self.paxos_handler.consensus_reached = False
            self.paxos_handler.final_value = None
            self.proposal_start_step = None
        if self.state == RobotState.PROPOSING:
            if (
                self.proposal_start_step is not None
                and self.paxos_handler.did_proposal_fail(step, self.proposal_start_step)
            ):
                self.log_paxos(
                    step,
                    f"Proposal TIMED OUT. Had {len(self.paxos_handler.promises_recieved)} promises (need 6+)",
                )
                self.failed_proposal_count += 1
                backoff_time = min(2**self.failed_proposal_count, 50)
                self.backoff_until_step = step + backoff_time
                self.log_paxos(step, f"Entering backoff for {backoff_time} steps")

                self.state = RobotState.EXPLORING
                self.paxos_handler.reset_proposer_state()
                self.proposal_start_step = None
            elif not self.paxos_handler.is_proposing:
                self.log_paxos(step, "Proposal ABANDONED (higher proposal seen)")
                self.failed_proposal_count += 1
                backoff_time = min(2 ** (self.failed_proposal_count - 1), 30)
                self.backoff_until_step = step + backoff_time

                self.state = RobotState.EXPLORING
                self.proposal_start_step = None

        if self.state == RobotState.EXPLORING:
            best_local_mission = self.find_best_mission_from_map(step)
            if (
                best_local_mission
                and not self.current_mission
                and step >= self.backoff_until_step
            ):
                self.log_paxos(
                    step,
                    f"Starting election for mission at {best_local_mission.target_tile}",
                )
                prepare_request = self.paxos_handler.start_election(
                    mission=best_local_mission, step=step
                )
                self.outgoing_messages.append((None, prepare_request))

                self.proposal_start_step = step

                self.saved_actions.clear()
                self.saved_actions.extend(self.pathfind(best_local_mission.target_tile))

                self.state = RobotState.PROPOSING
            else:
                target = self.decide_exploration_target(step)

                if target != self.target_pos:
                    self.target_pos = target
                    self.saved_actions.clear()
                    self.saved_actions.extend(self.pathfind(self.target_pos))

            if len(self.saved_actions) != 0:
                return self.saved_actions.popleft()
            return WaitAction(robot_id=self.robot_id)

        elif self.state == RobotState.PROPOSING:
            if len(self.saved_actions) != 0:
                return self.saved_actions.popleft()
            else:
                return WaitAction(robot_id=self.robot_id)

        elif self.state == RobotState.EXECUTING:
            if not self.saved_actions and self.current_mission:
                self.saved_actions.extend(
                    self.pathfind(self.current_mission.target_tile)
                )

            if self.saved_actions:
                return self.saved_actions.popleft()
            else:
                self.state = RobotState.AWAITING_PARTNER
                return WaitAction(robot_id=self.robot_id)

        elif self.state == RobotState.AWAITING_PARTNER:
            if self.timeout_counter is None:
                self.timeout_counter = 500
            elif self.timeout_counter == 0:
                self.outgoing_messages.append(
                    (None, MissionAbortMessage(sender_id=self.robot_id, step=step))
                )
                self.state = RobotState.EXPLORING
            else:
                self.timeout_counter -= 1

            if self.partner_id:
                self.agreed_direction = None
                self.sensed_map[self.pos].gold_count -= 1

                self.outgoing_messages.append(
                    (
                        None,
                        GoldConsumedMessage(
                            sender_id=self.robot_id, step=step, value=self.pos
                        ),
                    )
                )

                if self.robot_id == self.current_mission.leader_id:
                    self.outgoing_messages.append(
                        (
                            None,
                            ObservationsMessage(
                                sender_id=self.robot_id,
                                step=step,
                                value=[
                                    Observation(
                                        pos=self.pos,
                                        gold_count=self.sensed_map[self.pos].gold_count,
                                        same_team_robot_count=2,
                                    )
                                ],
                            ),
                        )
                    )

                self.state = RobotState.DELIVERING
                self.saved_actions.extend(self.pathfind(self.team_info.deposit_pos))
                return self.saved_actions.popleft()

            if not self.current_mission:
                self.state = RobotState.EXPLORING
                return WaitAction(robot_id=self.robot_id)

            if not self.agreed_direction:
                if self.current_mission.leader_id == self.robot_id:
                    path_to_deposit = self.pathfind(self.team_info.deposit_pos)

                    self.agreed_direction = self.direction
                    for action in path_to_deposit:
                        if type(action) is TurnAction:
                            self.agreed_direction = action.new_direction
                            break

                    if self.agreed_direction != self.direction:
                        self.outgoing_messages.append(
                            (
                                self.current_mission.follower_id,
                                TurnMessage(
                                    sender_id=self.robot_id,
                                    step=step,
                                    value=self.agreed_direction,
                                ),
                            )
                        )
                elif self.current_mission.follower_id == self.robot_id:
                    return WaitAction(robot_id=self.robot_id)

            if self.agreed_direction and self.agreed_direction != self.direction:
                return TurnAction(
                    robot_id=self.robot_id, new_direction=self.agreed_direction
                )
            else:
                return PickupAction(robot_id=self.robot_id, pos=self.pos)

        elif self.state == RobotState.DELIVERING:
            if not self.partner_id and self.pos != self.team_info.deposit_pos:
                self.outgoing_messages.append(
                    (None, MissionAbortMessage(sender_id=self.robot_id, step=step))
                )

                self.saved_actions.clear()
                self.state = RobotState.EXPLORING
                return WaitAction(robot_id=self.robot_id)
            elif not self.partner_id and self.pos == self.team_info.deposit_pos:
                if (
                    self.current_mission
                    and self.current_mission.leader_id == self.robot_id
                ):
                    self.outgoing_messages.append(
                        (
                            None,
                            MissionCompleteMessage(sender_id=self.robot_id, step=step),
                        )
                    )

                self.state = RobotState.EXPLORING
                self.paxos_handler.reset_proposer_state()
                return WaitAction(robot_id=self.robot_id)

            if len(self.saved_actions) != 0:
                return self.saved_actions.popleft()
            return WaitAction(robot_id=self.robot_id)

        return WaitAction(robot_id=self.robot_id)
