from paxos.message import *
from utils.distance import manhattan_distance


class PaxosHandler:
    PROPOSAL_TIMEOUT = 75

    def __init__(self, robot_id: int, team_size: int = 10) -> None:
        self.robot_id = robot_id
        self.team_size = team_size
        self.proposal_counter = 0

        self.reset_acceptor_state()
        self.reset_proposer_state()

    def reset_acceptor_state(self):
        self.promised_id: tuple | None = None
        self.accepted_id: tuple | None = None
        self.accepted_value: FullMission | None = None

    def reset_proposer_state(self):
        self.is_proposing = False
        self.proposal_id: tuple | None = None
        self.proposed_mission: Mission | None = None
        self.promises_recieved: list[PrepareResponse] = []
        self.acceptance_tally: dict = {}
        self.consensus_reached = False
        self.final_value = None

    def _get_next_proposal_id(self) -> tuple:
        self.proposal_counter += 1
        return (self.proposal_counter, self.robot_id)

    def start_election(self, mission: Mission, step: int) -> PrepareRequest:
        self.reset_proposer_state()
        self.is_proposing = True
        self.proposal_id = self._get_next_proposal_id()
        self.proposed_mission = mission

        return PrepareRequest(
            sender_id=self.robot_id,
            step=step,
            paxos_id=self.proposal_id,
            value=mission,
        )

    def handle_promise_response(
        self, message: PrepareResponse, step: int
    ) -> AcceptRequest | None:
        if not self.is_proposing or message.paxos_id != self.proposal_id:
            return None

        self.promises_recieved.append(message)

        if len(self.promises_recieved) <= self.team_size / 2:
            return None

        self.is_proposing = False

        value_to_propose = self.proposed_mission
        best_promise = max(
            self.promises_recieved, key=lambda p: p.paxos_id or (-1,), default=None
        )
        if best_promise and best_promise.value:
            value_to_propose = best_promise.value

        best_follower = min(self.promises_recieved, key=lambda p: p.follower_bid)

        if not value_to_propose or self.proposal_id is None:
            return None

        full_mission = FullMission(
            cost=value_to_propose.cost,
            target_tile=value_to_propose.target_tile,
            leader_id=self.robot_id,
            follower_id=best_follower.sender_id,
        )

        return AcceptRequest(
            sender_id=self.robot_id,
            step=step,
            paxos_id=self.proposal_id,
            value=full_mission,
        )

    def handle_prepare_request(
        self, message: PrepareRequest, step: int, current_tile: tuple[int, int]
    ) -> tuple[int, PrepareResponse] | None:
        if self.promised_id is not None and message.paxos_id < self.promised_id:
            return None

        if self.is_proposing and message.paxos_id > self.proposal_id:
            self.reset_proposer_state()

        self.promised_id = message.paxos_id
        cost = manhattan_distance(current_tile, message.value.target_tile)

        return message.sender_id, PrepareResponse(
            sender_id=self.robot_id,
            step=step,
            paxos_id=message.paxos_id,
            value=self.accepted_value,
            follower_bid=cost,
        )

    def handle_accept_request(
        self, message: AcceptRequest, step: int
    ) -> AcceptResponse | None:
        if self.promised_id is not None and message.paxos_id < self.promised_id:
            return None

        self.promised_id = message.paxos_id
        self.accepted_id = message.paxos_id
        self.accepted_value = message.value

        return AcceptResponse(
            sender_id=self.robot_id,
            step=step,
            paxos_id=self.accepted_id,
            value=self.accepted_value,
        )

    def handle_accept_response(self, message: AcceptResponse) -> None:
        if self.accepted_id is None:
            return

        key = tuple(message.value.model_dump().items())
        self.acceptance_tally[key] = self.acceptance_tally.get(key, 0) + 1

        if (
            not self.consensus_reached
            and self.acceptance_tally[key] > self.team_size / 2
        ):
            self.consensus_reached = True
            self.final_value = message.value

    def did_proposal_fail(self, current_step: int, proposal_start_step: int) -> bool:
        return (
            self.is_proposing
            and (current_step - proposal_start_step) > self.PROPOSAL_TIMEOUT
        )
