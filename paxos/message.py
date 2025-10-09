from pydantic import BaseModel

from core.message_handler import Message


class Mission(BaseModel):
    cost: int
    target_tile: tuple[int, int]


class FullMission(Mission):
    leader_id: int
    follower_id: int


class PrepareRequest(Message):
    paxos_id: tuple[int, int]
    value: Mission


class PrepareResponse(Message):
    paxos_id: tuple[int, int] | None
    value: FullMission | None
    follower_bid: int


class AcceptRequest(Message):
    paxos_id: tuple[int, int]
    value: FullMission


class AcceptResponse(Message):
    paxos_id: tuple[int, int]
    value: FullMission
