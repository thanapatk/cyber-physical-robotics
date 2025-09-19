from pydantic import BaseModel

from core.enums import Direction


class Action(BaseModel):
    robot_id: int


class TurnAction(Action):
    new_direction: Direction


class MoveAction(Action):
    pass


class PickupAction(Action):
    pos: tuple[int, int]
