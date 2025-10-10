import logging
from core.enums import TeamEnum


class BaseTile:
    def __init__(self, gold_count: int = 0) -> None:
        self.gold_count = gold_count

    def take(self):
        if self.gold_count == 0:
            raise ValueError("Attempted to take gold from an empty tile.")

        self.gold_count -= 1

    def add(self):
        self.gold_count += 1


class DepositTile(BaseTile):
    def __init__(self, team: TeamEnum) -> None:
        super().__init__()
        self.team = team
        self.logger = logging.getLogger(__name__)

    def take(self):
        raise NotImplementedError("Cannot take gold from a deposit tile")

    def add(self):
        raise NotImplementedError("Cannot add/drop gold on a deposit tile")

    def deposit(self):
        self.gold_count += 1

        self.logger.info(
            f"Team {self.team.value} deposited! Current gold count: {self.gold_count}"
        )
