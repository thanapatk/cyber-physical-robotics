import random
from collections import defaultdict
from typing import Any, Sequence

from pydantic import BaseModel


class Message(BaseModel):
    sender_id: int
    step: int
    value: Any


class MessageItem(BaseModel):
    sender_id: int
    receiver_id: int | None  # If None -> Broadcast Message Type
    message: Message


class MessageHandler:
    def __init__(self) -> None:
        self.messages: dict[int, list[MessageItem]] = defaultdict(list)

    @staticmethod
    def _get_random_delay() -> int:
        return 1
        # return random.randint(
        #     1, 3
        # )  # TODO: uncomment this line when what to test random delay

    def _add_message(self, message_item: MessageItem):
        self.messages[message_item.message.step + self._get_random_delay()].append(
            message_item
        )

    def broadcast(self, message: Message):
        self._add_message(
            MessageItem(sender_id=message.sender_id, receiver_id=None, message=message)
        )

    def direct_message(self, receiver_id: int, message: Message):
        self._add_message(
            MessageItem(
                sender_id=message.sender_id, receiver_id=receiver_id, message=message
            )
        )

    def get_messages(
        self,
        current_step: int,
        red_team_id: Sequence[int] = range(10),
        blue_team_id: Sequence[int] = range(10, 20),
    ) -> list[tuple[int, Message]]:
        if current_step not in self.messages:
            return []

        output = list()
        for message_item in self.messages[current_step]:
            if message_item.receiver_id is None:
                output.extend(
                    [
                        (i, message_item.message)
                        for i in (
                            red_team_id
                            if (message_item.sender_id in red_team_id)
                            else blue_team_id
                        )
                    ]
                )
            else:
                output.append((message_item.receiver_id, message_item.message))

        del self.messages[current_step]

        # TODO: implement message drop

        return output
