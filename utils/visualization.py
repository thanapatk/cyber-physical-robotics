import os
import sys
from typing import Sequence

import numpy as np

from core.board import BaseTile, DepositTile
from core.enums import TeamEnum
from core.robot import Robot


class ColoredText:
    @staticmethod
    def red(s):
        return "\033[91m{}\033[00m".format(s)

    @staticmethod
    def yellow(s):
        return "\033[93m{}\033[00m".format(s)

    @staticmethod
    def cyan(s):
        return "\033[96m{}\033[00m".format(s)

    @staticmethod
    def grey(s):
        return "\033[97m{}\033[00m".format(s)


class Visualizer:
    @staticmethod
    def clear_console():
        if os.name == "nt":
            os.system("cls")
        else:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()

    @staticmethod
    def direction_string(robot: Robot):
        direction = robot.direction
        return "▲▼▶◀"[direction.value]

    @staticmethod
    def cell(
        red_robots: Sequence[Robot], blue_robots: Sequence[Robot], tile: BaseTile
    ) -> tuple[str, str]:
        red = "".join(map(Visualizer.direction_string, red_robots))
        blue = "".join(map(Visualizer.direction_string, blue_robots))

        line1 = (
            ColoredText.red(f"{red if len(red) <= 3 else '---':^3}")
            + ColoredText.grey(":")
            + ColoredText.cyan(f"{blue if len(blue) <= 3 else '---':^3}")
        )
        line2 = f"{tile.gold_count:^7}"

        if type(tile) is DepositTile:
            line2 = (
                ColoredText.red(line2)
                if tile.team == TeamEnum.RED
                else ColoredText.cyan(line2)
            )
        else:
            line2 = (
                ColoredText.yellow(line2)
                if tile.gold_count != 0
                else ColoredText.grey(line2)
            )

        return line1, line2

    @staticmethod
    def visualize_board(
        simulation_step: int,
        board_status: np.ndarray,
    ):
        separator = "+" + "+".join(["-" * 7] * board_status.shape[1]) + "+"

        Visualizer.clear_console()

        print(f"Simulation step {simulation_step}")
        print(separator)
        for j in range(board_status.shape[0]):
            line_1 = []
            line_2 = []
            for i in range(board_status.shape[0]):
                robots, tile = board_status[j, i]

                red_robots = []
                blue_robots = []

                for robot in robots:
                    if robot.team_info.team == TeamEnum.RED:
                        red_robots.append(robot)
                    else:
                        blue_robots.append(robot)

                cell_line_1, cell_line_2 = Visualizer.cell(
                    red_robots=red_robots,
                    blue_robots=blue_robots,
                    tile=tile,
                )

                line_1.append(cell_line_1)
                line_2.append(cell_line_2)

            print(f"|{'|'.join(line_1)}|")
            print(f"|{'|'.join(line_2)}|")
        print(separator)
