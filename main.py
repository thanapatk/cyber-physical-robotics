import random

from tqdm import tqdm

from core.board import Board
from core.enums import Direction, TeamEnum
from core.robot import TeamInfo
from core.simulation import SimulationController
from robot import Robot
from utils.visualization import Visualizer

TOTAL_GOLD_COUNT = 40

if __name__ == "__main__":
    board = Board(deposit_pos=((9, 0), (9, 19)), total_gold_count=TOTAL_GOLD_COUNT)

    red_team_info = TeamInfo(team=TeamEnum.RED, deposit_pos=board.red_deposit_pos)
    blue_team_info = TeamInfo(team=TeamEnum.BLUE, deposit_pos=board.blue_deposit_pos)

    robots = [
        Robot(
            robot_id=i,
            pos=board.get_random_tile_pos(),
            team_info=red_team_info if i < 10 else blue_team_info,
            direction=random.choice(list(Direction)),
            board_size=board.board_size,
        )
        for i in range(20)
    ]

    simulation_controller = SimulationController(robots=robots, board=board)

    for i in tqdm(range(10_000)):
        if (
            board.get_deposit_tile(team=TeamEnum.RED).gold_count
            + board.get_deposit_tile(team=TeamEnum.BLUE).gold_count
            == TOTAL_GOLD_COUNT
        ):
            break

        step = simulation_controller.step_count
        state = simulation_controller.get_simulation_state()

        Visualizer.visualize_board(step, state)
        simulation_controller.step()

    print(f"RED SCORE: {board.get_deposit_tile(team=TeamEnum.RED).gold_count}")
    print(f"BLUE SCORE: {board.get_deposit_tile(team=TeamEnum.BLUE).gold_count}")
