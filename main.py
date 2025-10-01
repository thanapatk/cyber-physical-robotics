import random
import time

import numpy as np

from core.board import Board
from core.enums import Direction, TeamEnum
from core.robot import TeamInfo
from core.simulation import SimulationController
from robot import Robot
from utils.visualization import Visualizer

TARGET_FPS = 100

if __name__ == "__main__":
    board = Board(deposit_pos=((9, 0), (9, 19)), total_gold_count=40)

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
    step_times = []

    robots_set = set(robots)

    for i in range(1_000):
        start_loop_time = time.perf_counter()
        step = simulation_controller.step_count
        state = simulation_controller.get_simulation_state()

        Visualizer.visualize_board(step, state)

        # for robot in robots:
        #     print(f"{robot.robot_id}: {len(robot.sensed_map)}")
        # print(
        #     np.int64(
        #         robots[i % len(robots)].get_coldness_map(i) * 1.5
        #         + robots[i % len(robots)].generate_cost_matrix()
        #     )
        # )

        start = time.perf_counter_ns()
        simulation_controller.step()
        step_times.append(time.perf_counter_ns() - start)

        time.sleep(max(0, 1 / TARGET_FPS - (time.perf_counter() - start_loop_time)))

    print(f"Mean step time: {np.mean(step_times) * 1e-9/1e-3} ms")
    print(f"Median step time: {np.median(step_times) * 1e-9/1e-3} ms")
    print(f"Std step time: {np.std(step_times) * 1e-9/1e-3} ms")
