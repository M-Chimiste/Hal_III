import os
import random

MAX_TURNS = 50

map_turns_by_size = { 64: 500, 56: 475, 48: 450, 40: 425, 32: 400}

while True:
    map_choice = random.choice(list(map_turns_by_size.keys()))
    commands = [f'halite.exe --replay-directory replays/ --turn-limit {MAX_TURNS} -vvv --width {map_choice} --height {map_choice} "python HAL_III.py" "python HAL_III-2.py"',
                f'halite.exe --replay-directory replays/ --turn-limit {MAX_TURNS} -vvv --width {map_choice} --height {map_choice} "python HAL_III.py" "python HAL_III-2.py" "python HAL_III-2.py" "python HAL_III-2.py"']

    command = random.choice(commands)
    os.system(command)