# Parsing replay files.  Adopted from https://github.com/HaliteChallenge/Halite-III/blob/master/starter_kits/ml/SVM/parse.py
import copy
import json
import os
import os.path
import zstd
import numpy as np
import time

import hlt
from hlt.entity import Shipyard, Dropoff, Ship
from hlt.game_map import MapCell, GameMap
from hlt.constants import *

ARBITRARY_ID = -1

def find_player_data(data):
    rank = [p for p in data['game_statistics']['player_statistics']]
    for i in rank:
        if i['rank'] == 1:
            player_id = i['player_id']
    player_name = [p for p in data['players'] if p['player_id'] == player_id]
    player_name = player_name[0]['name']
    return int(player_id)
    


def parse_replay_file(file_name):
    print("Load Replay: " + file_name)
    with open(file_name, 'rb') as f:
        data = json.loads(zstd.loads(f.read()))

    print("Load Basic Information")
    player_id = find_player_data(data)
    player = [p for p in data['players'] if p['player_id'] == player_id]
    player = player[0]
    
    my_shipyard = Shipyard(player_id, ARBITRARY_ID,
                               hlt.Position(player['factory_location']['x'], player['factory_location']['y']))
    other_shipyards = [
        Shipyard(p['player_id'], ARBITRARY_ID, hlt.Position(p['factory_location']['x'], p['factory_location']['y']))
        for p in data['players'] if int(p['player_id']) != player_id]
    width = data['production_map']['width']
    height = data['production_map']['height']

    print("Load Cell Information")
    first_cells = []
    for x in range(len(data['production_map']['grid'])):
        row = []
        for y in range(len(data['production_map']['grid'][x])):
            row += [MapCell(hlt.Position(x, y), data['production_map']['grid'][x][y]['energy'])]
        first_cells.append(row)
    frames = []
    for f in data['full_frames']:
        prev_cells = first_cells if len(frames) == 0 else frames[-1]._cells
        new_cells = copy.deepcopy(prev_cells)
        for c in f['cells']:
            new_cells[c['y']][c['x']].halite_amount = c['production']
        frames.append(GameMap(new_cells, width, height))

    print("Load Player Ships")
    moves = [{} if str(player_id) not in f['moves'] else {m['id']: m['direction'] for m in f['moves'][str(player_id)] if
                                                          m['type'] == "m"} for f in data['full_frames']]
    ships = [{} if str(player_id) not in f['entities'] else {
        int(sid): Ship(player_id, int(sid), hlt.Position(ship['x'], ship['y']), ship['energy']) for sid, ship in
        f['entities'][str(player_id)].items()} for f in data['full_frames']]

    print("Load Other Player Ships")
    other_ships = [
        {int(sid): Ship(int(pid), int(sid), hlt.Position(ship['x'], ship['y']), ship['energy']) for pid, p in
         f['entities'].items() if
         int(pid) != player_id for sid, ship in p.items()} for f in data['full_frames']]

    print("Load Droppoff Information")
    first_my_dropoffs = [my_shipyard]
    first_them_dropoffs = other_shipyards
    my_dropoffs = []
    them_dropoffs = []
    for f in data['full_frames']:
        new_my_dropoffs = copy.deepcopy(first_my_dropoffs if len(my_dropoffs) == 0 else my_dropoffs[-1])
        new_them_dropoffs = copy.deepcopy(first_them_dropoffs if len(them_dropoffs) == 0 else them_dropoffs[-1])
        for e in f['events']:
            if e['type'] == 'construct':
                if int(e['owner_id']) == player_id:
                    new_my_dropoffs.append(
                        Dropoff(player_id, ARBITRARY_ID, hlt.Position(e['location']['x'], e['location']['y'])))
                else:
                    new_them_dropoffs.append(
                        Dropoff(e['owner_id'], ARBITRARY_ID, hlt.Position(e['location']['x'], e['location']['y'])))
        my_dropoffs.append(new_my_dropoffs)
        them_dropoffs.append(new_them_dropoffs)
    return list(zip(frames, moves, ships, other_ships, my_dropoffs, them_dropoffs))


def parse_replay_folder(folder_name, max_files=None):
    replay_buffer = []
    for file_name in sorted(os.listdir(folder_name)):
        if not file_name.endswith(".hlt"):
            continue
        elif max_files is not None and len(replay_buffer) >= max_files:
            break
        else:
            replay_buffer.append(parse_replay_file(os.path.join(folder_name, file_name)))
    return replay_buffer


def surrounding_ocean(ship, data_frame, area):
    # Ultimate list that will house the data of positionals around a ship.  This is with the ship at origin.  Consider this a "mask."
    # Once we have the area we need to convert back to the map positionals to query status / halite amount.
    gameMap = data_frame[0]
    ship_keys = list(data_frame[2].keys())
    friendly_ships = data_frame[2]
    structure_keys = list(data_frame[4].keys())
    friendly_structures = list(data_frame[4])
    enemy_ship_keys = list(data_frame[3].keys())
    enemy_ship_data = data_frame[3]
    enemy_struct_keys = list(data_frame[5].keys())
    enemy_struc_data = data_frame[5]
    
    enemy_ships = []
    for i in enemy_ship_keys:
        enemy_ships.append(enemy_ship_data[i])
    enemy_structures = []
    for i in enemy_struct_keys:
        enemy_structures.append(enemy_struc_data[i])
    
    all_ships = []
    for i in ship_keys:
        all_ships.append(friendly_ships[1].position)
    
    all_structures = []
    for i in structure_keys:
        all_structures.append(friendly_structures[i].position)
    
    map_box = [] # converted "map cell" positions
    for yval in range(-1 * area, area + 1):
        row = []
        for xval in range(-1 * area, area + 1):
            cell = gameMap[hlt.Position(ship.position.y,ship.position.x) + hlt.Position(xval, yval)]

            if cell.position in all_structures:
                structure_friend_foe = 1
            elif cell.position in enemy_structures:
                structure_friend_foe = -1
            else:
                structure_friend_foe = 0
            
            if cell.position in all_ships:
                ship_friend_foe = 1
                friendly_ship = all_ships.index(cell.position)
                friendly_ship = ship_keys[friendly_ship]
                ship_halite = friendly_ships[friendly_ship].halite_amount
                ships = round(ship_friend_foe * ship_halite / MAX_HALITE, 3)
                if ships > 1.000:
                    ships = 1.000
            elif cell.position in enemy_ships:
                ship_friend_foe = -1
                enemy_ship = enemy_ships.index(cell.position)
                enemy_ship = enemy_ship_keys[enemy_ship]
                ship_halite = enemy_ship_data[enemy_ship].halite_amount
                ships = round(ship_friend_foe * ship_halite / MAX_HALITE, 3)
                if ships > 1.000:
                    ships = 1.000
            else:
                ship_friend_foe = 0
                ships = 0
            
            current_halite = round(cell.halite_amount / MAX_HALITE, 3)
            # logic to ensure that the value of halite never is greater than 1
            if current_halite > 1:
                current_halite = 1.000
            
            map_values = (current_halite, ships, structure_friend_foe)
            row.append(map_values)
        map_box.append(row)
            
    return map_box


def convert_moves(ship, ships):
    ship_move = ships[ship]
    try:
        if ship_move == 'n':
            ship_move = 0
        elif ship_move == 's':
            ship_move = 1
        elif ship_move == 'e':
            ship_move = 2
        elif ship_move == 'w':
            ship_move = 3
        elif ship_move == 'o':
            ship_move = 4
    except KeyError:
        ship_move = 4
    return ship_move

if __name__ == "__main__":

    folder_name = "replay_data"
    
    for file_name in sorted(os.listdir(folder_name)):
        training_data = []
        data = parse_replay_file(os.path.join(folder_name, file_name))

        for d in data:
            ships = d[1]
            if ships is not {}:
                ship_ids = list(ships.keys())
                for s in ship_ids:
                    map_box = surrounding_ocean
                    move = convert_moves(s, ships)
                    training_data.append(map_box, move)
                    
        np.save(f"parsed_data/{int(time.time()*1000)}", training_data)