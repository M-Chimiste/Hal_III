import hlt
from hlt import constants
from hlt.game_map import *  #Refine later for what you are actually using.
from hlt.positionals import Direction, Position
import logging
import numpy as np
import random #for making a random choice
import time
from model import model
import sys
import os


using_model = True

if using_model is True:
    bot_model = model.AI()


# surrounding_ocean
# Will take a size frame dictated by the user and will return the positionals around the ship
def surrounding_ocean(ship, area, all_ships, all_structures):
    # Ultimate list that will house the data of positionals around a ship.  This is with the ship at origin.  Consider this a "mask."
    # Once we have the area we need to convert back to the map positionals to query status / halite amount.
    map_box = [] # converted "map cell" positions
    for yval in range(-1 * area, area + 1):
        row = []
        for xval in range(-1 * area, area + 1):
            cell = game_map[ship.position + Position(xval, yval)]

            if cell.position in all_structures:
                structure_friend_foe = 1
            else:
                structure_friend_foe = -1
            
            if cell.position in all_ships:
                ship_friend_foe = 1
            else:
                ship_friend_foe = -1
            
            current_halite = round(cell.halite_amount / constants.MAX_HALITE, 3)
            # logic to ensure that the value of halite never is greater than 1
            if current_halite > 1:
                current_halite = 1.000
            
            structures = cell.structure
            if structures is None:
                structures = 0
            else:
                structures = structure_friend_foe
            
            ships = cell.ship
            if ships is None:
                ships = 0
            else:
                ships = round(ship_friend_foe * ships.halite_amount / constants.MAX_HALITE, 3)
                if ships > 1:
                    ships = 1.000
            
            map_values = (current_halite, ships, structures)
            row.append(map_values)
        map_box.append(row)
            
    return map_box

game = hlt.Game()
""" <<<Game Begin>>> """
game.ready("HAL_III")  #Point of no return

# Query the game map size to determine the turns
map_turns_by_size = { 64: 500, 56: 475, 48: 450, 40: 425, 32: 400}

SAVE_THRESHOLD = 4700 # Amount of Halite needed to be collected in order to save the training data
TOTAL_TURNS = 50 # Total number of turns to ensure that signal to noise is high
MAX_SHIPS = 1 # Total number of ships.  Limited to allow for micro actions
AREA = 16 # Ship box distance


training_data = [] # Empty array that will house the training data.


while True:
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map
    all_ships = me.get_ships
    # Commands for each ship
    command_queue = []
    directions = [Direction.North, Direction.South, Direction.East, Direction.West, Direction.Still]
    # list of all friendly structure positions
    all_structures = [drop_off.position for drop_off in list(me.get_dropoffs()) + [me.shipyard]]
    # lsit of all friendly ship positions
    all_ships = [ships.position for ships in list(me.get_ships())]

    #position_choices = []

    if len(me.get_ships()) < MAX_SHIPS:
        if me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
            command_queue.append(me.shipyard.spawn())
    for ship in me.get_ships():
        
        ship_surroundings = surrounding_ocean(ship, AREA, all_ships, all_structures)
        if using_model is False:
            choice = random.choice(range(len(directions)))  # Random choice for a direction to move
            training_data.append([ship_surroundings, choice]) # Appending the movement and area information
        else:
            inputs = np.array(ship_surroundings).reshape(-1, 33, 33, 3)
            choice = bot_model.prediction(inputs)
            logging.info(f"Prediction is {choice}")

        
        command_queue.append(ship.move(directions[choice])) # Add ship commands to the queue for execution.
    
    if game.turn_number == TOTAL_TURNS:

        if me.halite_amount >= SAVE_THRESHOLD:
            np.save(f"training_data/{me.halite_amount}-{int(time.time()*1000)}.npy", training_data)
        

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

