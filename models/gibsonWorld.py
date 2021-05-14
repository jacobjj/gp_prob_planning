'''A script to load a test environment for testing GP models.
'''
import numpy as np
import GPy
from os import path as osp
from scipy import stats
import pickle

import time

folder = '/root/prob_planning/assets/Allensville/{}.urdf'
# folder = '/home/jacoblab/prob_planning/assets/Allensville/{}.urdf'
# Load environment
items = [
    'coffee_table',
    'side_table',
    'kitchen_counter', 
    'chair1', 
    'chair2',
    'outerWall2',
    'outerWall3',
    'outerWall4',
    'kitchen2',
    ]

def set_obstacles(client_obj):
    '''
    Set up the obstacles in the given client.
    :param clinet_obj:
    :returns list: A list of obstacles
    '''
    obstacles = [client_obj.loadURDF(folder.format(i)) for i in items]
    return obstacles
