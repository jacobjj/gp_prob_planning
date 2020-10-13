''' A code to generate multiple RRT paths for the environment.
'''

import numpy as np
import pybullet as p
from scipy import stats, optimize

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    print("Could not f  ind OMPL")
    raise ImportError("Run inside docker!!")

class check_motion(ob.MotionValidator):
    '''A motion validator check using chance constrained.
    '''
    def __init__(self, spaceInformation):
        '''
        Initialize the motion validator.
        :param spaceInformation: A ob.SpaceInformation object.
        '''
        super(check_motion, self).__init__(spaceInformation)

    def checkMotion(self, start, state2):
        '''
        Check if there exists a valid motion between start and state2, that
        satisfies the given constraints.
        :param start: an object of type og.State ,representing the start state
        :param goal: an object of type og.State , representing the goal state.
        :returns bool: True if there exists a path between start and goal.
        '''
        return True

# Define the space
space = ob.SE2StateSpace()

# Set the bounds 
bounds = ob.RealVectorBounds(2)
bounds.setLow(-1)
bounds.setHigh(1)
space.setBounds(bounds)

# Define the SpaceInformation object.
si = ob.SpaceInformation(space)

# Set the  MotionValidator
motion_validator = check_motion(si)
si.setMotionValidator(motion_validator)

# Create a simple setup
ss = og.SimpleSetup(si)

# Define the start and goal location
start = ob.State(space)
start[0] = 0.0
start[1] = 0.0
goal = ob.State(space)
goal[0] = 1.0
goal[1] = 1.0

# Set the start and goal states:
ss.setStartAndGoalStates(start, goal)

# define the planner
planner = og.RRT(si)
ss.setPlanner(planner)

# Attempt to solve within the given time
solved = ss.solve(10.0)
if solved:
    print("Found solution")