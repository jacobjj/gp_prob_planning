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

# Import project specific files
from models import point
from gp_model import get_model

# Set up environment details
thresh = 0.001
N = stats.norm(scale=np.sqrt(1/2))
c = N.ppf(1-thresh)

obstacles, robot = point.set_env()

# Define the GP model
m = get_model(robot, obstacles, point)
# Define LTI system
A, B = point.get_dyn()

fig, ax = plt.subplots(1,2)
sns.set()

# TODO: Set up the plotting code.

K = m.kern.K(m.X)
prior_var = 1e-6
# TODO : There is a better way to this inverse!!
K_inv = np.linalg.inv(K+np.eye(K.shape[0])*prior_var)
weights = K_inv@ m.Y
k_x_x = m.kern.K(np.c_[0,0])

def get_GP_G(start, goal):
    '''
    The guassian function from x_start and x_goal.Return the function for line
    :param start:  A np.array representing the start position
    :param goal: A np.array representing the goal position
    :return function: Return the function of F along the line
    '''
    def G(alpha, *args):
        '''
        The function G
        '''
        if alpha<0:
            alpha=0
        elif alpha>1:
            alpha = 1
        x_hat = (1-alpha)*start + alpha*goal
        k_star = m.kern.K(x_hat, m.X)
        var = k_x_x-k_star@K_inv@k_star.T
        return ((k_star@weights)/np.sqrt(2*var))[0]
    return G


class check_motion(ob.MotionValidator):
    '''A motion validator check using chance constrained.
    '''
    def __init__(self, spaceInformation):
        '''
        Initialize the motion validator.
        :param spaceInformation: A ob.SpaceInformation object.
        '''
        super(check_motion, self).__init__(spaceInformation)

    def checkMotion(self, start, goal):
        '''
        Check if there exists a valid motion between start and state2, that
        satisfies the given constraints.
        :param start: an object of type og.State ,representing the start state
        :param goal: an object of type og.State , representing the goal state.
        :returns bool: True if there exists a path between start and goal.
        '''
        # assert isinstance(start, ob.State), "Start has to be of ob.State"
        # assert isinstance(goal, ob.State), "Goal has t obe of ob.State"
        G = get_GP_G(np.c_[start.getX(), start.getY()], np.c_[goal.getX(), goal.getY()])
        sol = optimize.shgo(G, bounds=[(0, 1)], iters=10)
        if sol.success and sol.fun>c:
            return True
        return False


# Define the space
space = ob.SE2StateSpace()

# Set the bounds 
bounds = ob.RealVectorBounds(2)
bounds.setLow(-0.2)
bounds.setHigh(10)
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
goal[0] = 9.0
goal[1] = 9.0

# Set the start and goal states:
ss.setStartAndGoalStates(start, goal)

# define the planner
planner = og.RRT(si)
ss.setPlanner(planner)

# Attempt to solve within the given time
solved = ss.solve(10.0)
if solved:
    print("Found solution")
    path = [
        [ss.getSolutionPath().getState(i).getX(), ss.getSolutionPath().getState(i).getY()]
        for i in range(ss.getSolutionPath().getStateCount())
        ] 