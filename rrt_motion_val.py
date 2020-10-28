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
    print("Could not find OMPL")
    raise ImportError("Run inside docker!!")

# Import project specific files
from models import point
from gp_model import get_model
from config import box_width, box_length, xy, cir_radius

# Set up environment details
thresh = 0.001
N = stats.norm(scale=np.sqrt(1/2))
c = N.ppf(1-thresh)

obstacles, robot = point.set_env()

# Define LTI system
A, B, M, N = point.get_dyn()

# Define the GP model
m = get_model(robot, obstacles, point)
# Define LTI system
A, B = point.get_dyn()

fig, ax = plt.subplots(1,2)
sns.set()

ax[0].set_xlim((-0.2, 10.2))
ax[0].set_ylim((-0.2, 10.2))

# Initialize the position of obstacles
dimensions = [box_length, box_width]
rectangle_corner = np.r_[(-dimensions[0]/2, -dimensions[1]/2)]  

for xy_i in point.xy_circle:
    plt_cir = plt.Circle(xy_i, radius=cir_radius, color='r')
    ax[0].add_patch(plt_cir)

for xy_i in point.xy:
    plt_box = plt.Rectangle(xy_i+rectangle_corner, dimensions[0], dimensions[1], color='r')
    ax[0].add_patch(plt_box)

K = m.kern.K(m.X)
prior_var = 1e-1
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
        mean, var = m.predict(x_hat)
        return (mean/np.sqrt(2*var))
    return G

class ValidityChecker(ob.StateValidityChecker):
    '''A class to check the validity of the state
    '''
    def isValid(self, state):
        '''
        Check if the given state is valid.
        :param state: An ob.State object to be checked.
        :returns bool: True if the state is valid. 
        '''
        x_hat = np.c_[state.getX(), state.getY()]
        return self.getZscore(x_hat)>c

    def getZscore(self, x):
        '''
        The ratio of E[f(x)]/sqrt(2*var(f(x))).
        :param x: a numpy array of state x.
        :return float: The value of E[f(x)]/sqrt(2*var(f(x)))
        '''
        mean, var = m.predict(x)
        return (mean/np.sqrt(2*var))[0,0]


class ValidityCheckerDistance(ob.StateValidityChecker):
    '''A class to check the validity of the state, by checking distance function
    '''
    defaultOrientation = p.getQuaternionFromEuler([0., 0., 0.])
    
    def isValid(self, state):
        '''
        Check if the given state is valid.
        :param state: An ob.STate object to be checked.
        :return bool: True if the state is valid.
        '''
        return self.getDistance(state)>0

    def getDistance(self, state):
        '''
        Get the shortest distance from robot to obstacle.
        :param x: A numpy array of state x.
        :returns float: The closest distance between the robot and obstacle.
        '''
        p.resetBasePositionAndOrientation(robot, np.r_[state.getX(), state.getY(), 0.1], self.defaultOrientation)
        return point.get_distance(obstacles, robot)

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

def main(GP_check=True):

    # Define the SpaceInformation object.
    si = ob.SpaceInformation(space)

    if GP_check:
        print("Using GP model for collision check")
        # Set the StateValidator
        ValidityChecker_obj = ValidityChecker(si)

        # Set the MotionValidator
        MotionValidator_obj = check_motion(si)
        si.setMotionValidator(MotionValidator_obj)
    else:
        print("Using noisy distance function")
        ValidityChecker_obj = ValidityCheckerDistance(si)

    si.setStateValidityChecker(ValidityChecker_obj)

    # Create a simple setup
    ss = og.SimpleSetup(si)

    # Define the start and goal location
    start = ob.State(space)
    start[0] = 2.0
    start[1] = 4.0
    goal = ob.State(space)
    goal[0] = 9.0
    goal[1] = 9.0

    # Set the start and goal states:
    ss.setStartAndGoalStates(start, goal)

    # define the planner
    planner = og.RRT(si)
    ss.setPlanner(planner)

    # Attempt to solve within the given time
    solved = ss.solve(15.0)
    if solved:
        print("Found solution")
        path = [
            [ss.getSolutionPath().getState(i).getX(), ss.getSolutionPath().getState(i).getY()]
            for i in range(ss.getSolutionPath().getStateCount())
            ]
    return path

ax[0].scatter(2, 4, color='r', marker='x')
ax[0].scatter(9, 9, color='g', marker='o')

# Check the distance function.
ax[1].plot([0,1], [c, c,], color='k')

if __name__ == "__main__":
    
    path = np.array(main(True))
    ax[0].plot(path[:,0], path[:,1], color='b', alpha=0.5, label='With CC')
    ax[0].scatter(path[:,0], path[:,1], color='b', alpha=0.5, label='With CC')

    for j, _ in enumerate(path[:-1]):
        G = get_GP_G(path[j][None,:], path[j+1][None,:])
        G_samp = np.squeeze([G(a) for a in np.linspace(0, 1, 100)])
        ax[1].plot(np.linspace(0, 1, 100), G_samp, color='b')

    path = np.array(main(False))
    ax[0].plot(path[:,0], path[:,1], alpha=0.5, color='r', label='With noisy distance')
    ax[0].scatter(path[:,0], path[:,1], color='r', alpha=0.5, label='With noisy distance')

    for j, _ in enumerate(path[:-1]):
        G = get_GP_G(path[j][None,:], path[j+1][None,:])
        G_samp = np.squeeze([G(a) for a in np.linspace(0, 1, 100)])
        ax[1].plot(np.linspace(0, 1, 100), G_samp, color='r', label='noisy distance')

    ax[0].legend()
    fig.show()