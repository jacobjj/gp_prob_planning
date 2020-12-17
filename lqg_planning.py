'''Planning according to LQG-MP.
'''
import numpy as np
import pybullet as p
from scipy import stats, optimize, special

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
from gp_model import get_model, get_model_KF
from config import box_width, box_length, xy, cir_radius

obstacles, robot = point.set_env()

# Define LTI system
A, B, M, N= point.get_dyn()

# Noise model
N_n = stats.multivariate_normal(cov=N) # Observation Noise
M_n = stats.multivariate_normal(cov=M) # Motion Noise

# Define gamma function for evaluating path
S = lambda x: special.gammainc(1, x)

# Threshold for each point.
thresh = 0.1

fig, ax = plt.subplots()
sns.set()

ax.set_xlim((-0.2, 10.2))
ax.set_ylim((-0.2, 10.2))

# Initialize the position of obstacles
dimensions = [box_length, box_width]
rectangle_corner = np.r_[(-dimensions[0]/2, -dimensions[1]/2)]  

for xy_i in point.xy_circle:
    plt_cir = plt.Circle(xy_i, radius=cir_radius, color='r')
    ax.add_patch(plt_cir)

for xy_i in point.xy:
    plt_box = plt.Rectangle(xy_i+rectangle_corner, dimensions[0], dimensions[1], color='r')
    ax.add_patch(plt_box)


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
        p.resetBasePositionAndOrientation(robot, np.r_[state[0], state[1], 0.1], self.defaultOrientation)
        return point.get_distance(obstacles, robot)


# Define the space
space = ob.RealVectorStateSpace(2)

# Set the bounds 
bounds = ob.RealVectorBounds(2)
bounds.setLow(-0.2)
bounds.setHigh(10)
space.setBounds(bounds)


# Define the start and goal location
start = ob.State(space)
# start.random()
# while not ValidityChecker_obj.isValid(start()):
#     start.random()
# start[0] = 1.0
# start[1] = 8.0
goal = ob.State(space)
goal[0] = 8.5
goal[1] = 8.5

# Function to get the distribution of path.
A = A[:2, :2]
B = B[:2, :2]
# LQG parameter:
C = np.eye(2)
D = np.eye(2)

Q = np.block([
    [M, np.zeros((2,2))], 
    [np.zeros((2,2)), N]
])

# try:
#     path = np.load("path_lqg.npy")
#     path_interpolated = np.load("path_interpolated_lqg.npy")
#     # start[0] = path[0,0]
#     # start[1] = path[0,1]
# except FileNotFoundError:

def get_path(start, goal):
    '''
    Get a RRT path from start and goal.
    :param start: og.State object.
    :param goal: og.State object.
    returns (np.array, np.array): A tuple of numpy arrays of a valid path and 
    interpolated path.
    '''

    # Define the SpaceInformation object.
    si = ob.SpaceInformation(space)

    ValidityChecker_obj = ValidityCheckerDistance(si)

    si.setStateValidityChecker(ValidityChecker_obj)

    # Create a simple setup
    ss = og.SimpleSetup(si)

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
            [ss.getSolutionPath().getState(i)[0], ss.getSolutionPath().getState(i)[1]]
            for i in range(ss.getSolutionPath().getStateCount())
            ]
        ss.getSolutionPath().interpolate(100)
        path_obj = ss.getSolutionPath()
        path_interpolated = np.array([
            [path_obj.getState(i)[0], path_obj.getState(i)[1]] 
            for i in range(path_obj.getStateCount())
        ])
    else:
        path = []
        path_interpolated = []
    return np.array(path), np.array(path_interpolated)


def lqg_mp(start, goal):
    '''
    Get the lqg_mp plan for a the given start and goal position.
    :param start: A np.array representing the start position.
    :param goal: A np.array representing the goal positon.
    :return (path, path_interpolated): Returns a tuple of path and interpolated path.
    '''

    best_path = []
    best_cost = 0.0
    best_path_interpolated = []
    best_pi = []
    for _ in range(1000):
        path, path_interpolated = get_path(start, goal)

        # Define initial parameters:
        P = np.eye(2)*1e-1
        R = np.block([
            [P, np.zeros((2,2))],
            [np.zeros((2,2)), np.zeros((2,2))]
        ])

        L_list = point.get_lqr(path_interpolated, C, D)

        V_list = [P]
        for p_i,L in zip(path_interpolated[:-1], reversed(L_list)):
            # Kalman-updates - forward
            P_bar = A@P@A.T + M
            K = P_bar@np.linalg.inv(P_bar+N)
            P = (np.eye(2)-K)@P_bar

            # Distribution
            F = np.block([
                [A, B@L],
                [K@A, A+B@L-K@A]
            ])
            G = np.block([
                [np.eye(2), np.zeros((2,2))],
                [K, K]
            ])

            R = F@R@F.T + G@Q@G.T
            lam = np.block([
                [np.eye(2), np.zeros((2,2))],
                [np.zeros((2,2)), L]
            ])
            V = lam@R@lam.T
            V_list.append(V)
        
        # Calculate cost:
        cost = 1
        p_list = []
        for i, p_i in enumerate(path_interpolated):
            p.resetBasePositionAndOrientation(robot, np.r_[p_i, 0.1], (0.0, 0.0, 0.0, 0.1))
            d = point.get_distance(obstacles, robot)
            c_i = d/np.sqrt(V_list[i][0,0])
            p_i = S((c_i**2)/2)
            cost *=S((c_i**2)/2)
            p_list.append(p_i)

        # Doing a filter of 
        if cost>best_cost :#and all(p_i>(1-thresh) for p_i in p_list):
            best_path = path
            best_path_interpolated = path_interpolated
            best_cost = cost
            best_pi = p_list

    return best_path, best_path_interpolated, best_pi

import os
import pickle

def start_experiment():
    '''
    Run the LQG experiment for the start and goal points for the same RRT-paths
    '''
    # Define path
    folder_loc = '/root/data'
    
    # for file_i in os.listdir(folder_loc):
    for i in range(350, 400):
        # if '.p' in file_i:
        file_i = 'path_{}.p'.format(i)
        data = pickle.load(open(os.path.join(folder_loc, file_i), 'rb'))
        start_array = data['path'][0]
        goal_array = data['path'][-1]
        start = ob.State(space)
        start[0] = start_array[0]
        start[1] = start_array[1]
        goal = ob.State(space)
        goal[0] = goal_array[0]
        goal[1] = goal_array[1]

        path_param = {}
        path, path_interpolated, p_list = lqg_mp(start, goal)
        path_param['path'] = path
        path_param['path_interpolated'] = path_interpolated
        path_param['p_list'] = p_list
        # Evaluate 100 paths
        accuracy = 0
        si_check = ob.SpaceInformation(space)
        ValidityChecker_dis_obj = ValidityCheckerDistance(si_check)
        si_check.setStateValidityChecker(ValidityChecker_dis_obj)
        for _ in range(100):
            _, _, done = point.execute_path(path_interpolated, C, D, si_check)
            if done:
                accuracy += 1
        path_param['accuracy'] = accuracy

        pickle.dump(path_param, open(os.path.join(folder_loc, 'lqg_mp', file_i), 'wb'))

if __name__ == "__main__":

    start_experiment()

    if False:
        ax.scatter(start[0], start[1], color='r', marker='x')
        ax.scatter(goal[0], goal[1], color='g', marker='o')

        if path!=[]:
            ax.plot(path[:,0], path[:,1], color='b', alpha=0.5, label='RRT path')
            ax.scatter(path[:,0], path[:,1], color='b', label='RRT path')

        # Evaluate 10 paths
        successfull = 0
        si_check = ob.SpaceInformation(space)
        ValidityChecker_obj = ValidityCheckerDistance(si_check)
        si_check.setStateValidityChecker(ValidityChecker_obj)
        for _ in range(10):
            path_est, path_noisy, done = point.execute_path(path_interpolated, C, D, si_check)
            # if not done:
            path_est = np.array(path_est)
            # ax.plot(path_est[:,0], path_est[:,1], color='r', label='estimated state')
            path_noisy = np.array(path_noisy)
            ax.plot(path_noisy[:,0], path_noisy[:,1], '--',color='g', alpha=0.5, label='real state')
            # else:
            #     successfull+=1
        print("Total succesful paths {}".format(successfull))