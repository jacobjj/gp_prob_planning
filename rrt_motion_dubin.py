import pybullet as p
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

import GPy

import os
from os import path as osp
import sys
try:
    import ompl.base as ob
    import ompl.geometric as og
except ImportError:
    print("Run code in a the ompl docker")
    print("ValidityChecker and ValidityCheckerDistance won't work")
    raise ImportError("Run in a docker with ompl")

from models import racecarv2
from config import box_width, box_length, xy, cir_radius

sns.set()
obstacles, robot = racecarv2.set_env()

def SE2State2Tuple(state):
    '''
    convert SE2 state object to tuple
    :param state: An SE2 object
    :return tuple: (x,y, theta) 
    '''
    return (state.getX(), state.getY(), state.getYaw())


# Set up planning threshold limits
thresh = 0.05
N = stats.norm(scale=np.sqrt(1/2))
c = N.ppf(1-thresh)
def reset_threshold(thresh):
    '''
    A function to reset the threshold for planning environments:
    :param thresh: A value between [0, 1] to set as threshold for planning.
    '''
    global c
    c = N.ppf(1-thresh)
    print("New threshold set as {}".format(thresh)) 

m = racecarv2.get_model_KF(robot, obstacles)

class ValidityChecker(ob.StateValidityChecker):
    '''A class to check the validity of the state
    '''
    def isValid(self, state):
        '''
        Check if the given state is valid.
        '''
        yaw = state.getYaw()
        x_hat = np.r_[[[state.getX(), state.getY(), np.cos(yaw), np.sin(yaw)]]]
        mean, var = m.predict(x_hat)
        return (mean/np.sqrt(2*var))[0,0] > c

class ValidityCheckerDistance(ob.StateValidityChecker):
    '''A class to check the validity of the state by calculating
    the distance to obstacle
    '''
    def isValid(self, state):
        '''
        Check if the given state is valid.
        '''
        collision = racecarv2.check_collision(
            np.r_[state.getX(), state.getY(), state.getYaw()],
            obstacles,
            robot
        )
        return not collision

# Wheel rotation is kept at np.pi*0.35, R = wheelbase/np.sin(max_steer)
dubinSpace = ob.DubinsStateSpace(0.5)

# Define SpaceInformation object
si = ob.SpaceInformation(dubinSpace)

# Collision checker obj
GP_check = False
if GP_check:
    ValidityChecker_obj = ValidityChecker(si)
else:
    ValidityChecker_obj = ValidityCheckerDistance(si)
si.setStateValidityChecker(ValidityChecker_obj)

# Set the bounds of the space
bounds = ob.RealVectorBounds(2)
bounds.setLow(0.0)
bounds.setHigh(10)
dubinSpace.setBounds(bounds)


def get_path(start, goal):
    '''
    Get the RRT* for SE2 space for a given start and goal.
    :param start: og.State object.
    :param goal: og.State object.
    returns (np.array, np.array, success): A tuple of numpy arrays of a valid path,  
    interpolated path and whether the plan was successful or not.
    '''
    success = False
    ss = og.SimpleSetup(si)
    ss.setStartAndGoalStates(start, goal, 0.1)

    # planner = og.RRT(si)
    planner = og.RRTstar(si)
    ss.setPlanner(planner)

    time = 60.0
    solved = ss.solve(time)

    while not ss.haveExactSolutionPath():
        solved = ss.solve(30.0)
        time += 30
        if time>1200:
            break

    if ss.haveExactSolutionPath():
        ss.simplifySolution()
        success = True
        print("Found Solution")
        path = [
            SE2State2Tuple(ss.getSolutionPath().getState(i))
            for i in range(ss.getSolutionPath().getStateCount())
        ]
        ss.getSolutionPath().interpolate(5000)
        path_obj = ss.getSolutionPath()        
        path_interpolated = [
            SE2State2Tuple(path_obj.getState(i)) 
            for i in range(path_obj.getStateCount())
        ]
    else:
        path = [SE2State2Tuple(start()), SE2State2Tuple(goal())]
        path_interpolated = []
    
    return np.array(path), np.array(path_interpolated), success

def start_experiment(start, samples, expFolder, randomSample, compExpFolder=None, ccgpFlag=False):
    '''
    Save the RRT* experiments for a given range of (start, start+samples) with or without 
    CCGP. 
    :param start: The start index
    :param samples: The number of samples to collect.
    :param expFolder: Location to save the data.
    :param RandomSample: If True, randomly sample valid start and goal pairs.
    :param compExpFolder: The experiment Folder to select start and goal pairs from.
    :param ccgpFlag: If True, plan RRT* using CCGP 
    '''
    assert osp.isdir(expFolder), f"The file directory, {expFolder}, does not exits"

    if not randomSample and compExpFolder is not None:
        assert osp.isdir(compExpFolder), "No folder found to compare"

    if ccgpFlag:
        assert GP_check, "Turn on GP_check and rerun experiment"
    else:
        assert (not GP_check), "Turn off GP_check and rerun experiment"
    
    for i in range(start, start+samples):
        path_param = {}
        if randomSample:
            # Define random start and goal locations
            start = ob.State(dubinSpace)
            start.random()
            while not ValidityChecker_obj.isValid(start()):
                start.random()

            goal = ob.State(dubinSpace)
            goal.random()
            while not ValidityChecker_obj.isValid(goal()):
                goal.random()  
        else:
            data = pickle.load(open(osp.join(compExpFolder, f'path_{i}.p'), 'rb'))

            start = ob.State(dubinSpace)
            start[0] = data['path'][0, 0]
            start[1] = data['path'][0, 1]
            start[2] = data['path'][0, 2]

            goal = ob.State(dubinSpace)
            goal[0] = data['path'][-1, 0]
            goal[1] = data['path'][-1, 1]
            goal[2] = data['path'][-1, 2]
        
        path, path_interpolated, success = get_path(start, goal)

        path_param['path'] = path
        path_param['path_interpolated'] = path_interpolated
        path_param['success'] = success

        pickle.dump(path_param, open(osp.join(expFolder, f'path_{i}.p'), 'wb'))
        

def evaluate_path(start, samples, expFolder):
    '''
    Evaluate the path of the trajectory for 100 trials and save the stats back
    in the pickle file.
    :param start: The start index
    :param samples: The number of samples to be collected
    :param expFolder: The folder with the paths
    '''
    for i in range(start, start+samples):
        root_file = osp.join(expFolder, f'path_{i}.p')
        path_param = pickle.load(open(root_file, 'rb'))
        accuracy = 0
        if path_param['success']:
            for _ in range(100):
                _, done = racecarv2.execute_path(robot, path_param['path_interpolated'], obstacles)
                if done:
                    accuracy += 1
        path_param['accuracy'] = accuracy
        print("Accuracy for path {} : {}".format(i, accuracy))
        pickle.dump(path_param, open(root_file, 'wb'))


def collect_data(num):
    '''
    The function to collect data for fig:diff_thresh_paths for the paper.
    '''
    root_folder = '/root/data/dubins/'
    file_loc = osp.join(root_folder, 'CCGP-MP', 'exp16', 'path_{}.p'.format(num))
    data = {}
    path_param = pickle.load(open(file_loc, 'rb'))
    # Set the start and goal locations
    start_array = path_param['path'][0]
    goal_array = path_param['path'][-1]
    # Define start and goal locations from file
    start = ob.State(dubinSpace)
    start().setX(start_array[0])
    start().setY(start_array[1])
    start().setYaw(start_array[2])

    goal = ob.State(dubinSpace)
    goal().setX(goal_array[0])
    goal().setY(goal_array[1])
    goal().setYaw(goal_array[2])
    # Evaluate the trajectory for these thresholds
    if path_param['success']:
        path_param['true_traj'] = []
        path_param['est_traj'] = []
        for _ in range(20):
            done, true_traj, est_traj = racecar.execute_path_LQR_data(
                robot, 
                path_param['path_interpolated'], 
                obstacles,
                get_log=True
            )
            path_param['true_traj'].append(np.array(true_traj))
            path_param['est_traj'].append(np.array(est_traj))
    data['0.01'] = path_param

    thresh_list = [0.05, 0.1]
    # Plan the trajectory for different thresholds
    for thresh in thresh_list:
        reset_threshold(thresh)
        path_param = {}
        path, path_interpolated, success = get_path(start, goal)
        path_param['path'] = path
        path_param['path_interpolated'] = path_interpolated
        path_param['success'] = success

        if success:
            # Evaluate the trajectory for these thresholds
            path_param['true_traj'] = []
            path_param['est_traj'] = []
            for _ in range(20):
                done, true_traj, est_traj = racecar.execute_path_LQR_data(
                    robot, 
                    path_param['path_interpolated'], 
                    obstacles,
                    get_log=True
                )
                path_param['true_traj'].append(np.array(true_traj))
                path_param['est_traj'].append(np.array(est_traj))
        data['{}'.format(thresh)] = path_param

    save_file_loc = osp.join(root_folder,'fig_data','fig3','path_{}.p'.format(num))
    pickle.dump(data, open(save_file_loc, 'wb'))

def collect_data_rrt(num):
    '''
    The function to collect data for fig:diff_thresh_paths
    :param num: The path number
    '''
    root_folder = '/root/data/dubins/'
    file_loc = osp.join(root_folder, 'CCGP-MP', 'exp16', 'path_{}.p'.format(num))
    path_param = pickle.load(open(file_loc, 'rb'))
    # Set the start and goal locations
    start_array = path_param['path'][0]
    goal_array = path_param['path'][-1]
    # Define start and goal locations from file
    start = ob.State(dubinSpace)
    start().setX(start_array[0])
    start().setY(start_array[1])
    start().setYaw(start_array[2])

    goal = ob.State(dubinSpace)
    goal().setX(goal_array[0])
    goal().setY(goal_array[1])
    goal().setYaw(goal_array[2])
    # Evaluate the trajectory for these thresholds
    path, path_interpolated, success = get_path(start, goal)
    path_param['path'] = path
    path_param['path_interpolated'] = path_interpolated
    path_param['success'] = success

    data_file = osp.join(root_folder, 'fig_data', 'fig3', 'path_{}.p'.format(num))
    data = pickle.load(open(data_file, 'rb'))

    if success:
        # Evaluate the trajectory for these thresholds
        path_param['true_traj'] = []
        path_param['est_traj'] = []
        for _ in range(20):
            done, true_traj, est_traj = racecar.execute_path_LQR_data(
                robot, 
                path_param['path_interpolated'], 
                obstacles,
                get_log=True
            )
            path_param['true_traj'].append(np.array(true_traj))
            path_param['est_traj'].append(np.array(est_traj))
    data['rrt'] = path_param
    pickle.dump(data, open(data_file, 'wb'))

import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Collect RRT/RRT* paths using w/ without CCGP-MP")
    parser.add_argument('--start', help='The start index of the experiment', default=0, type=int)
    parser.add_argument('--samples', help='Number of paths to collect', default=1, type=int)
    parser.add_argument('--expFolder', help='Location to save the collected data')
    parser.add_argument('--compExpFolder', help='Location to save the collected data', default=None)
    parser.add_argument('-r', '--randomSample', help='Sample random start and goal location', action="store_true")
    parser.add_argument('--CCGP', help='Plan with CCGP enabled', action="store_true")

    args = parser.parse_args()

    start_experiment(args.start, args.samples, args.expFolder, args.randomSample, compExpFolder=args.compExpFolder)
    evaluate_path(args.start, args.samples, args.expFolder)
    
    # path_param = pickle.load(open('/root/data/dubins/path_0.p', 'rb'))
    # done = racecar.execute_path(car, path_param['path_interpolated'], obstacles)

    # fig, ax = plt.subplots()

    # ax.set_xlim((-0.2, 10.2))
    # ax.set_ylim((-0.2, 10.2))

    # # Initialize the position of obstacles
    # dimensions = [box_length, box_width]
    # rectangle_corner = np.r_[(-dimensions[0]/2, -dimensions[1]/2)]  

    # for xy_i in racecar.xy_circle:
    #     plt_cir = plt.Circle(xy_i, radius=cir_radius, color='r', alpha=0.5)
    #     ax.add_patch(plt_cir)

    # for xy_i in racecar.xy:
    #     plt_box = plt.Rectangle(xy_i+rectangle_corner, dimensions[0], dimensions[1], color='r', alpha=0.5)
    #     ax.add_patch(plt_box)

    if False:
        # Define random start and goal locations
        start = ob.State(dubinSpace)
        start.random()
        while not ValidityChecker_obj.isValid(start()):
            start.random()

        goal = ob.State(dubinSpace)
        goal.random()
        while not ValidityChecker_obj.isValid(goal()):
            goal.random()  
        path, path_interpolated, success = get_path(start, goal)

        ax.scatter(start().getX(), start().getY(), color='g')
        ax.scatter(goal().getX(), goal().getY(), color='r')

        if success:
            for path_i in path:
                ax.arrow(path_i[0], path_i[1], 0.1*np.cos(path_i[2]), 0.1*np.sin(path_i[2]), color='k')


            for path_i in path_interpolated:
                ax.arrow(path_i[0], path_i[1], 0.1*np.cos(path_i[2]), 0.1*np.sin(path_i[2]), color='b', width=0.01)
        
    # fig.show()
    # fig, ax = plt.subplots()
    if False:
        plt_objects = m.plot(ax = ax, visible_dims=np.array([0, 1]), plot_data=True, cmap='Greys')
        # Get the contour plot objects
        contour = plt_objects['gpmean'][0]
        ax.clabel(contour, contour.levels, inline=True, fontsize=10)
        # plt.colorbar(ax)