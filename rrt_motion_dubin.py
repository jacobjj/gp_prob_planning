import pybullet as p
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

import GPy

import os.path as osp
import sys
try:
    import ompl.base as ob
    import ompl.geometric as og
except ImportError:
    print("Run code in a the ompl docker")
    print("ValidityChecker and ValidityCheckerDistance won't work")
    raise ImportError("Run in a docker with ompl")

from models import racecar
from config import box_width, box_length, xy, cir_radius

sns.set()
obstacles, robot = racecar.set_env()


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

m = racecar.get_model_KF(robot, obstacles, racecar)
# m = racecar.get_model(robot, obstacles, racecar)

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
        racecar.reset(robot, state.getX(), state.getY(), state.getYaw())
        d = racecar.get_distance(obstacles, robot)
        return d>=0

# Wheel rotation is kept at np.pi*0.35, R = wheelbase/np.sin(max_steer)
dubinSpace = ob.DubinsStateSpace(0.4)

# Define SpaceInformation object
si = ob.SpaceInformation(dubinSpace)

# Collision checker obj
GP_check = True
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
        if time>300:
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

def start_experiment(start, samples):
    '''
    Run the RRT* experiments for 100 start and goal points for the same map
    '''
    
    if not GP_check:
        print("Turn on GP_check and rerun experiment")
        raise NameError("GP_check value does not satisfy")
    
    for i in range(start, start+samples):
        path_param = {}
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

        path_param['path'] = path
        path_param['path_interpolated'] = path_interpolated
        path_param['success'] = success

        pickle.dump(path_param, open('/root/data/dubins/CCGP-MP/exp16/path_{}.p'.format(i), 'wb'))


def start_experiment_rrt(start, samples):
    '''
    Run the RRT* experiment with collision checking using distance.
    :param start: The start index of the experiment.
    :param samples: The number of samples to be collected
    '''
    exp_num = 15
    if GP_check:
        print("Turn off GP_check and rerun experiment")
        raise NameError("GP_check value does not satisfy")
    
    for i in range(start, start+samples):
        print("Planning Path: {}".format(i))
        data = pickle.load(open('/root/data/dubins/CCGP-MP/exp{}/path_{}.p'.format(exp_num, i), 'rb'))
        start_array = data['path'][0]
        goal_array = data['path'][-1]
        path_param = {}
        # Define start and goal locations from file
        start = ob.State(dubinSpace)
        start().setX(start_array[0])
        start().setY(start_array[1])
        start().setYaw(start_array[2])

        goal = ob.State(dubinSpace)
        goal().setX(goal_array[0])
        goal().setY(goal_array[1])
        goal().setYaw(goal_array[2])

        path, path_interpolated, success = get_path(start, goal)
        path_param['path'] = path
        path_param['path_interpolated'] = path_interpolated
        path_param['success'] = success

        pickle.dump(path_param, open('/root/data/dubins/RRT/exp{}/path_{}.p'.format(exp_num, i), 'wb'))
        

def evaluate_path(start, samples):
    '''
    Evaluate the path of the trajectory.
    :param start: The start index
    :param samples: The number of samples to be collected
    '''
    exp = 'RRT'
    exp_num = 16
    for i in range(start, start+samples):
        root_file = '/root/data/dubins/{}/exp{}/path_{}.p'.format(exp, exp_num, i)
        path_param = pickle.load(open(root_file, 'rb'))
        accuracy = 0
        if path_param['success']:
            for _ in range(100):
                done = racecar.execute_path_LQR(robot, path_param['path_interpolated'], obstacles)
                if done:
                    accuracy += 1
        path_param['accuracy'] = accuracy
        print("Accuracy for path {} : {}".format(i, accuracy))
        pickle.dump(path_param, open(root_file.format(i), 'wb'))


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


if __name__=="__main__":
    start, samples = int(sys.argv[1]), int(sys.argv[2])

    # paths = [27, 19, 21, 28, 39, 23, 7, 12, 9]
    # collect_data_rrt(start)
    # for i in range(start, start+samples):
        # collect_data(paths[i])
        # collect_data_rrt(paths[i])
    # start_experiment(start, samples)
    # start_experiment_rrt(start, samples)
    evaluate_path(start, samples)
    # path_param = pickle.load(open('/root/data/dubins/path_0.p', 'rb'))
    # done = racecar.execute_path(car, path_param['path_interpolated'], obstacles)

    fig, ax = plt.subplots()

    ax.set_xlim((-0.2, 10.2))
    ax.set_ylim((-0.2, 10.2))

    # Initialize the position of obstacles
    dimensions = [box_length, box_width]
    rectangle_corner = np.r_[(-dimensions[0]/2, -dimensions[1]/2)]  

    for xy_i in racecar.xy_circle:
        plt_cir = plt.Circle(xy_i, radius=cir_radius, color='r', alpha=0.5)
        ax.add_patch(plt_cir)

    for xy_i in racecar.xy:
        plt_box = plt.Rectangle(xy_i+rectangle_corner, dimensions[0], dimensions[1], color='r', alpha=0.5)
        ax.add_patch(plt_box)

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
    if True:
        plt_objects = m.plot(ax = ax, visible_dims=np.array([0, 1]), plot_data=True, cmap='Greys')
        # Get the contour plot objects
        contour = plt_objects['gpmean'][0]
        ax.clabel(contour, contour.levels, inline=True, fontsize=10)
        # plt.colorbar(ax)
    
    racecar.del_all()