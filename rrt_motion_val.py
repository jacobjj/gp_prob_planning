''' A code to generate multiple RRT paths for the environment.
'''

import numpy as np
import pybullet as p
from scipy import stats, optimize

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import sys

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    print("Could not find OMPL")
    raise ImportError("Run inside docker!!")

# Import project specific files
from models import point
from models.randomWorld import plot_env
from gp_model import get_model, get_model_KFv2, get_model_KFv2_sparse
from config import box_width, box_length, xy, cir_radius

# Set up environment details
thresh = 0.10
N = stats.norm(scale=np.sqrt(1/2))
c = N.ppf(1-thresh)

obstacles, robot = point.set_env(
    seed=3,
    num_boxes=7,
    num_circles=3
)

# Define LTI system
A, B, M, N = point.get_dyn()

# # Define the GP model
# m = get_model(robot, obstacles, point)
# Define the GP model using state-estimation
# Noise model
N_n = stats.multivariate_normal(cov=N)
M_n = stats.multivariate_normal(cov=M)

# m = get_model_KFv2(A[:2,:2], B[:2, :2], M_n, N_n, robot, obstacles, point, 10000)
m = get_model_KFv2_sparse(A[:2,:2], B[:2, :2], M_n, N_n, robot, obstacles, point, 2500)

plot_GP = False
if plot_GP:
    fig_GP, ax_GP = plt.subplots()
    import GPy
    GPy.plotting.change_plotting_library("matplotlib")
    fig = m.plot()
    fig_GP.show()


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
        x_hat = np.c_[state[0], state[1]]
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
        p.resetBasePositionAndOrientation(robot, np.r_[state[0], state[1], 0.1], self.defaultOrientation)
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
        G = get_GP_G(np.c_[start[0], start[1]], np.c_[goal[0], goal[1]])
        sol = optimize.shgo(G, bounds=[(0, 1)], iters=10)
        if sol.success and sol.fun>c:
            return True
        return False


# Define the space
space = ob.RealVectorStateSpace(2)

# Set the bounds 
bounds = ob.RealVectorBounds(2)
bounds.setLow(-0.2)
bounds.setHigh(10)
space.setBounds(bounds)

# Define the SpaceInformation object.
si = ob.SpaceInformation(space)

GP_check = True
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

def get_path(start, goal):
    '''
    Get a RRT path from start and goal.
    :param start: og.State object.
    :param goal: og.State object.
    returns (np.array, np.array, success): A tuple of numpy arrays of a valid path,  
    interpolated path and whether the plan was successful or not.
    '''
    success = False
    # Create a simple setup
    ss = og.SimpleSetup(si)

    # Set the start and goal states:
    ss.setStartAndGoalStates(start, goal, 0.1)

    # Use RRT
    planner = og.RRT(si)
    # planner.setRange(0.5)
    # # Use RRT*
    # planner = og.RRTstar(si)

    ss.setPlanner(planner)

    # Attempt to solve within the given time
    time = 60
    solved = ss.solve(60.0)
    while not ss.haveExactSolutionPath():
        solved = ss.solve(30.0)
        time +=30
        if time>600:
            break
    if ss.haveExactSolutionPath():
        success = True
        print("Found solution")
        path = [
            [ss.getSolutionPath().getState(i)[0], ss.getSolutionPath().getState(i)[1]]
            for i in range(ss.getSolutionPath().getStateCount())
            ]
        # Define path
        ss.getSolutionPath().interpolate(100)
        path_obj = ss.getSolutionPath()
        path_interpolated = np.array([
            [path_obj.getState(i)[0], path_obj.getState(i)[1]] 
            for i in range(path_obj.getStateCount())
            ])
    else:
        path = [[start[0], start[1]], [goal[0], goal[1]]]
        path_interpolated = []

    return np.array(path), np.array(path_interpolated), success

def start_experiment(start, samples):
    '''
    Run the ccgp-mp experiment for random start and goal points.
    :param start: The starting index.
    :param samples: The number of samples to collect
    '''
    if not GP_check:
        raise NameError("Toggle GP_check to True")

    exp = 'ccgp-mp-star'
    exp_num = 8

    for i in range(start, start+samples):
        path_param = {}

        data = pickle.load(open('/root/data/point/ccgp-mp-star/exp6/path_{}.p'.format(i), 'rb'))

        start = ob.State(space)
        start[0] = data['path'][0,0]
        start[1] = data['path'][0,1]

        goal = ob.State(space)
        goal[0] = data['path'][-1, 0]
        goal[1] = data['path'][-1, 1]

        # # Define the start and goal location
        # start = ob.State(space)
        # start.random()
        # while not ValidityChecker_obj.isValid(start()):
        #     start.random()
        # goal = ob.State(space)
        # goal.random()
        # while not ValidityChecker_obj.isValid(goal()):   
        #     goal.random()

        path, path_interpolated, success = get_path(start, goal)
        path_param['path'] = path
        path_param['path_interpolated'] = path_interpolated
        path_param['success'] = success
        pickle.dump(path_param, open('/root/data/point/{}/exp{}/path_{}.p'.format(exp, exp_num, i), 'wb'))
    

def start_experiment_rrt(start, samples):
    '''
    Run the ccgp-mp experiment for random start and goal points.
    '''
    if GP_check:
        raise NameError("Toggle GP_check to False")
    
    exp = 1

    for i in range(start, start+samples):
        path_param = {}

        data = pickle.load(open('/root/data/point/ccgp-mp-star/exp6/path_{}.p'.format(i), 'rb'))

        start = ob.State(space)
        start[0] = data['path'][0,0]
        start[1] = data['path'][0,1]

        goal = ob.State(space)
        goal[0] = data['path'][-1, 0]
        goal[1] = data['path'][-1, 1]

        path, path_interpolated, success = get_path(start, goal)
        path_param['path'] = path
        path_param['path_interpolated'] = path_interpolated
        path_param['success'] = success

        pickle.dump(path_param, open('/root/data/point/rrt-star/exp{}/path_{}.p'.format(exp, i), 'wb'))


# LQG parameter:
C = np.eye(2)*10
D = np.eye(2)*0.1

def evaluate_path(start, samples):
    '''
    Evalute the path by executing it multiple times.
    :param start: The start index of the experiment
    :param samples: Number of samples in this experiment
    '''
    exp = 'ccgp-mp-star'
    exp_num = 8
    
    si_check = ob.SpaceInformation(space)
    ValidityChecker_dis_obj = ValidityCheckerDistance(si_check)
    si_check.setStateValidityChecker(ValidityChecker_dis_obj)

    for i in range(start, start+samples):
        root_file = '/root/data/point/{}/exp{}/path_{}.p'.format(exp, exp_num, i)
        path_param = pickle.load(open(root_file, 'rb'))
        accuracy = 0
        # if path_param['success']:
        if len(path_param['path_interpolated'])>0:
            ompl_traj = og.PathGeometric(si_check)
            state_temp = ob.State(si_check.getStateSpace())
            for path_i in path_param['path']:    
                state_temp[0], state_temp[1] = path_i[0], path_i[1]
                ompl_traj.append(state_temp())
            ompl_traj.interpolate(1000)
            traj = np.array([[ompl_traj.getState(i)[0], ompl_traj.getState(i)[1]] for i in range(ompl_traj.getStateCount())])
            for _ in range(100):
                _, _, done = point.execute_path(traj, C, D, si_check)
                if done:
                    accuracy += 1
        path_param['accuracy'] = accuracy
        print("Accuracy for path {} : {}".format(i, accuracy))
        pickle.dump(path_param, open(root_file.format(i), 'wb'))


if __name__ == "__main__":
    start, samples = int(sys.argv[1]), int(sys.argv[2])
    
    # start_experiment(start, samples)
    # start_experiment_rrt(start, samples)
    evaluate_path(start, samples)

    # Visualize the planning network
    visualize_network = False
    if visualize_network:
        num = 0
        # data = pickle.load(open('/root/data/path_{}.p'.format(num), 'rb'))

        # start = ob.State(space)
        # start[0] = data['path'][0,0]
        # start[1] = data['path'][0,1]

        # goal = ob.State(space)
        # goal[0] = data['path'][-1, 0]
        # goal[1] = data['path'][-1, 1]

        start = ob.State(space)
        start.random()
        while not ValidityChecker_obj.isValid(start()):
            start.random()
        goal = ob.State(space)
        goal.random()
        while not ValidityChecker_obj.isValid(goal()):
            goal.random()

        fig, ax = plt.subplots()
        sns.set()
        plot_env(ax)

        ax.scatter(start[0], start[1], color='g')
        ax.scatter(goal[0], goal[1], color='r')
        success = False
        # Create a simple setup
        ss = og.SimpleSetup(si)

        # Set the start and goal states:
        ss.setStartAndGoalStates(start, goal, 0.1)

        # # Use RRT
        # planner = og.RRT(si)

        # Use RRT*
        planner = og.RRTstar(si)

        ss.setPlanner(planner)

        # Attempt to solve within the given time
        time = 60
        solved = ss.solve(60.0)
        while not ss.haveExactSolutionPath():
            solved = ss.solve(30.0)
            time +=30
            if time>600:
                break
        if ss.haveExactSolutionPath():
            success = True
            print("Found solution")

        import networkx as nx
        
        planner_data = ob.PlannerData(si)
        ss.getPlannerData(planner_data)
        G = nx.parse_graphml(planner_data.printGraphML())
        pos = nx.get_node_attributes(G, 'coords')
        new_pos = {key:np.array(value.split(','), dtype=np.float) for key, value in pos.items()}

        nx.draw_networkx(G, new_pos, ax = ax, alpha=0.5)
        fig.show()

    visualize_paths = False
    if visualize_paths:
        fig, ax = plt.subplots()
        sns.set()

        plot_env(ax)
        
        total_success = 0
        for i in range(40):
            data = pickle.load(open('/root/data/path_{}.p'.format(i),'rb'))
            path = data['path']
            path_interpolated = data['path_interpolated']
            total_success += int(data['success'])

            ax.scatter(path[0, 0], path[0, 1], color='r', marker='x')
            ax.scatter(path[-1, 0], path[-1, 1], color='g', marker='o')

            if path!=[]:
                ax.plot(path[:,0], path[:,1], color='b', alpha=0.5)
                ax.scatter(path[:,0], path[:,1], color='b', alpha=0.5)
                temp = path.shape[0]-1