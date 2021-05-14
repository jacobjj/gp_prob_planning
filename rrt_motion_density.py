''' Script to generate random environments with different density and path RRT
paths
'''

import numpy as np
import pickle
import json
from os import path as osp
import os
from scipy import stats, optimize
import time
import networkx as nx
import matplotlib.pyplot as plt
try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    print("Could not find OMPL")
    raise ImportError("Run inside docker!!")

# Import project specific files
from models import point
from models.randomWorld import plot_env
import json
import pybullet as pyb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--envNum', required=True, type=int)
parser.add_argument('--numObstacles', required=True, type=int)
parser.add_argument('--seed', required=True, type=int)

args = parser.parse_args()

dataFolder = f'/root/data/density/env{args.envNum}'

if not osp.isdir(dataFolder):
    os.mkdir(dataFolder)

def plan_again(count, planPath=True, evaluatePath=False):
    paramFile = osp.join(dataFolder, 'envParam.json')
    if osp.exists(paramFile):
        envParam = json.load(open(paramFile, 'r'))
    else:
        envParam = dict(
            seed=args.seed+count,
            num_boxes=args.numObstacles,
            num_circles=args.numObstacles
        )

        json.dump(envParam, open(paramFile, 'w'))

    # Set up environment details
    thresh = 0.01
    N = stats.norm(scale=np.sqrt(1/2))
    c = N.ppf(1-thresh)

    # Define environment
    obstacles, robot = point.set_env(
        **envParam
    )
    # Define the space
    space = ob.RealVectorStateSpace(2)

    # Set the bounds 
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-2.)
    bounds.setHigh(12.0)
    space.setBounds(bounds)

    if planPath:
        # Get GP_model
        m = point.get_model_KF(robot, obstacles, samples=2500, dataFolder=dataFolder)
        # m = point.get_model_KF_sparse(robot, obstacles, samples=2500, dataFolder=dataFolder)
        # TODO: Initialize model with GP_model
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
                sol = optimize.shgo(G, bounds=[(0, 1)], iters=10, n=10, sampling_method='sobol')
                if sol.success and sol.fun>c:
                    return True
                return False

        # Define the SpaceInformation object.
        si = ob.SpaceInformation(space)
        print("Using GP model for collision check")
        # Set the StateValidator
        ValidityChecker_obj = ValidityChecker(si)

        # Set the MotionValidator
        MotionValidator_obj = check_motion(si)
        si.setMotionValidator(MotionValidator_obj)

        si.setStateValidityChecker(ValidityChecker_obj)
        for pathNum in range(25, 30):
            # Define start and goal
            posN = stats.norm(scale=0.5)
            startMean = np.r_[-0.25, 10.0]
            startSample = startMean + posN.rvs()
            start = ob.State(space)
            start[0] = startSample[0]
            start[1] = startSample[1]
            mean, var = m.predict(np.c_[start[0], start[1]])
            c_start = (mean/np.sqrt(2*var))[0,0]
            print(f"Start Zscore :{c_start, c}")
            goalMean = np.r_[11.0, 0.0]
            goalSample = goalMean + posN.rvs()
            goal = ob.State(space)
            goal[0] = goalSample[0]
            goal[1] = goalSample[1]
            mean, var = m.predict(np.c_[goal[0], goal[1]])
            c_goal = (mean/np.sqrt(2*var))[0,0]
            print(f"Goal Zscore :{c_goal, c}")
            # Plan Path
            success = False
            # Create a simple setup
            ss = og.SimpleSetup(si)

            # Set the start and goal states:
            ss.setStartAndGoalStates(start, goal, 0.1)

            # Use RRT
            planner = og.RRT(si)
            # planner = og.LazyRRT(si)
            # planner.setRange(0.5)

            ss.setPlanner(planner)

            # fig, ax = plt.subplots()
            # ax.scatter(start[0], start[1], color='g')
            # ax.scatter(goal[0], goal[1], color='r')
            def plot_graph(ax, pd):
                '''
                Plot planner data on the graph.
                :param ax: The axis on which to plot the graph.
                :param pd: A ompl.base.PlannerData object
                '''
                G = nx.parse_graphml(pd.printGraphML())
                pos = nx.get_node_attributes(G, 'coords')
                new_pos = {key:np.array(value.split(','), dtype=np.float) for key, value in pos.items()}
                nx.draw_networkx(G, new_pos, ax = ax, alpha=0.5)
                plot_env(ax, alpha=1)

            # Attempt to solve within the given time
            planTime = 60
            startTime = time.time()
            solved = ss.solve(planTime)
            # planner_data = ob.PlannerData(si)
            # ss.getPlannerData(planner_data)
            # plot_graph(ax, planner_data)
            # plt.show(block=False)
            # plt.pause(5)

            while not ss.haveExactSolutionPath():
                solved = ss.solve(30.0)
                # ss.getPlannerData(planner_data)
                # plot_graph(ax, planner_data)
                # plt.show(block=False)
                # plt.pause(1)
                planTime +=30
                if planTime>600:
                    break
            totalTime = time.time()-startTime
            # plt.show()
            # Visualize search tree

            if ss.haveExactSolutionPath():
                ss.simplifySolution()
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

            # Save path parameters
            pathParam = dict(
                path=path,
                path_interpolated=path_interpolated,
                success=success,
                totalTime=totalTime
            )
            pickle.dump(pathParam, open(osp.join(dataFolder, f'path_{pathNum}.p'), 'wb'))

    if evaluatePath:
        # LQG parameter:
        C = np.eye(2)*10
        D = np.eye(2)*0.1

        class ValidityCheckerDistance(ob.StateValidityChecker):
            '''A class to check the validity of the state, by checking distance function
            '''
            defaultOrientation = pyb.getQuaternionFromEuler([0., 0., 0.])
            
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
                pyb.resetBasePositionAndOrientation(robot, np.r_[state[0], state[1], 0.1], self.defaultOrientation)
                return point.get_distance(obstacles, robot)


        si_check = ob.SpaceInformation(space)
        ValidityChecker_dis_obj = ValidityCheckerDistance(si_check)
        si_check.setStateValidityChecker(ValidityChecker_dis_obj)

        for pathNum in range(0, 30):
            path_param = pickle.load(open(osp.join(dataFolder, f'path_{pathNum}.p'), 'rb'))
            accuracy = 0
            if path_param['success']:
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
            pickle.dump(path_param, open(osp.join(dataFolder, f'path_{pathNum}.p'), 'wb'))


plan_again(0, planPath=False, evaluatePath=True)
# plan_again(0)

# TODO: Save Evaluation results