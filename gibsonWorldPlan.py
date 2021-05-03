'''A script to plan for the gibsonWorldEnv
'''

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
            collision = gibsonWorld.check_collision(
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
except ImportError:
    print("Run code in a the ompl docker")
    print("ValidityChecker and ValidityCheckerDistance won't work")
    # raise ImportError("Run in a docker with ompl")

from models import gibsonWorld

obstacles, robot = gibsonWorld.set_env()

def SE2State2Tuple(state):
    '''
    convert SE2 state object to tuple
    :param state: An SE2 object
    :return tuple: (x,y, theta) 
    '''
    return (state.getX(), state.getY(), state.getYaw())


# Set up planning threshold limits
thresh = 0.01
N = stats.norm(scale=np.sqrt(1/2))
c = N.ppf(1-thresh)

m = gibsonWorld.get_model_KF(robot, obstacles, gibsonWorld)


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
        if time>1500:
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
    
    return path, path_interpolated, success


def plan_path():
    start = ob.State(dubinSpace)
    start[0] = 6.5
    start[1] = 6.5
    start[2] = -3*np.pi/4

    goal = ob.State(dubinSpace)
    goal[0] = 4 #2.4
    goal[1] = 2 #2.5
    goal[2] = -np.pi #3*np.pi/4
    path, path_interpolated, success = get_path(start, goal)
    path_param = {'path':path, 'path_interpolated':path_interpolated, 'success': success}

    if GP_check:
        exp = 'ccgp'
    else:
        exp = 'rrt_star'
    pickle.dump(path_param, open('/root/data/gibson_path_{}.p'.format(exp), 'wb'))


steering = [0, 2]

def control_robot():
    '''
    Intialize the control of the robot.
    '''
    targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -50, 50, 0)
    maxForceSlider = p.addUserDebugParameter("maxForce", 0, 50, 20)
    steeringSlider = p.addUserDebugParameter("steering", -1, 1, 0)
    while True:
        maxForce = p.readUserDebugParameter(maxForceSlider)
        targetVelocity = p.readUserDebugParameter(targetVelocitySlider)
        steeringAngle = p.readUserDebugParameter(steeringSlider)
        #print(targetVelocity)

        for wheel in gibsonWorld.wheels:
            p.setJointMotorControl2(robot,
                                    wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=targetVelocity,
                                    force=maxForce)

        for steer in steering:
            p.setJointMotorControl2(robot, steer, p.POSITION_CONTROL, targetPosition=-steeringAngle)

        steering
        print(gibsonWorld.get_distance(obstacles, robot))
        p.stepSimulation()
        time.sleep(0.01)

if __name__ == "__main__":
    plan_path()
    # control_robot()
    # pass
    


