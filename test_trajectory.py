''' Track the trajectory of the robot
'''
import pybullet as p
import numpy as np
import pickle
import time

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from scipy import stats

from models import racecar
from config import box_width, box_length, xy, cir_radius

obstacles, robot = racecar.set_env()
dt = 1/240
wheelbase = 0.325

fig, ax = plt.subplots()

def plot_env(ax, x, y, theta):
    '''
    Plot the environment and state estimate.
    :param ax: The matplotlib.pyplot.axis object
    :param x: The x dimension
    :param y: The y dimension
    :param theta: The theta dimension
    '''
    ax.cla()
    ax.set_xlim((-0.2, 10.2))
    ax.set_ylim((-0.2, 10.2))

    # Initialize the position of obstacles
    dimensions = [box_length, box_width]
    rectangle_corner = np.r_[(-dimensions[0]/2, -dimensions[1]/2)]  

    for xy_i in racecar.xy_circle:
        plt_cir = plt.Circle(xy_i, radius=cir_radius, color='r')
        ax.add_patch(plt_cir)

    for xy_i in racecar.xy:
        plt_box = plt.Rectangle(xy_i+rectangle_corner, dimensions[0], dimensions[1], color='r')
        ax.add_patch(plt_box)
    plt.arrow(x, y, 0.4*np.cos(theta), 0.4*np.sin(theta), width=0.05)


if __name__=="__main__":
    num = 1
    path_param = pickle.load(open('/home/jacoblab/prob_planning_data/dubins/CCGP-MP/exp4/path_{}.p'.format(num), 'rb'))
    
    traj = path_param['path_interpolated']

    x, y, theta, _, _ = racecar.get_state(robot)
    x_est = np.r_[x, y, theta, 1e-3, 0.0]

    goal_reached = False
    Q = np.eye(4)*5
    R = np.eye(1)
    P = np.eye(5)*0
    goal= traj[-1,:]
    num = 0
    while not goal_reached and num<1e4:
        ind, e = racecar.calc_nearest_index(x_est, traj)
        v = x_est[3]
        phi = x_est[4]
        theta = x_est[2]
        A = np.array([
            [1, 0., -v*np.sin(theta)*dt, np.cos(theta)*dt],
            [0, 1., v*np.cos(theta)*dt, np.sin(theta)*dt],
            [0, 0., 1., np.tan(phi)*dt/wheelbase],
            [0, 0., 0., 1.]
        ])

        B = np.zeros((4, 1))
        B[2, 0] = v*dt/(wheelbase*(np.cos(theta)**2+1e-2))

        K, _, eigvals = racecar.dlqr(A, B, Q, R)

        x = np.zeros((4, 1))
        th_e = racecar.pi_2_pi(x_est[2] - traj[ind, 2])
        x[0, 0] = x_est[0]-traj[ind, 0]
        x[1, 0] = x_est[1]-traj[ind, 1]
        x[2, 0] = th_e
        x[3, 0] = 0#20-v
        
        fb = racecar.pi_2_pi((-K @ x)[0, 0])
        delta = np.clip(fb, -np.pi*0.35, np.pi*0.35)
        
        # TODO : Explain why the control has to be negative??
        racecar.step(robot, 20, -delta)
        state, obs =  racecar.get_noisy_state(robot)
        x_est, P =  racecar.x_hat(robot, x_est, np.r_[-delta], obs, P)
        x, y, theta, phi, speed = racecar.get_state(robot)
    
        # if any((p.getContactPoints(robot, obs) for obs in obstacles)):
        #     # print("Collision")
        #     break
        if racecar.est_check_collision(x_est[0], x_est[1], x_est[2]):
            plot_env(ax, x_est[0], x_est[1], x_est[2])
            print("Collision")
            break

        if np.linalg.norm(x_est[:2]-goal[:2])<0.25:
            goal_reached = True
        num+=1