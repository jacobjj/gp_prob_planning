'''Guarantee collision free path between adjacent waypoints 
'''
from config import box_width, box_length, xy

import GPy
import numpy as np
import pybullet as p
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sparse_rrt.systems.system_interface import BaseSystem

from models import point
from gp_model import get_model, return_collision_prob, return_collision_prob_line

obstacles, robot = point.set_env()

thresh = 0.01
N = stats.norm(scale=np.sqrt(1/2))
c = N.ppf(1-thresh)

# Define gaussian model
m = get_model(robot, obstacles, point)
# Define LTI system
A,B = point.get_dyn()

fig,ax = plt.subplots(1,2)
sns.set()

ax[0].set_xlim((-0.2, 10.2))
ax[0].set_ylim((-0.2, 10.2))

# Initialize the position of obstacles
dimensions = [box_length, box_width]
rectangle_corner = np.r_[(-dimensions[0]/2, -dimensions[1]/2)]  

for xy_i in xy:
    plt_box = plt.Rectangle(xy_i+rectangle_corner, dimensions[0], dimensions[1], color='r')
    ax[0].add_patch(plt_box)

# Visualize how lines on the map translates to distance functions
samples = 10
alpha = np.linspace(0,1, 100)
ax[1].plot([0,99], [c, c,], color='k')
np.random.seed(5)
for _ in range(samples):
    c = np.random.rand(3,)
    x_start, y_start = np.random.rand(2)*10
    x_goal, y_goal = np.random.rand(2)*10
    print("Start: ({}, {}) , Goal : ({}, {})".format(x_start, y_start, x_goal, y_goal))

    points = np.c_[
        (1-alpha)*x_start+alpha*x_goal,
        (1-alpha)*y_start+alpha*y_goal
    ]

    # Predict with noise
    # f_mu, f_sigma = m.predict(points)
    
    # Predict without noise
    f_mu, f_sigma = m.predict_noiseless(points, kern=m.kern)

    ax[0].plot([x_start,x_goal],[y_start,y_goal], c=c)
    ax[1].plot(np.arange(100), f_mu/(np.sqrt(2)*f_sigma), c=c)

    # Set up the optimization problem
    # NOTE: Adding this jitter is important, to prevent negative eigenvalues
    # while inverting the matrix
    K = m.kern.K(m.X)
    jitter = np.mean(np.diag(K))*1e-6
    K_inv = np.linalg.inv(K+np.eye(K.shape[0])*jitter)
    weights = K_inv @ m.Y

    def F_init(start, goal):
        '''
        The guassian function from x_start and x_goal
        :param x_start : The start position
        :param x_goal : The goal position
        :returns function: A function of alpha
        '''
        # TODO: Check if this function can mimic predict_noiseless function
        def F(alpha):
            '''
            The function F
            '''
            x_hat = (1-alpha)*start + alpha*goal
            k_star = m.kern.K(x_hat, m.X)
            sigma = m.kern.K(x_hat, x_hat)-k_star@K_inv@k_star.T
            # return -(np.abs(k_star)@weights)*(1+0.5*k_star@K_inv@k_star.T)
            return (k_star@weights)/np.sqrt(2*sigma)
        return F

    F = F_init(np.c_[x_start, y_start], np.c_[x_goal, y_goal])
    F_lb = np.squeeze([F(a) for a in np.linspace(0,1,100)])
    # ax[1].plot(np.arange(100), F_lb,'--',c=c)
# NOTE: use m.kern.K_of_r() --> for getting k* or use m.kern.K itself !!
# NOTE: use m.kern.K(X) --> to get the Gram matrix
# NOTE: use m.Y --> for getting f
# NOTE: use m.X --> for getting X
# Find the value of minimum c0