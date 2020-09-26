'''Guarantee collision free path between adjacent waypoints
by investigating a sphere encapsulating the start and goal 
location. 
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

fig,ax = plt.subplots(1,3)
sns.set()

ax[0].set_xlim((-0.2, 10.2))
ax[0].set_ylim((-0.2, 10.2))

# Initialize the position of obstacles
dimensions = [box_length, box_width]
rectangle_corner = np.r_[(-dimensions[0]/2, -dimensions[1]/2)]

for xy_i in point.xy_circle:
    plt_cir = plt.Circle(xy_i, radius=0.2, color='r')
    ax[0].add_patch(plt_cir)

for xy_i in point.xy:
    plt_box = plt.Rectangle(xy_i+rectangle_corner, dimensions[0], dimensions[1], color='r')
    ax[0].add_patch(plt_box)

# # Plot variances
# eig, eig_V = np.linalg.eig(K_inv)
# k_x_x = m.kern.K(p1)
# @np.vectorize
# def var(r, theta):
#     v = np.r_[r*np.cos(theta), r*np.sin(theta)]
#     k_x_X = m.kern.K(p1+v, m.X)
#     return k_x_x - k_x_X@K_inv@k_x_X.T

# theta = np.linspace(0.0, np.pi*2, 180)
# radius = np.linspace(0, r, 100)
# grid_r, grid_theta = np.meshgrid(radius, theta)
# var_sam = var(grid_r, grid_theta)

# fig = plt.figure()
# ax = plt.subplot(projection="polar")
# im = ax.pcolormesh(theta, radius, var_sam.T)
# plt.colorbar(im)
# ax.plot([0, 1], np.r_[var2, var2])