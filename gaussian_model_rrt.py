# A GP-model for collision depth information

import pybullet as p
import pybullet_data

import GPy
import numpy as np
from scipy import stats

import seaborn as sns

from itertools import compress
import matplotlib.pyplot as plt

from models import point
from gp_model import get_model, return_collision_prob_line, return_collision_prob

GPy.plotting.change_plotting_library("plotly_offline")
obstacles, robot = point.set_env()

# Get the GP model
m = get_model(robot, obstacles, point)
# m.optimize_restarts(num_restarts = 10)
# from IPython.display import display
# display(m)
# fig = m.plot(plot_density=True,legend=True)
# GPy.plotting.show(fig, filename='basic_gp_regression_notebook_optimized')

# Define robot model
A, B = point.get_dyn()

# Control input - RV
U = stats.uniform()

# Simple RRT-planners
start = np.r_[0.0, 0.0, 0.1]
robotOrientation = p.getQuaternionFromEuler([0., 0., 0.])
p.resetBasePositionAndOrientation(robot, start, robotOrientation)

goal = np.r_[9.0, 9.0, 0.1]

# NOTE: currently did not see a straight way to integrate probability checks in OMPL
# TODO: One way is to set up a motion validation scheme that details whether it is possible
# to reach from one state to the other.

start_node = {
    'mean':start, 
    'var':np.eye(3)*0.1, 
    'parent':None, 
    'control':None
    }
node_list = [start_node]

goal_reach = False
X = stats.uniform()

col_thresh = 0.05 # 1% of chance of collision
fig, ax = plt.subplots()
def plot_point(ax, point, color):
    '''
    Plot the point on the given axis
    :param ax: The axis to which the point should be plotted
    :param point: The point to be plotted
    :param color: The color of the plot
    '''
    ax.scatter(point[0], point[1], color = 'g')


# fig.show()

while not goal_reach:
    # Sample control
    u = U.rvs(size=(3,))
    # Search for the nearest nodes and probability check
    dlist = [
        np.linalg.norm(node['mean']-A@node['mean']-B@u) 
        for node in node_list]
    # pColList = [
    #     return_collision_prob(node['mean'], node['var'], u, m, A, B)<=col_thresh 
    #     for node in node_list
    #     ]
    pColList = [
        return_collision_prob_line(node['mean'], node['var'], u, m, A, B)<=col_thresh 
        for node in node_list
        ]
    # Find all the projects that satisfy collision probability
    data_points = compress(zip(range(len(node_list)), dlist), pColList)
    
    # The parent of the node has to be the point from which the projection
    # is made not the minimum point.
    # Observation : RRT is too slow, need to use something faster, like SST
    for data_point in data_points:
        parent_node = data_point[0]
        # Check if the points satisfy bound conditions
        new_node_mean = A@node_list[parent_node]['mean']+B@u
        if all(new_node_mean<=[10.1]*3) and all(new_node_mean>=[-0.1]*3):
            new_node = {
                'mean' : new_node_mean,
                'var' :  A@node_list[parent_node]['var']@A.T,
                'parent': parent_node,
                'control': u
                }
            node_list.append(new_node)
            # plot_point(ax, new_node_mean, 'k')
            # edge = np.c_[node_list[parent_node]['mean'],new_node_mean].T
            # ax.plot(edge[:,0], edge[:, 1])
            # Check for goal reached
            if np.linalg.norm(new_node['mean'] - goal)<=1:
                goal_reach = True
    # fig.show()

    if len(node_list)>1e3:
        break
    else:
        print(len(node_list))

fig, ax = plt.subplots(figsize=(10,10))
sns.set()
ax.set_xlim((-0.2, 10.0))
ax.set_ylim((-0.2, 10.0))

plt_cir = plt.Circle(np.r_[0.0, 0.0], radius = 0.1)

plt_obst = [
    plt.Rectangle(xy_i+np.r_[0.2, 0.2], 0.4, 0.4, color='r')
    for xy_i in point.xy
    ]

ax.add_patch(plt_cir)
for plt_box in plt_obst:
    ax.add_patch(plt_box)

plot_point(ax, start, 'r')
plot_point(ax, goal, 'g')

if goal_reach:
    final_path = []
    last_node = node_list[-1]
    plot_point(ax, last_node['mean'], 'k')
    edge = np.c_[goal,last_node['mean']].T
    ax.plot(edge[:,0], edge[:, 1], color='b')

    while last_node['parent'] is not None:
        final_path.append({'x':last_node['mean'], 'u':last_node['control']})
        edge = np.c_[node_list[last_node['parent']]['mean'],last_node['mean']].T
        ax.plot(edge[:,0], edge[:, 1], color='b')
        last_node = node_list[last_node['parent']]
        plot_point(ax, last_node['mean'], 'k')

else:
    print("Goal not reached")