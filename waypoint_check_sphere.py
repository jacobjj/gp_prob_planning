'''Guarantee collision free path between adjacent waypoints
by investigating a sphere encapsulating the start and goal 
location. 
'''
from config import box_width, box_length, xy

import GPy
import numpy as np
import pybullet as p
from scipy import stats
from scipy import optimize
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

# Define bound for trajectory
L = np.linalg.norm(A)

fig,ax = plt.subplots(1,3)
sns.set()

ax[0].set_xlim((-0.2, 10.2))
ax[0].set_ylim((-0.2, 10.2))

# Initialize the position of obstacles
dimensions = [box_length, box_width]
rectangle_corner = np.r_[(-dimensions[0]/2, -dimensions[1]/2)]

true_K = m.kern.K(m.X)
# NOTE: The prior noise of f(x)
prior_var = 1e-6
K_inv = np.linalg.inv(true_K+np.eye(true_K.shape[0])*prior_var)
weights = K_inv @ m.Y
k_x_x = m.kern.K(np.c_[0,0])

for xy_i in point.xy_circle:
    plt_cir = plt.Circle(xy_i, radius=0.2, color='r')
    ax[0].add_patch(plt_cir)

for xy_i in point.xy:
    plt_box = plt.Rectangle(xy_i+rectangle_corner, dimensions[0], dimensions[1], color='r')
    ax[0].add_patch(plt_box)


def F(x, *args):
    '''
    The guassian function.
    :param x: The state of the robot
    :returns float: E[g(x)]/sqrt(2var(g(x)))
    '''
    if x.ndim!=m.X.ndim:
        x = x[None, :]
    k_star = m.kern.K(x, m.X)
    var = k_x_x-k_star@K_inv@k_star.T
    return ((k_star@weights)/np.sqrt(2*var))[0]

# Plot the bounding balls for trajectory
samples = 5
for i in range(samples):
    x_start, y_start = np.random.rand(2)*10
    p1 = np.r_[x_start, y_start, 0]
    u = np.random.rand(3,1)

    r0 = np.linalg.norm(A@p1 + B@u)/L
    theta = np.linspace(0, 2*np.pi, 20)
    points = np.c_[np.cos(theta), np.sin(theta)]

    t = 0.1
    plot_circle = p1[:2] + r0*(np.exp(t)-1)*points
    ax[0].plot(plot_circle[:,0], plot_circle[:,1], '--')

    ri = r0*(np.exp(t)-1)
    p1_min, p1_max = p1-ri, p1+ri
    sol = optimize.shgo( 
        F,  
        bounds=[(p1_min[0],p1_max[0]),(p1_min[1],p1_max[1])], 
        constraints = [
                        {
                            'type':'ineq', 
                            'fun':lambda x: ri - np.linalg.norm(x-p1[:2]), 
                            'jac': lambda x:-(x-p1[:2])
                        }
                    ]
        )
    if sol.fun>c:
        print("{} Safe region".format(i))
    else:
        print("Likely to hit obstacles")