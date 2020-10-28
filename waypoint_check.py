'''Guarantee collision free path between adjacent waypoints 
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
np.random.seed(8)

obstacles, robot = point.set_env()

thresh = 0.01
N = stats.norm(scale=np.sqrt(1/2))
c = N.ppf(1-thresh)

# Define gaussian model
m = get_model(robot, obstacles, point)
# Define LTI system
A,B, _, _ = point.get_dyn()

fig,ax = plt.subplots(1,2)
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

# Plot the threshold
ax[1].plot([0,1], [c, c,], color='k')

# Visualize how lines on the map translates to distance functions
samples = 5

# TODO: Not the best way to write this code!!
# TODO: Inversion should be done using cholesky factorization
# TODO: Inverse of K should be stored in memeory for better data management
true_K = m.kern.K(m.X)
# NOTE: The prior noise of f(x)
prior_var = 1e-6
K_inv = np.linalg.inv(true_K+np.eye(true_K.shape[0])*prior_var)
weights = K_inv @ m.Y
k_x_x = m.kern.K(np.c_[0,0])

# Number of divides
n = 1024#8192
delta = 1/(n-1)

def get_GP_F(start, goal):
    '''
    The guassian function from x_start and x_goal.Return the function for line
    :param start:  A np.array representing the start position
    :param goal: A np.array representing the goal position
    :return function: Return the function of F along the line
    '''
    def F(alpha, *args):
        '''
        The function F
        '''
        if alpha<0:
            alpha=0
        elif alpha>1:
            alpha = 1
        x_hat = (1-alpha)*start + alpha*goal
        k_star = m.kern.K(x_hat, m.X)
        var = k_x_x-k_star@K_inv@k_star.T
        return ((k_star@weights)/np.sqrt(2*var))[0]
    return F

def get_distance(start, goal):
    '''
    Get the closest L2 distance between the training samples and the 
    line segment joinining start and goal.
    :param start: np.narray representing the start point
    :param goal: np.narray representing the goal point
    :return np.narry: An array that returns the shortest distance between the
    training point and line segment between start and goal.
    '''
    delta_p = start-goal
    delta_line = start - m.X
    # if u is between 0 and 1, there exists a perpendicular line
    # from x to the line segment.
    u = (delta_line@delta_p.T)/np.linalg.norm(delta_p)**2
    # Find the perpendicular distance to line
    slope = delta_p[0,1]/delta_p[0,0]
    c = start[0,1]-slope*start[0,0]
    A = np.r_[-slope,1]
    d = np.abs(m.X@A-c)/np.linalg.norm(A)

    # If a perpendicular segment does not exit from a point to the line
    # segment, then measure the distance from the start/goal points.    
    d[u[:,0]<=0] = np.linalg.norm(m.X[u[:,0]<=0,:]-start, axis=1)
    d[u[:,0]>=1] = np.linalg.norm(m.X[u[:,0]>=1,:]-goal, axis=1)

    return d


def get_var(start, goal, alpha):
    '''
    Get the variance along a line segment represented by 
    start*(1-alpha) + alpha*goal
    :param start: An np.ndarray representing the start postion.
    :param goal: An np.ndarray represeenting the goal position.
    :param alpha: The point on the line
    :returns float: The variance of the posterior at alpha.
    '''
    x_hat = start*(1-alpha)+goal*alpha
    k_star = m.kern.K(x_hat, m.X)
    return (k_x_x - k_star@K_inv@k_star.T)[0]

def get_minimum_var(start, goal):
    '''
    TODO : Need to prove this theoretically that this is the minimum
    Get the minimum variance along a straight line by searching the minimum
    value at each of the training points that is close to the line segment.
    :param start: An np.ndarray representing the start position
    :param goal: An np.ndarray representing the goal position
    :returns float : The minimum variance along the trajectory.
    '''

    sol = optimize.minimize_scalar(
        lambda alpha: get_var(start,goal,alpha), 
        method='bounded', 
        bounds=[0,1]
        )
    if sol.success:
        return sol.fun[0]
    raise RuntimeError("There is something wrong with the variance function!!")

def get_maximum_var(start, goal):
    '''
    Get the maximum variance along as trajectory.
    :param start: An np.ndarray representing the start position.
    :param goal: An np.ndarray representing the goal position.
    :returns float: The maximum variance along the trajectory.
    '''

    sol = optimize.minimize_scalar(
        lambda alpha: -get_var(start, goal, alpha), 
        method='bounded', 
        bounds=[0,1]
        )
    if sol.success:
        return -sol.fun[0]
    raise RuntimeError("There is something wrong with the variance function!!")


def get_GP_lb_F(start, goal):
    '''
    Generate F using the sum of all lower bounds
    :param data: A np.array for the data
    :param start:  A np.array representing the start position
    :param goal: A np.array representing the goal position
    :parma K: The number of points to check
    :return function: Return the lower bound of function of F
    '''
    # Minimum sigma for the given line segment
    var_min = get_minimum_var(start, goal)

    # Maximimum sigma for the given line segment
    var_max = get_maximum_var(start, goal)

    sigma_hat = np.ones((K_inv.shape[0],1))*np.sqrt(2*var_min)
    sigma_hat[weights[:,0]>0] = np.sqrt(2*var_max)
    
    # thresh = np.linalg.norm(start - goal)
    # sigma_hat = np.sqrt(2*var_max)
    def F(alpha):
        '''
        The lower bound function F.
        '''
        x_hat = (1-alpha)*start + alpha*goal
        k_star = m.kern.K(x_hat, m.X)
        return k_star@(weights/sigma_hat) 

    return F


for _ in range(samples):
    c = np.random.rand(3,)
    x_start, y_start = np.random.rand(2)*10
    # # Generate a point at r-distance away
    # r=4
    # theta = np.random.rand()*np.pi*2
    # x_goal, y_goal = x_start + r*np.cos(theta), y_start + r*np.sin(theta)
    # Generate a random point
    x_goal, y_goal = np.random.rand(2)*10
    print("Start: ({}, {}) , Goal : ({}, {})".format(x_start, y_start, x_goal, y_goal))
    
    # Find the distance to projection
    p1 = np.c_[x_start, y_start]
    p2 = np.c_[x_goal, y_goal]

    a = np.linspace(0,1,100)
    points = np.c_[
        (1-a)*x_start+a*x_goal,
        (1-a)*y_start+a*y_goal
    ]

    ax[0].plot([x_start,x_goal],[y_start,y_goal], c=c)

    # Predict with noise
    # f_mu, f_var = m.predict(points)
    # ax[1].plot(np.linspace(0,1,100), f_mu/np.sqrt(2*f_var), c=c)

    # Plot the true values
    F = get_GP_F(p1, p2)
    F_v = np.squeeze([F(a) for a in np.linspace(0,1,100)])
    ax[1].plot(np.linspace(0,1,100), F_v, c=c)

    sol = optimize.shgo(F, bounds =[(0,1)], iters=10)
    # Simplical homology
    if sol.success:
        ax[1].scatter(sol.x, sol.fun, label='shgo', c='g')
    else:
        print("No global solution found")

    # # Basinhopping
    # sol = optimize.basinhopping(F, np.array(0.5))
    # ax[1].scatter(sol.x, sol.fun, label='basinhopping', c='b')

    # # Dual Annealing
    # sol = optimize.dual_annealing(F, bounds=np.array([[0,1]]))
    # ax[1].scatter(sol.x, sol.fun, label='dual annealing', c='k')
    # plt.legend()
    # # Plot the initial bounds
    # F = get_GP_lb_F(p1, p2)
    # F_lb = np.squeeze([F(a) for a in np.linspace(0,1,100)])
    # ax[1].plot(np.linspace(0,1,100), F_lb, '--', c='g')    
    # fig_var, ax_var = plt.subplots()

    # for alpha in np.linspace(0,1,n)[:-1]:
    #     s = p1*(1-alpha)+ p2*alpha
    #     g = p1*(1-alpha-delta) + p2*(alpha+delta)
    #     F = get_GP_lb_F(s, g)
    #     F_lb = np.squeeze([F(a) for a in np.linspace(0,1,10)])
    #     # ax[1].plot(np.linspace(alpha, alpha+delta, 10), F_lb, '--', c='k')
    #     ax[1].scatter(alpha, F(0.5), marker='x', color='r', alpha=0.5)
    #     max_var = get_maximum_var(s, g)
    #     min_var = get_minimum_var(s, g)
    #     var = get_var(s, g, alpha+1/(2*n))
    #     ax_var.plot([alpha, alpha+delta], [max_var, max_var], '--', color='g', label='max variance')
    #     ax_var.plot([alpha, alpha+delta], [min_var, min_var], '--', color='r', label='min variance')
    #     ax_var.plot([alpha, alpha+delta], [var, var], color='k', label='variance')

ax[0].scatter(m.X[:,0], m.X[:,1], alpha=0.25)

def plot_bound(start, end):
    '''
    Plot the points near the line segment between start and goal.
    :param start: The start position
    :param goal: The goal position
    '''
    delta_p = start-end
    delta_line = start - m.X
    u = (delta_line@delta_p.T)/np.linalg.norm(delta_p)**2
    # Find the perpendicular distance to line
    slope = delta_p[0,1]/delta_p[0,0]
    c = start[0,1]-slope*start[0,0]
    A = np.r_[-slope,1]
    d = np.abs(m.X@A-c)/np.linalg.norm(A)

    # Find the closest points to the line
    dist_data = zip(range(d.shape[0]),u, d)
    dist_sorted = sorted(dist_data, key=lambda x:x[2] if x[1]>=-0.1 and x[1]<=1.1 else 20)
    pick_index = [data[0] for data in dist_sorted[:100]]
    ax[0].scatter(start[0,0], start[0,1], color='r')
    ax[0].scatter(m.X[pick_index,0],m.X[pick_index,1],color='g',alpha=0.5)

# NOTE: use m.kern.K_of_r() --> for getting k* or use m.kern.K itself !!
# NOTE: use m.kern.K(X) --> to get the Gram matrix
# NOTE: use m.Y --> for getting f
# NOTE: use m.X --> for getting X
# Find the value of minimum c0

def get_var_training_sample(i):
    '''
    Get the variance at the training point.
    :param i: the index of the training point
    :returns float: The posterior variance at that point
    '''
    x_hat = m.X[i,:].reshape((1,2))
    k_star = m.kern.K(x_hat, m.X)
    return (k_x_x - k_star@K_inv@k_star.T)[0]

# var_min = get_minimum_var(p1, p2)
# ax.plot([0, 1], [var_min, var_min], '-.', color='r')
# var_max = get_maximum_var(p1, p2)
# ax.plot([0, 1], [var_max, var_max], '-.', color='g')
# ax.plot(np.linspace(0,1,100), np.ones(100)*prior_var, '.', color='k')
# n = 64
# delta = 1/(n-1)
# for alpha in np.linspace(0,1,n)[:-1]:
#     s = p1*(1-alpha)+ p2*alpha
#     g = p1*(1-alpha-delta) + p2*(alpha+delta)
#     var_min = get_minimum_var(s, g)
#     var_max = get_maximum_var(s, g)
#     ax.plot([alpha, alpha + delta], [var_min, var_min], '--', color='r')
#     ax.plot([alpha, alpha + delta], [var_max, var_max], '--', color='g')

def get_F(start, goal, i):
    '''
    Get the loweights[i,0]er bound function for the i^th element of F, between start, and goal.
    :param start: An np.ndarray representing the starting position.
    :param goal: An np.ndarray representing the goal position.
    :param i: F_i to plot.
    :reuturns function: The lower bound function of F_i
    '''
    min_var = get_minimum_var(start, goal)
    max_var = get_maximum_var(start, goal)
    def Fl(alpha):
        x_hat = (1-alpha)*start + alpha*goal 
        k_star = m.kern.K(x_hat, m.X)
        var = k_x_x - k_star@K_inv@k_star.T 
        num = weights[i,0]*k_star[0,i] 
        return num/np.sqrt(2*var), num/np.sqrt(2*min_var) 
    def Fu(alpha):
        x_hat = (1-alpha)*start + alpha*goal
        k_star = m.kern.K(x_hat, m.X)
        var = k_x_x - k_star@K_inv@k_star.T
        num = weights[i,0]*k_star[0,i]

        return num/np.sqrt(2*var), num/np.sqrt(2*max_var)
    if weights[i,0]>0:
        return Fu
    return Fl        

def plot_lines(i, ax): 
    # Plot for each interval
    for alpha in np.linspace(0,1,n)[:-1]:
        s = p1*(1-alpha)+ p2*alpha
        g = p1*(1-alpha-delta) + p2*(alpha+delta)
        F = get_F(s, g, i)
        # f = np.squeeze([F(a) for a in np.linspace(0,1,10)])
        # ax.plot(np.linspace(alpha, alpha+delta,10), f[:,0], color='g', label='F(x)')
        # ax.plot(np.linspace(alpha, alpha+delta,10), f[:,1], '--', color='r', label='~F(x)') 

        ax.scatter(alpha, F(alpha)[0], color='g', label='F(x)')
        ax.scatter(alpha, F(alpha)[1], color='r', label='~F(x)', alpha=0.5)

# fig, ax = plt.subplots(3,3)
# # Positive weights
# positive_weights, = np.where(weights[:,0]>0)
# tmp = positive_weights[:9].reshape((3,3))
# # Highest weights
# sorted_weights_index = np.argsort(np.abs(weights[:,0]))
# tmp = sorted_weights_index[-9:].reshape((3,3))ee
# for i in range(3):
#     for j in range(3):
#         plot_lines(tmp[i][j], ax[i,j])