'''A set of utilities functions for modeliing Gaussian Process 
models
'''

import GPy
import numpy as np
import pybullet as p
from scipy import stats

# TODO: 1. Can switch out for a more fancy sampler
# TODO: 2. Estimate the best sequence of control inputs
def get_model(robot, obstacles, model):
    '''
    Samples closest distance from obstacles for different sample points
    :param robot: pybullet.MultiBody object representing the robot
    :param obstacles: pybullet.MultiBody object representing the obstacles in the environment
    :param model: custom function found in folder "models" of the dynamic system
    :returns GPy.models.GPRegression model representing the obstacle space
    '''
    # Define gaussian model
    samples = 500
    X = np.random.rand(samples, 2)*10
    robotOrientation = p.getQuaternionFromEuler([0., 0., 0.])
    Y = np.zeros((samples, 1))
    for i in range(samples):
        p.resetBasePositionAndOrientation(robot, np.r_[X[i, :], 0.1], robotOrientation)
        Y[i] = model.get_distance(obstacles, robot)

    # Add noise to the distance measure
    Y = Y + np.random.rand(samples,2)
    # kernel = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)
    kernel = GPy.kern.RBF(input_dim=2, variance=1)
    m = GPy.models.GPRegression(X,Y,kernel)
    m.optimize()
    return m


def return_collision_prob(x_mu, x_sigma, u, m, A, B):
    '''
    Return the probability of collision by sampling multiple samples
    from the gaussian distribution.
    :param x_mu: The current position mean
    :param x_sigma: The covariance of current position
    :param u: The control input
    :param m: Gaussian model for environment
    :param A: LTI system A 
    :param B: LTI system B
    :return float: The collision probability of the given point
    '''
    assert x_mu.shape[0]==3, "The dimension of axis 1 has to be 3"
    assert u.shape[0] == 3, "The control input axis 1 has to be 3"
    X_t = stats.multivariate_normal(mean = x_mu, cov=x_sigma)
    x_t = X_t.rvs(size = 100)
    x_t_1 = A@x_t.T + (B@u)[:, None]
    f_x_t_1_mu, f_x_t_1_sigma = m.predict(x_t_1.T)
    return np.mean(stats.norm.cdf(-f_x_t_1_mu/np.sqrt(f_x_t_1_sigma)))

def return_collision_prob_line(x_mu, x_sigma, u, m, A, B):
    '''
    Return the probablity of collision along a line.
    :param x_mu : The current position mean
    :param x_sigma : The covariance of current position
    :param u: The control input.
    :param m: Gaussian model for environment
    :param A: LTI system A 
    :param B: LTI system B
    :return float: The maximum collision probability along the trajectory
    '''
    assert x_mu.shape[0] == 3, "The dimension of axis 1 has to be 3"
    assert u.shape[0] == 3, "The dimension of axis 1 has to be 3, but the shape is {}".format(u.shape)

    return max(return_collision_prob(x_mu, x_sigma, u_i, m, A, B) for u_i in np.linspace(0, u, 100))