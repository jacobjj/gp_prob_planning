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
    print("Using model from sampled distance points")
    samples = 1000
    X = np.random.rand(samples, 2)*10
    robotOrientation = p.getQuaternionFromEuler([0., 0., 0.])
    Y = np.zeros((samples, 1))
    for i in range(samples):
        p.resetBasePositionAndOrientation(robot, np.r_[X[i, :], 0.1], robotOrientation)
        Y[i] = model.get_distance(obstacles, robot)

    # Add noise to the distance measure
    Y = Y + np.random.normal(scale=10**(-0.5),size=(samples,1))
    # kernel = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)
    kernel = GPy.kern.RBF(input_dim=2, variance=1)
    m = GPy.models.GPRegression(X,Y,kernel)
    m.optimize()
    return m

def get_path(A, B, M_n, N_n, step_size=10):
    '''
    Returns a random walk path with step_size, along with noisy
    observations.
    :param A,B : The linear state of the model
    :param N_n: A scipy.stats random variable representing the observation noise
    :param M_n: A scipy.stats random variable representing the motion noise
    :param step_size: Number of integer steps
    :returns tuple: A tuple with a list of (true_states, observations)
    '''
    x = np.random.rand(2)*10
    path = [x]
    obs = [x+N_n.rvs()]
    control = []
    steps = 0 
    while steps<step_size:
        u = np.random.rand(2)*1.5 - 0.75
        x_temp = A@x + B@u + M_n.rvs()
        if any(x_temp>=10) or any(x_temp<=0):
            continue
        x = x_temp
        steps+=1
        path.append(x)
        obs.append(x+N_n.rvs())
        control.append(u)
    return path, control, obs

def get_state_est(A, B, M, N):
    '''
    Return a function to estimate the state using KF.
    :param A, B : The linear parameters of the motion model  
    :param M, N : The motion and observation uncertainty. 
    :returns func: A function to evaluate the state estimate
    ''' 
    # TODO: The P should be a parameter
    def x_hat( x_est, control, obs, P):
        '''
        :param x_est: The previous state estimate
        :param control: The current control
        :param obs: The current observatoin
        :returns tuple: A tuple of the current state estimate and Covariance matrix
        '''
        x_bar = A@x_est + B@control
        P_bar = A@P@A.T + M
        K = P_bar@np.linalg.inv(P_bar+N)
        x_est = x_bar + K@(obs-x_bar)
        P = (np.eye(2)-K)@P_bar
        return x_est, P
    return x_hat


def get_path_est(A, B, M, N, path, control, obs):
    '''
    Get the path estimate using Kalman-filter
    :param A,B : The linear Co-efficient of motion model.
    :param M: The covariance matrix of motion noise model
    :param N: The covariance matrix of observation noise model
    :param path: A list of states representing a path.
    :param control: A list of control's sequence
    :param obs: A sequence of observation points.
    '''
    P = np.eye(2)*1e-1
    x_est = path[0]
    path_est = [x_est]
    x_hat = get_state_est(A, B, M, N)
    for c,obs in zip(control, obs):
        x_est, P = x_hat(x_est, c, obs, P)
        path_est.append(x_est)
    return path_est


def get_model_KF(A, B, M_n, N_n, robot, obstacles, model):
    '''
    Getting the GP model using the KF.
    :param A,B : The linear state of the model
    :param N_n: A scipy.stats random variable representing the obeservation noise
    :param M_n: A scipy.stats random variable representing the motion noise
    :param robot: pybullet.MultiBody object representing the robot
    :param obstacles: pybullet.MultiBody object representing the obstacles in the environment
    :param model: custom function found in folder "models" of the dynamic system
    :returns GPy.models.GPRegression model representing the obstacle space
    '''
    try:
        X = np.load('X.npy')
        Y = np.load('Y.npy')
        print("Loading data for map")
    except FileNotFoundError:
        print("Could not find data, gathering data")
        X = np.array([[0.0, 0.0]])
        Y = np.array([0.0])
        robotOrientation = (0.0, 0.0, 0.0, 1.0)
        for _ in range(5):
            path, control, obs = get_path(A, B, M_n, N_n, step_size=1000)
            path_est = get_path_est(A, B, M_n.cov, N_n.cov, path, control, obs)
            Y_temp = []
            for path_i in path:
                p.resetBasePositionAndOrientation(robot, np.r_[path_i, 0.1], robotOrientation)
                Y_temp.append(model.get_distance(obstacles, robot))
            Y = np.r_[Y, Y_temp]
            X = np.r_[X, np.array(path_est)]
        np.save("X.npy", X)
        np.save("Y.npy", Y)

    print("Using model from state-estimated points")
    kernel = GPy.kern.RBF(input_dim=2, variance=1)
    try:
        # Ignore the first row, since it is just filler values.
        m = GPy.models.GPRegression(X[1:,:], Y[1:,None], kernel)
        m.update_model(False)
        m.initialize_parameter()
        m[:] = np.load('env_5_param.npy')
        print("Loading saved model")
        m.update_model(True)
    except FileNotFoundError:
        print("Could not find trained model")
        m = GPy.models.GPRegression(X[1:,:], Y[1:,None], kernel)
        m.optimize()
        np.save('env_5_param.npy', m.param_array)
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

def return_collision_deterministic_line(x_mu, x_sigma, u, m, A, B):
    '''
    Return the minimum value of mu/(sqrt(2)*sigma) for the GP, along the 
    trajectory of the control input.
    :param x_mu: The mean of the initial state
    :param x_sigma: The variance of the input state
    :param u: The control input
    :param m: The GP model of the obstacles
    :param A,B: The parameters of the LTI system. x_t_1 = Ax_t + B_u
    :return float: The maximum value along the trajectory with input u.
    '''
    assert x_mu.shape[0]==3,"The dimension of axis 1 has to be 3"
    assert u.shape[0] == 3, "The control input axis 1 has to be 3"
    x_mu_t_1 = A@x_mu + B@u
    alpha = np.linspace(0,1,100)
    traj = (1-alpha)*x_mu[:,None] + alpha*x_mu_t_1[:,None]
    f_mu, f_sigma = m.predict(traj[:2,:].T, kern=m.kern)
    # import pdb; pdb.set_trace()
    return min(f_mu/(np.sqrt(2*f_sigma)))

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