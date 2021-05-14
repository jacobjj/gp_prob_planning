'''Define a LTI point system
'''
from os import path as osp
import pybullet_utils.bullet_client as bc
import pybullet as pyb
import pybullet_data
import numpy as np

import GPy

from models.randomWorld import set_obstacles, RandomWorld

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    print("Could not find OMPL")
    print("execute_path() will not work!!")

from scipy import stats

from config import box_length, box_width, cir_radius

p = bc.BulletClient(connection_mode=pyb.DIRECT)
# p = bc.BulletClient(connection_mode=pyb.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")

p.resetDebugVisualizerCamera(
    cameraDistance=10, 
    cameraYaw=0, 
    cameraPitch=-89.9, 
    cameraTargetPosition=[5,5,0])

geomRobot = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.2)

def set_env(**kwargs):
    '''
    Set the environment up with the obstacles and robot.
    :param kwargs: Environment parameters
    :return tuple: The pybullet ID of obstacles and robot
    '''
    obstacles = set_obstacles(p, **kwargs)
    robot = p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=geomRobot, 
        basePosition=[0.0, 0.0, 0.1])
    return obstacles, robot


def get_observations(robot):
    '''
    Returns the sensor observations, similar to lidar data.
    :param robot: The robot ID
    '''
    num_rays = 8
    ray_length = 1
    robot_position, _ = p.getBasePositionAndOrientation(robot)
    ray_from = [robot_position]*num_rays
    ray_to  = [
        (robot_position[0]+ray_length*np.cos(theta), robot_position[1]+ray_length*np.sin(theta), 0.1) 
        for theta in np.linspace(0,2*np.pi, num_rays)
        ]
    results = p.rayTestBatch(ray_from, ray_to)
    # return the distance to obstacle for each ray
    return [result[2]*ray_length for result in results]


def get_distance(obstacles, robot):
    '''
    Get the shortest distance between the obstacles and robot, for the given 
    setup.
    :param obs: A list of all obstacle ID's
    :param robot: The robot ID
    :return float: The penetration distance between the robot and obstacles, If the 
    distance is negative, object is in collision.
    '''
    assert isinstance(obstacles, list), "Obstacles has to be a list"
    distance =  min(
            (
                p.getClosestPoints(bodyA=obs, bodyB=robot, distance=100)[0][8]
                for obs in obstacles
            )
        )
    return distance

# Define Robot model
A = np.eye(2)
B = np.eye(2)

M = np.eye(2)*1e-1 # Motion Noise
N = np.eye(2)*1e-2 # Observation Noise
def get_dyn():
    '''
    Return the dynamics of the LTI system represented by:
    x_t+1 = A@x_t + B@u_t + m_t, m_t ~ N(0, M)
    z_t+1 = x_t + n_t, n_t ~ N(0,N)
    :returns (A,B,M,N): where the parameters correspond to the above equation.
    '''
    return A, B, M, N

# KF state-estimator
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


def get_lqr(traj, C, D):
    '''
    Returns the LQR gains for a finite horizon system.
    :param traj: The trajectory to be followed.
    :param C : The cost for state.
    :param D : The cost for control.
    :return list: A list of LQR gains
    '''
    # Get LQR gains
    SN = C
    L_list = []
    # Evaluate L from the end
    for _ in traj[:-1]:
        L = -np.linalg.inv(B.T@SN@B+D)@B.T@SN@A 
        L_list.append(L)
        SN = C + A.T@SN@A + A.T@SN@B@L 
    return L_list

# Using infinite horizon lqr gains
def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 150
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T @ X @ A - A.T @ X @ B @ \
            np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        if (abs(Xn - X)).max() < eps:
            break
        X = Xn

    return Xn


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = np.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eigVals, eigVecs = np.linalg.eig(A - B @ K)

    return K, X, eigVals


def calc_nearest_index(state, traj):
    '''
    Calculate the closest index of trajectory path.
    '''
    d = np.linalg.norm(state[:2] - traj[:, :2], axis=1)
    ind = np.argmin(d)

    return ind


# Noise model
N_n = stats.multivariate_normal(cov=N)
M_n = stats.multivariate_normal(cov=M)
def execute_path(traj, C, D, si_check):
    '''
    Execute the path trajectory using LQR controller. Assumes error of P.
    :param traj: The path to follow.
    :param C, D: The LQR parameters
    :param si_check: An ompl.base.SpaceInformation with a valid collision checker.
    :returns (estimated state, real state, valid): A tuple containing the estimated state 
    and real state of the robot, and whether the path was completed or not.
    '''
    K, _ , eigvals = dlqr(A, B, C, D)

    # Define the parameters for the path.
    ompl_path_obj = og.PathGeometric(si_check)
    state_temp = ob.State(si_check.getStateSpace())
    start = traj[0]
    goal = traj[-1]

    P = np.eye(2)*0
    # L_list = get_lqr(traj, C, D)
    x = np.r_[start[0], start[1]] #+ stats.multivariate_normal(cov = P).rvs()
    x_real = x
    state_temp[0], state_temp[1] = x[0], x[1]
    ompl_path_obj.append(state_temp())
    path_est = [x]
    path_noisy = [x]
    done = False
    count = 0
    ind = 0
    # for i,T in enumerate(zip(traj[:-1], reversed(L_list))):
    while not done and count<1500:
        count += 1
        # x_t, L = T
        temp_ind = calc_nearest_index(x, traj)
        if temp_ind>ind:
            ind = temp_ind
        x_t = traj[ind]

        c = K@(x_t-x)
        c = np.clip(c, -0.75, 0.75)

        # delta_c = L@(x_real-x_t)
        # c = np.linalg.inv(B)@(traj[i+1]-A@x_t) + delta_c
        
        x_real = A@x_real + B@c + M_n.rvs()
        state_temp[0], state_temp[1] = x_real[0], x_real[1]
        ompl_path_obj.append(state_temp())

        z_real = x_real + N_n.rvs()
        x, P  = x_hat(x, c, z_real, P)
        path_est.append(x)
        path_noisy.append(x_real)
        # Check if goal is reached
        if np.linalg.norm(x-np.r_[goal[0], goal[1]])<0.5:
            done = True
            break
        if not ompl_path_obj.check():
            done = False
            break

    return path_est, path_noisy, done


def execute_traj_ignoreCol(traj):
    '''
    execute the trajectory by ignoring collison.
    :param traj: 
    return tuple: A tuple of estimated path and the target trajectory
    '''

    # Define the parameters for the path.
    start = traj[0]
    goal = traj[-1]
    ind = 1
    P = np.eye(2)*0
    x = np.r_[start[0], start[1]] #+ stats.multivariate_normal(cov = P).rvs()
    x_real = x
    C = np.eye(2)*10
    D = np.eye(2)*1e-1
    K, _ , eigvals = dlqr(A, B, C, D)
    path_est = [x]
    path_target = [x]

    done = False
    ind = 0
    count = 0
    while not done and count<1000:
        count +=1
        temp_ind = calc_nearest_index(x, traj)
        if temp_ind>ind:
            ind = temp_ind
        x_t = traj[ind]
        path_target.append(x_t)
        
        # Asymptotic Gains
        c = K@(x_t-x)     
        c = np.clip(c, -0.75, 0.75)

        # Take a step
        x_real = A@x_real + B@c + M_n.rvs()
        z_real = x_real + N_n.rvs()

        x, P  = x_hat(x, c, z_real, P)
        path_est.append(x)

        # Check if goal is reached
        if np.linalg.norm(x-np.r_[goal[0], goal[1]])<0.5:
            done = True
            break
        
    return path_est, path_target


def get_model_KF(robot, obstacles, samples, dataFolder=None):
    '''
    Getting the GP model using the KF. Generating points by executing random trajectories
    using controller.
    :param robot: pybullet.MultiBody object representing the robot
    :param obstacles: pybullet.MultiBody object representing the obstacles in the environment
    :param samples: The number of samples to use for the model
    :param dataFolder: The folder where the data and modelParams are stored
    :returns GPy.models.GPRegression model representing the obstacle space
    '''
    if dataFolder is None:
        dataFolder = 'param'
    seed = RandomWorld.seed
    try:
        print("Loading data for map")
        X = np.load(osp.join(dataFolder, f'X_{seed}.npy'))
        Y = np.load(osp.join(dataFolder, f'Y_{seed}.npy'))
    except FileNotFoundError:
        print("Could not find data, gathering data")
        X = np.array([[0.0, 0.0]])
        Y = np.array([0.0])
        robotOrientation = (0.0, 0.0, 0.0, 1.0)
        count = 0
        while count<samples:
            start = np.random.rand(2)*14-2
            goal = np.random.rand(2)*14-2
            alpha = np.arange(0, 1, step=0.05)
            traj = (start[:, None]*(1-alpha) + goal[:, None]*alpha).T
            path_est, path_target = execute_traj_ignoreCol(traj)
            Y_temp = []
            for path_i in path_est:
                p.resetBasePositionAndOrientation(robot, np.r_[path_i, 0.1], robotOrientation)
                Y_temp.append(get_distance(obstacles, robot))
            Y = np.r_[Y, Y_temp]
            X = np.r_[X, np.array(path_target)]
            count+=len(path_target)
        np.save(osp.join(dataFolder, f"X_{seed}.npy"), X)
        np.save(osp.join(dataFolder, f"Y_{seed}.npy"), Y)

    print("Using model from state-estimated points")
    kernel = GPy.kern.RBF(input_dim=2, variance=1)
    # Ignore the first row, since it is just filler values.
    m = GPy.models.GPRegression(X[1:,:], Y[1:,None], kernel)
    try:
        model_param = np.load(osp.join(dataFolder, f'env_{seed}_param.npy'))
        m.update_model(False)
        m.initialize_parameter()
        m[:] = model_param
        print("Loading saved model")
        m.update_model(True)
    except FileNotFoundError:
        print("Could not find trained model")
        m.optimize()
        np.save(osp.join(dataFolder, f'env_{seed}_param.npy'), m.param_array)
    return m


def get_model_KF_sparse(robot, obstacles, samples, dataFolder):
    '''
    Getting the sparse-GP model using the KF.
    :param robot: pybullet.MultiBody object representing the robot
    :param obstacles: pybullet.MultiBody object representing the obstacles in the environment
    :param model: custom function found in folder "models" of the dynamic system
    :param samples: The number of samples to use for the model
    :returns GPy.models.SparseGPRegression model representing the obstacle space
    '''
    try:
        X = np.load(osp.join(dataFolder, 'X.npy'))
        Y = np.load(osp.join(dataFolder, 'Y.npy'))
        print("Loading data for map")
    except FileNotFoundError:
        print("Could not find data, gathering data")
        X = np.array([[0.0, 0.0]])
        Y = np.array([0.0])
        robotOrientation = (0.0, 0.0, 0.0, 1.0)
        count = 0
        while count<samples:
            start = np.random.rand(2)*10
            goal = np.random.rand(2)*10
            alpha = np.arange(0, 1, step=0.05)
            traj = (start[:, None]*(1-alpha) + goal[:, None]*alpha).T
            path_est, path_target = execute_traj_ignoreCol(traj)
            Y_temp = []
            for path_i in path_est:
                p.resetBasePositionAndOrientation(robot, np.r_[path_i, 0.1], robotOrientation)
                Y_temp.append(get_distance(obstacles, robot))
            Y = np.r_[Y, Y_temp]
            X = np.r_[X, np.array(path_target)]
            count+=len(path_target)
        np.save(osp.join(dataFolder, "X.npy"), X)
        np.save(osp.join(dataFolder, "Y.npy"), Y)

    print("Using model from state-estimated points")
    kernel = GPy.kern.RBF(input_dim=2, variance=1)
    # Ignore the first row, since it is just filler values.
    rand_index = np.random.permutation(np.arange(1, X.shape[0]))
    Z = np.random.rand(100, 2)*10
    m = GPy.models.SparseGPRegression(
        X=X[rand_index[:samples],:], 
        Y=Y[rand_index[:samples],None], 
        Z=Z, 
        kernel=kernel
    )
    try:
        model_param = np.load(osp.join(dataFolder, 'env_3_param_sparse.npy'))
        m.update_model(False)
        m.initialize_parameter()
        m[:] = model_param
        print("Loading saved model")
        m.update_model(True)
    except FileNotFoundError:
        print("Could not find trained model")
        m.optimize()
        np.save(osp.join(dataFolder, 'env_3_param_sparse.npy'), m.param_array)
    return m