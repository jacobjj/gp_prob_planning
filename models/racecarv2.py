''' A car model from Probabilistic Robotics.
'''
import pybullet_utils.bullet_client as bc
import pybullet as pyb
import pybullet_data

import numpy as np
from scipy import stats

import GPy

# Define the environment
from config import box_length, box_width, cir_radius

# Assuming the wheelbase of the car is the same used in MIT racecar
wheelbase = 0.325
# Initialize few model parameters
steering = [0, 2]  
wheels = [8, 15]

useRealTimeSim = 0
# p =  bc.BulletClient(connection_mode=pyb.GUI)
p = bc.BulletClient(connection_mode=pyb.DIRECT)

# Model parameters
delta_t = 1/20
class NoiseParams:
    # Motion Noise
    M = None
    M_t = None
    gamma_t = None
    # Observation Noise
    N = None
    N_t = None
    modelName= None

def set_simulation_env(client_obj):
    '''
    Set environment for the given client_object.
    :param client_obj: A pybullet_utils.BulletClient object
    '''

    client_obj.setAdditionalSearchPath(pybullet_data.getDataPath())
    client_obj.resetSimulation()
    client_obj.setGravity(0, 0, -9.8)

    #for video recording (works best on Mac and Linux, not well on Windows)
    #client_obj.startStateLogging(client_obj.STATE_LOGGING_VIDEO_MP4, "racecar.mp4")
    client_obj.setRealTimeSimulation(useRealTimeSim)  # either this
    client_obj.loadURDF("plane.urdf")

    client_obj.resetDebugVisualizerCamera(
        cameraDistance=10, 
        cameraYaw=0, 
        cameraPitch=-89.9, 
        cameraTargetPosition=[5,5,0])

def initialize_constraints(client_obj, car):
    '''
    Initialize constraints for the car model.
    :param car: The pybullet id of the car model.
    '''
    for wheel in range(client_obj.getNumJoints(car)):
        client_obj.setJointMotorControl2(car, wheel, pyb.VELOCITY_CONTROL, targetVelocity=0, force=0)
        client_obj.getJointInfo(car, wheel)

    print("----------------")

    #client_obj.setJointMotorControl2(car,10,client_obj.VELOCITY_CONTROL,targetVelocity=1,force=10)
    c = client_obj.createConstraint(car,
                        9,
                        car,
                        11,
                        jointType=pyb.JOINT_GEAR,
                        jointAxis=[0, 1, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
    client_obj.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = client_obj.createConstraint(car,
                        10,
                        car,
                        13,
                        jointType=pyb.JOINT_GEAR,
                        jointAxis=[0, 1, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
    client_obj.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = client_obj.createConstraint(car,
                        9,
                        car,
                        13,
                        jointType=pyb.JOINT_GEAR,
                        jointAxis=[0, 1, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
    client_obj.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = client_obj.createConstraint(car,
                        16,
                        car,
                        18,
                        jointType=pyb.JOINT_GEAR,
                        jointAxis=[0, 1, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
    client_obj.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = client_obj.createConstraint(car,
                        16,
                        car,
                        19,
                        jointType=pyb.JOINT_GEAR,
                        jointAxis=[0, 1, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
    client_obj.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = client_obj.createConstraint(car,
                        17,
                        car,
                        19,
                        jointType=pyb.JOINT_GEAR,
                        jointAxis=[0, 1, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
    client_obj.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = client_obj.createConstraint(car,
                        1,
                        car,
                        18,
                        jointType=pyb.JOINT_GEAR,
                        jointAxis=[0, 1, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
    client_obj.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
    c = client_obj.createConstraint(car,
                        3,
                        car,
                        19,
                        jointType=pyb.JOINT_GEAR,
                        jointAxis=[0, 1, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
    client_obj.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)



def set_env(name='Random', **kwargs):
    '''
    Set the environment as either Random or Gibson
    :param name: Either has to be 'Random' or 'Gibon'
    :param kwargs: Other environment parameters
    :return tuple: The pybullet ID of obstacles and robot
    '''
    assert name in ['Random', 'Gibson'], 'Only two environments possible Gibson/Random'
    print('------------------------------------')
    set_simulation_env(p)
    # Load car on simulator
    car = p.loadURDF("racecar/racecar_differential.urdf", [0.0, 0.0, 0.05074242991633105])  #, [0,0,2],useFixedBase=True)  #, [0,0,2],useFixedBase=True)
    initialize_constraints(client_obj=p, car=car)
    if name=='Random':
        from models.randomWorld import set_obstacles
        # Motion Noise
        NoiseParams.M = np.eye(3)*1e-1
        NoiseParams.M[2, 2] = np.deg2rad(10)**2
        NoiseParams.M_t = stats.multivariate_normal(cov=NoiseParams.M[:2, :2])
        NoiseParams.gamma_t = stats.norm(scale=np.sqrt(NoiseParams.M[2, 2]))
        # Observation Noise
        NoiseParams.N = np.eye(3)*1e-1
        NoiseParams.N[2, 2] = np.deg2rad(5)**2
        NoiseParams.N_t = stats.multivariate_normal(cov=NoiseParams.N)
        NoiseParams.modelName = f'random_{seed}'
    else:
        from models.gibsonWorld import set_obstacles
        # Motion Noise
        NoiseParams.M = np.eye(3)*5e-2
        NoiseParams.M[2, 2] = np.deg2rad(10)**2
        NoiseParams.M_t = stats.multivariate_normal(cov=NoiseParams.M[:2, :2])
        NoiseParams.gamma_t = stats.norm(scale=np.sqrt(NoiseParams.M[2, 2]))
        # Observation Noise
        NoiseParams.N = np.eye(3)*5e-2
        NoiseParams.N[2, 2] = np.deg2rad(5)**2
        NoiseParams.N_t = stats.multivariate_normal(cov=NoiseParams.N)
        NoiseParams.modelName = f'gibson'

    obstacles = set_obstacles(p, **kwargs)
    return obstacles, car

def check_collision(x, obstacles, robot):
    '''
    Check if the robot is in collision at x with obstacles.
    :param x: The state of the robot.
    :param obstacles: A list of all obstacles ID's
    :param robot: The robot ID
    :return bool: True if the x is in collision.
    '''
    robotOrientation = p.getQuaternionFromEuler((0, 0, x[2]))
    p.resetBasePositionAndOrientation(robot, np.r_[x[:2], 0.05074242991633105], robotOrientation)
    return get_distance(obstacles, robot)<0


def get_distance(obstacles ,robot):
    '''
    Get the shortest distance between the obstacles and robot, for the given setup.
    :param obstacles: A list of all obstacle ID's
    :param robot: The robot ID
    :return float : The penetration distance between the robot and obstacles, If the distance is
    negative, object is in collision.
    '''
    assert isinstance(obstacles, list), "Obstacles has to be a list"
    try:
        distance = min(
            (
                min(p.getClosestPoints(bodyA=obs, bodyB=robot, distance=100), key=lambda x:x[8])[8]
                for obs in obstacles
            )
        )
    except ValueError:
        import pdb;pdb.set_trace()
    return distance

def pi_2_pi(angle):
    '''
    Convert angle to -pi/pi.
    :param angle:The angle to convert
    :return float: The angle between -pi/pi.
    '''
    return (angle + np.pi) % (2 * np.pi) - np.pi


def step(x, u):
    '''
    Setting up the velocity model as proposed in Probablistic Planning textbook.
    :param u : The input control
    :param x : The current state of the robot.
    :return np.array: The next state of the robot.
    '''
    assert u.shape[0] == 2
    assert x.shape[0] == 3
    
    u = u + NoiseParams.M_t.rvs()
    theta = x[2] + NoiseParams.gamma_t.rvs()*delta_t
    x = get_forward(x, u)
    return x


def get_forward(x, u):
    '''
    Get the forward motion of the racecar without noise.
    :param x: The current state of the robot
    :param u: The input control
    :return np.array: The next state of the robot
    '''
    
    if np.isclose(u[1], 0):
        delta_x = np.r_[
            u[0]*delta_t*np.cos(x[2]),
            u[0]*delta_t*np.sin(x[2]),
            0.0
        ]
    else:
        K = u[0]/u[1]
        delta_x = np.r_[
            K*(-np.sin(x[2]) + np.sin(x[2]+u[1]*delta_t)),
            K*(np.cos(x[2]) - np.cos(x[2]+u[1]*delta_t)),
            u[1]*delta_t
        ]
    x = x + delta_x
    # x[2] = pi_2_pi(x[2])
    return x


def get_state_obs(x, u):
    '''
    Return the observation and next state of the robot.
    :param x: The current state of the robot
    :param u: The control input of the robot.
    :return (np.array, np.array): The next state and current observation 
    of the robot
    '''
    x = step(x, u)
    z = x + NoiseParams.N_t.rvs()
    return x, z


def get_dyn(x, u):
    '''
    Getting the next state wsing the EKF model.
    :param x: The cwrrent state.
    :param w: The cwrrent control inpwt.
    '''
    assert u.shape[0] == 2
    assert x.shape[0] == 3
    w = u[1]
    if np.isclose(w, 0):
        w = 1e-6
        
    V = u[0]/w
    A = np.array([
        [1, 0, V*(-np.cos(x[2]) + np.cos(x[2]+w*delta_t))],
        [0, 1, V*(-np.sin(x[2]) + np.sin(x[2]+w*delta_t))],
        [0, 0, 1]
    ])
    B = np.array([
        [(V/w)*(np.sin(x[2]) - np.sin(x[2]+w*delta_t)) + V*np.cos(x[2]+w*delta_t)*delta_t],
        [(V/w)*(-np.cos(x[2]) + np.cos(x[2]+w*delta_t)) + V*np.sin(x[2]+w*delta_t)*delta_t],
        [delta_t]
    ])

    V = np.array([
        [(1/w)*(-np.sin(x[2]) + np.sin(x[2]+w*delta_t)), (V/w)*(np.sin(x[2]) - np.sin(x[2]+delta_t*w)) + V*np.cos(x[2]+w*delta_t)*delta_t, 0], 
        [(1/w)*(np.cos(x[2]) - np.cos(x[2]+w*delta_t)), (V/w)*(-np.cos(x[2]) + np.cos(x[2]+delta_t*w)) + V*np.sin(x[2]+w*delta_t)*delta_t, 0],
        [0, delta_t, delta_t]
    ])
    return A, B, V


def ekf(x, Sigma, u, z):
    '''
    Estimate the target location and distribution.
    :param x: The current state of the robot
    :param Sigma : The variance of current state.
    :param u: The input control
    :param z: The current observation.
    :return (np.array, np.array): The mean and variance of the next state.
    '''

    # Get model parameters
    A, B, V = get_dyn(x, u)
    
    x_bar = get_forward(x, u)
    Sigma_bar = A@Sigma@A.T + V@NoiseParams.M@V.T
    
    K = Sigma_bar@np.linalg.inv(Sigma_bar+NoiseParams.N)
    
    x = x_bar + K@(z-x_bar)
    Sigma = (np.eye(3)-K)@Sigma_bar
    
    return x, Sigma

def get_starting_position(robot, obstacles):
    '''
    Get the random position of the robot.
    :param robot:
    :param obstacles: 
    :returns np.array: A viable starting position.
    '''
    valid_start = False
    while not valid_start:
        start = np.random.rand(2)*10
        theta = np.random.rand()*np.pi*2
        x_state = np.r_[start, theta]
        robotOrientation = p.getQuaternionFromEuler((0, 0, theta))
        p.resetBasePositionAndOrientation(robot, np.r_[start, 0.05074242991633105], robotOrientation)
        valid_start = get_distance(obstacles, robot)>0
    return x_state


def get_path(samples,robot, obstacles):
    '''
    Generate a random trajectory
    :param samples: The number of the samples to generate 
    :param robot:
    :param obstacles:
    :returns np.array: A trajectory and distance measure for a 
    random sample.
    '''
    x = get_starting_position(robot, obstacles)
    x_est = x
    Sigma = np.eye(3)*0
    
    # Get a random path
    true_path = [x]
    est_path = [x]
    for _ in range(samples):
        u = np.r_[1, np.random.uniform(low=-1, high=1)]
        x, z = get_state_obs(x, u)
        x_est, Sigma = ekf(x_est, Sigma, u, z)
        if any(x[:2]>10) or any(x[:2]<0):
            break
        true_path.append(x)
        est_path.append(x_est)
    
    # Get the distance
    d = []
    for x_est in est_path:
        robotOrientation = p.getQuaternionFromEuler((0, 0, x_est[2]))
        p.resetBasePositionAndOrientation(robot, np.r_[x_est[:2], 0.05074242991633105], robotOrientation)
        d.append(get_distance(obstacles, robot))
    
    # Return the true path, random path, and distance.
    return np.array(true_path), np.array(est_path), np.array(d)


def get_model_KF(robot, obstacles):
    '''
    Navigate the model around the environment, and collect
    distance to collision. Using the data, construct a GP model to
    estimate the distance to collision.
    :param robot: pybullet.MultiBody object representing the robot
    :param obstacles: pybullet.MultiBody object representing the obstacles in the environment
    :returns GPy.models.GPRegression model representing the obstacle space
    '''
    try:
        print("Loading data for map")
        X = np.load(f'param/X_{NoiseParams.modelName}_dubins.npy')
        Y = np.load(f'param/Y_{NoiseParams.modelName}_dubins.npy')
    except FileNotFoundError:
        print("Did not find file, Generating data ....")
        X = np.array([[0.0, 0.0, 0.0, 0.0]])
        Y = np.array([0.0])
        count = 0
        samples = 2000
        skip = 5
        while count<samples:
            true_path, est_path, d = get_path(100, robot, obstacles)
            X_orientation = np.c_[np.cos(est_path[::skip, 2]), np.sin(est_path[::skip, 2])]
            
            X_SE2 = np.c_[est_path[::skip, :2], X_orientation]    
            Y = np.r_[Y, d[::skip]]
            X = np.r_[X, X_SE2]
            count+=len(d[::skip])
        np.save(f'param/X_{NoiseParams.modelName}dubins.npy', X)
        np.save(f'param/Y_{NoiseParams.modelName}dubins.npy', Y)

    # Construct model
    kernel = GPy.kern.RBF(input_dim=4)
    m = GPy.models.GPRegression(X[1:,:], Y[1:,None], kernel)
    try:
        print("Loading data from file...")
        data = np.load(f'param/env_{NoiseParams.modelName}_param_dubins.npy')
        # Ignore the first row, since it is just filler values.
        m.update_model(False)
        m.initialize_parameter()
        m[:] = data
        print("Loading saved model")
        m.update_model(True)
    except FileNotFoundError:
        print("Could not find parameter file, optimizing ...")
        m.optimize()
        np.save(f'param/env_{NoiseParams.modelName}_param_dubins.npy', m.param_array)
    return m 


# Code for LQR control
# NOTE: Take from https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/lqr_steer_control/lqr_steer_control.py
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


def calc_nearest_index(state, traj, cur_ind=None):
    '''
    Calcualate the closest index of trajectory path.
    :param state: The current state of the robot.
    :param traj: The complete trajectory.
    :param cur_ind: The current index of the robot.
    '''
    d = np.linalg.norm(state[:2] - traj[:, :2], axis=1)
    if cur_ind is not None:
        start_ind = max(0, cur_ind-50)
        stop_ind = min(traj.shape[0], cur_ind+50)
        ind = start_ind + np.argmin(d[start_ind:stop_ind])
    else:
        ind = np.argmin(d)

    mind = np.sqrt(d[ind])

    dxl = traj[ind, 0] - state[0]
    dyl = traj[ind, 1] - state[1]

    angle = pi_2_pi(traj[ind, 2] - np.arctan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


Q = np.eye(3)*10
# Q[2, 2] = 5
R = np.eye(1)*0.1

#TODO: Make it Gibson Compatible
def execute_path(robot, traj_orien, obstacles):
    '''
    Execute the path for evaluating the quality of the path, using KF to estimate the state and 
    PD control for trajectory following.
    :param robot: The pybullet id of the robot
    :param traj: The trajectory to follow.
    :param obstacles: The list of obstacles
    :return Bool: True if path was sucessfully completed
    '''
    x = traj_orien[0, :].T
    goal = traj_orien[-1, :2]
    x_est = x
    
    Sigma = np.eye(3)*0
    num = 0
    ind = 1
    u = np.r_[0.0 , 0.0]
    goal_reached = False
    path = x[None, :]
    while (not goal_reached) and num<8e3:
        temp_ind, e = calc_nearest_index(x_est, traj_orien, ind)
        if temp_ind>ind:
            ind = temp_ind
        A, B, V = get_dyn(x, u)
        K, _, eigvals = dlqr(A, B, Q, R)

        delta_x = np.zeros((3, 1))
        th_e = pi_2_pi(x_est[2] - traj_orien[ind, 2])
        delta_x[0, 0] = x_est[0]-traj_orien[ind, 0]
        delta_x[1, 0] = x_est[1]-traj_orien[ind, 1]
        delta_x[2, 0] = th_e
        fb = (K @ delta_x)[0, 0]
        delta = np.clip(fb, -1, 1)
        u = np.r_[0.5, -delta]
        x, z = get_state_obs(x, u)
        path = np.r_[path, x[None, :]]
        # Check for collisions
        if check_collision(x, obstacles, robot):
            break
        x_est, Sigma = ekf(x_est, Sigma, u, z)
        if np.linalg.norm(x[:2]-goal.T)<0.25:
            goal_reached = True
        num +=1

    return path, goal_reached
