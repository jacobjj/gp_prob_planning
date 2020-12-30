'''Define a racecar based on pybullet example code.
'''
import pybullet_utils.bullet_client as bc
import pybullet as pyb
import pybullet_data
import numpy as np

import os
import time

from scipy import stats

import GPy

# Define the environment
from config import box_length, box_width, cir_radius

if False:
    # physicsClient =  p.connect (p.GUI)
    p = bc.BulletClient(connection_mode=pyb.GUI)
    p.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 0)
else:
    # physicsClient = p.connect(p.DIRECT)
    p = bc.BulletClient(connection_mode=pyb.DIRECT)

# For checking collision checking
p2 = bc.BulletClient(connection_mode=pyb.DIRECT)


# Default dt for pybullet is 1/240 or 240Hz
# We could change this by setting the parameter p.setTimeStep,
# but according to their documentation not ideal this parameter.
dt = 1/240

# Assuming the wheelbase of the car is the same used in MIT racecar
wheelbase = 0.325
# Initialize few model parameters
steering = [0, 2]  
wheels = [8, 15]


useRealTimeSim = 0

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

np.random.seed(10)
# Randomly generate boxes
num_boxes = 6
xy = np.random.rand(num_boxes, 2)*9 + 0.5

# Randomly generate circles
num_circles = 3
xy_circle = np.random.rand(num_circles, 2)*9 + 0.5

def set_obstacles(client_obj):
    '''
    A function to set obstacles in the environment
    :param client_obj: A pybullet_utils.BulletClient object
    '''
    geomBox = client_obj.createCollisionShape(pyb.GEOM_BOX, halfExtents=[box_length/2, box_width/2, 0.2])
    geomCircle = client_obj.createCollisionShape(pyb.GEOM_CYLINDER, radius=cir_radius, height = 0.4)
    geomRobot = client_obj.createCollisionShape(pyb.GEOM_CYLINDER, radius=0.1, height=0.2)

    obstacles_box = [
        client_obj.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=geomBox,
            basePosition=np.r_[xy_i, 0.2]
        ) 
        for xy_i in xy
        ]

    obstacles_circle = [
        client_obj.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=geomCircle,
            basePosition=np.r_[xy_i, 0.2]
        ) 
        for xy_i in xy_circle
    ]
    obstacles = obstacles_box + obstacles_circle
    return obstacles

def set_env():
    '''
    Set the environment up with obstacles and a racecar.
    :return tuple: The pybullet ID of obstacles and robot
    '''
    print('------------------------------------')
    set_simulation_env(p)
    # Load car on simulator
    car_1 = p.loadURDF("racecar/racecar_differential.urdf", [0.0, 0.0, 0.05074242991633105])  #, [0,0,2],useFixedBase=True)  #, [0,0,2],useFixedBase=True)
    initialize_constraints(client_obj=p, car=car_1)
    obstacles_1 = set_obstacles(p)        
    return obstacles_1, car_1

set_simulation_env(p2)
car_2 = p2.loadURDF("racecar/racecar_differential.urdf", [0.0, 0.0, 0.05074242991633105])  #, [0,0,2],useFixedBase=True)
initialize_constraints(client_obj=p2, car=car_2)
obstacles_2 = set_obstacles(p2)

targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -50, 50, 0)
maxForceSlider = p.addUserDebugParameter("maxForce", 0, 50, 20)
steeringSlider = p.addUserDebugParameter("steering", -1, 1, 0)

def reset(robot, x=0, y=0, theta=0):
    '''
    Reset the robot positon.
    :param robot: The robot ID of pybullet
    :param x: The x co-ordinate of the robot
    :param y: The y co-ordinate of the robot
    :param theta: The orientation of the robot
    '''
    # Reset the base velocity
    p.resetBaseVelocity(robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    quat = p.getQuaternionFromEuler((0, 0, theta))
    # Reset base position and orientation
    p.resetBasePositionAndOrientation(robot, np.r_[x, y, 0.05074242991633105], quat)
    # Reset the steering wheel position
    for steer in steering:
        p.resetJointState(robot, steer, targetValue=0.0)

# def step(targetVelocity=, steeringAngle, maxForce):
def step(robot, targetVelocity, steeringAngle, sliders = False):
    '''
    Do a single step of simulation for the robot. 
    NOTE: The actual velocity of the robot is scaled down 20 times because 
    of constraints.
    :param robot: The car that has to take a step
    :param targetVelocity: The target velocity to the wheels.
    :param steeringAngle: The angle of the steering angle, between -1&1.
    :param sliders: Enable GUI controllers to control the car.
    '''
    maxForce = 200
    if sliders:
        maxForce = p.readUserDebugParameter(maxForceSlider)
        targetVelocity = p.readUserDebugParameter(targetVelocitySlider)
        steeringAngle = p.readUserDebugParameter(steeringSlider)
    
    for wheel in wheels:
        p.setJointMotorControl2(robot,
                                wheel,
                                p.VELOCITY_CONTROL,
                                targetVelocity=targetVelocity,
                                force=maxForce)

    for steer in steering:
        p.setJointMotorControl2(robot, steer, p.POSITION_CONTROL, targetPosition=-steeringAngle)
    
    if (useRealTimeSim == 0):
        p.stepSimulation()
    # time.sleep(0.01)

M = np.eye(5)*5e-2 # Motion Noise
M[2, 2] = np.deg2rad(5) # Motion Noise for orientation
M[4, 4] = np.deg2rad(0)
N = np.eye(5)*1e-2 # Observation Noise
N[2,2] = np.deg2rad(2) # Motion Noise for orientation

def get_state(robot):
    '''
    Return the current state:(x, y, theta, steering angle, speed)
    :param robot: The pybullet robot ID
    :return tuple: (x, y, theta, steering angle, speed)
    '''
    pos, quat = p.getBasePositionAndOrientation(robot)
    vel, _ = p.getBaseVelocity(robot)
    speed = np.linalg.norm(vel[:2])
    theta = p.getEulerFromQuaternion(quat)[2]
    phi = -p.getJointState(robot, steering[0])[0]
    return (pos[0], pos[1], theta, phi, speed)

def get_noisy_state(robot):
    '''
    Return the noisy state and observation of the robot
    :param robot: The pybullet robot ID
    :return tuple: ((x,y,theta)+Motion noise, (x,y,theta)+Observation Noise
    '''
    x, y, theta, phi, speed  = get_state(robot)
    state = np.r_[x, y, theta, speed, phi]
    return state+stats.multivariate_normal.rvs(cov=M), state+stats.multivariate_normal.rvs(cov=N)

def get_dyn(robot):
    '''
    Return the parameters of the LTI system reprsented by :
    x_t+1 = A@x_t + B@u_t + m_t, m_t ~ N(0, M)
    z_t+1 = x_t + n_t, n_t ~ N(0,N)
    :returns (A,B,M,N): where the parameters correspond to the above equation.
    '''
    # Get the model kinematics
    _, _, theta, phi, speed = get_state(robot)

    # Define Robot model
    A = np.eye(3)
    B = np.zeros((3,1))

    # Set the parameters
    A[0, 2] = -dt*speed*np.sin(theta)
    A[1, 2] = dt*speed*np.cos(theta)
    B[2, 0] = dt*speed/(wheelbase*np.cos(phi)**2)

    return A, B, M, N

def get_jacobian(x_est, control):
    '''
    Return the Jacobian of the state
    '''
    theta = x_est[2]
    speed = x_est[3]
    # NOTE: Column 5 was just added with filler values, since observation
    # model just captues (x, y) and all other variables are 0.
    J = np.array([
        [1, 0, -dt*speed*np.sin(theta), dt*np.cos(theta), 0],
        [0, 1, dt*speed*np.cos(theta), dt*np.sin(theta), 0],
        [0, 0, 1, dt*np.tan(control[0])/wheelbase, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ])
    return J

def get_forward(x_est, control):
    '''
    Get the forward motion of the robot
    :param x_est: The previous estimate of the robot
    :param control: The steering angle of the robot
    :return np.array: The next state of the robot
    '''
    # max_a = 20
    speed = x_est[3]
    delta_phi = 0.1*(control-x_est[4])
    x = x_est + np.r_[
        speed*dt*np.cos(x_est[2]),
        speed*dt*np.sin(x_est[2]),
        speed*dt*np.tan(delta_phi+x_est[4])/wheelbase,
        (20-speed),
        delta_phi
        ]
    # x[2] = pi_2_pi(x[2])
    return x

H = np.eye(3, 5)

def x_hat(robot, x_est, control, obs, P):
    '''
    :param robot: The pybullet id for the car
    :param x_est: The previous state estimate
    :param control: The current control, (steering angle)
    :param obs: The current observatoin
    :returns tuple: A tuple of the current state estimate and Covariance matrix
    '''
    A, B, M, N = get_dyn(robot)
    # Predict step
    x_bar = get_forward(x_est, control)
    J = get_jacobian(x_est, control)
    P_bar = J@P@J.T + M
    # Update
    S = H@P_bar@H.T + N[:3,:3]

    eig,_= np.linalg.eig(S)
    K = P_bar @ H.T @ np.linalg.inv(S)
    y = H @ (obs-x_bar)
    x_est = x_bar + K@y
    P = (np.eye(5)-K@H)@P_bar
    return x_est, P

def get_lqr(traj, C, D):
    '''
    Returns the LQR gains for a finite horizon system.
    :param traj: The trajectory to be followed
    :param C: The cost for state
    :param D: The cost for control
    :return list: A list of LQR gains
    '''
    # Get LQR gains
    SN = C
    L_list = []
    # Evaluate L from the end
    for _ in traj[:-1]:
        # TODO: Calculate the LQR gains
        pass

    return L_list

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


# GP collision model
def get_model(robot, obstacles, model):
    '''
    Samples closest distance from obstacles from different points.
    :param robot: pybullet.MultiBody object representing the robot
    :param obstacles: pybullet.MultiBody object representing the obstacles in the environment
    :param model: custom function found in folder "models" of the dynamic system
    :returns GPy.models.GPRegression model representing the obstacle space.
    '''
    print("Using model from sampled distance points")
    samples = 1000
    X = np.random.rand(samples, 2)*10
    theta = (np.random.rand(samples)*2-1)*np.pi

    Y = np.zeros((samples, 1))
    for i in range(samples):
        robotOrient = p.getQuaternionFromEuler((0.0, 0.0, theta[i]))
        p.resetBasePositionAndOrientation(robot, np.r_[X[i,:], 0.05074242991633105], robotOrient)
        # Add random noise to the model
        Y[i] = model.get_distance(obstacles, robot) #+ stats.norm.rvs(scale=1)

    kernel = GPy.kern.RBF(input_dim=4, variance=1)
    X_orientation = np.c_[np.cos(theta), np.sin(theta)]
    # X = np.c_[X, theta]
    # m = GPy.models.GPRegression(X, Y, kernel)
    X_SE2 = np.c_[X, X_orientation]
    m = GPy.models.GPRegression(X_SE2, Y ,kernel)
    m.optimize()
    return m

def get_path(robot, obstacles, model):
    '''
    Generate a random path, until collision or, until the robot leaves the
    workspace and calculate the collision distance.
    :param robot: The pybullet robot ID
    :param obstacles: A list of pybullet obstacle ID
    :param model: the robot python object
    :return (est_state, d): A tuple of estimated state and true distance function.
    '''
    est_state = []
    d = []
    P = np.eye(5)*0

    x, y, theta, _, _ = model.get_state(robot)
    x_est = np.r_[x, y, theta, 0.0, 0.0]
    est_state.append(x_est)
    collision = False
    for _ in range(1000):
        # Sample a random steer value
        u = (np.random.rand(1)*2-1)*np.pi/2
        model.step(robot, 20, u)
        # Get state
        x, y, theta, _, _ = model.get_state(robot)
        # Check for collision 
        if any((p.getContactPoints(robot, obs) for obs in obstacles)):
            collision = True
            break 
        # Check for state is out of frame
        if x<-0.1 or x > 10.1 or y<-0.1 or y>10.1:
            break

        # Get estimated state
        state, obs = model.get_noisy_state(robot)
        x_est, P = model.x_hat(robot, x_est, u, obs, P)
        est_state.append(x_est)
    
    # Sub-samples position.
    est_state = est_state[::100]
    for state_i in est_state:
        model.reset(robot, state_i[0], state_i[1], state_i[2])
        d.append(model.get_distance(obstacles, robot))
    if collision:
        # Sample random points
        for _ in range(5):
            x_temp = x + np.random.rand()
            y_temp = y + np.random.rand()
            theta = (np.random.rand()*2-1)*np.pi/2
            model.reset(robot, x, y, theta)
            est_state.append(np.r_[x_temp, y_temp, theta, 0., 0.0])
            d.append(model.get_distance(obstacles, robot))
    return est_state, d


# GP collision model from KF state_estimation
def get_model_KF(robot, obstacles, model):
    '''
    Navigate the model around the environment, and collect
    distance to collision. Using the data, construct a GP model to
    estimate the distance to collision.
    :param robot: pybullet.MultiBody object representing the robot
    :param obstacles: pybullet.MultiBody object representing the obstacles in the environment
    :param model: custom function found in folder "models" of the dynamic system
    :returns GPy.models.GPRegression model representing the obstacle space
    '''
    print("Using model from calculated distance points")
    try:
        X = np.load('X_dubins.npy')
        Y = np.load('Y_dubins.npy')
        print("Loading data for map")
    except FileNotFoundError:
        X = np.array([[0.0, 0.0, 0.0, 0.0]])
        Y = np.array([0.0])
        samples = 2000
        num = 0
        while num<samples:
            state = np.random.rand(2)*10
            theta = (np.random.rand()*2-1)*np.pi
            model.reset(robot, state[0], state[1], theta)
            while model.get_distance(obstacles, robot)<0.0:
                state = np.random.rand(2)*10
                theta = (np.random.rand()*2-1)*np.pi
                model.reset(robot, state[0], state[1], theta)

            est_state, d = get_path(robot, obstacles, model)

            X_temp = np.array(est_state)
            X_orientation = np.c_[np.cos(X_temp[:, 2]), np.sin(X_temp[:, 2])]
            X_SE2 = np.c_[X_temp[:, :2], X_orientation]

            X = np.r_[X, X_SE2]
            Y = np.r_[Y, d]
            num += X_temp.shape[0]
        np.save('X_dubins.npy', X)
        np.save('Y_dubins.npy', Y)
    
    # Construct model
    kernel = GPy.kern.RBF(input_dim=4)
    m = GPy.models.GPRegression(X[1:,:], Y[1:,None], kernel)
    try:
        data = np.load('env_10_param.npy')
        # Ignore the first row, since it is just filler values.
        m.update_model(False)
        m.initialize_parameter()
        m[:] = data
        print("Loading saved model")
        m.update_model(True)
    except FileNotFoundError:
        m.optimize()
        np.save('env_10_param.npy', m.param_array)
    return m 


def pi_2_pi(angle):
    '''
    Convert angle to -pi/pi.
    :param angle:The angle to convert
    :return float: The angle between -pi/pi.
    '''
    return (angle + np.pi) % (2 * np.pi) - np.pi

def execute_path(robot, traj, obstacles):
    '''
    Execute the path for evaluating the quality of the path, using KF to estimate the state and 
    PD control for trajectory following.
    :param robot: The pybullet id of the robot
    :param traj: The trajectory to follow.
    :param obstacles: The list of obstacles
    :return Bool: True if path was sucessfully completed
    '''
    # Control parameters
    v = 20
    K = 2.

    reset(robot, traj[0, 0], traj[0, 1], traj[0, 2])
    x, y, theta, _, _ = get_state(robot)
    x_est = np.r_[x, y, theta, 0.0, 0.0]
    look_ahead = 0.05
    
    # KF Parameters
    P = np.eye(5)*1e-1
    goal_reached = False
    num = 0
    goal = traj[-1, :2]
    while not goal_reached and num<1e4:
        index_bound, = np.where(np.linalg.norm(x_est[:2]-traj[:,:2], axis=1)<look_ahead)
        if traj.shape[0]>1 and len(index_bound)>0:
            traj = traj[int(index_bound[-1]):]
            delay = 0
        elif traj.shape[0]>1:
            delay+=1
        if delay>400 and traj.shape[0]>1:
            traj = traj[1:]
        if np.linalg.norm(x_est[:2] - goal)<0.25:
            goal_reached = True
            break

        state_hat = traj[0,:]

        # Define the control
        delta_angle = pi_2_pi(theta - state_hat[2])
        delta_d = np.linalg.norm(state_hat[:2]-x_est[:2])
        steering = np.arctan2(2*wheelbase*np.sin(delta_angle/2), v) + K*delta_angle
        steering = np.clip(steering, -np.pi*0.35, np.pi*0.35)

        step(robot, v, steering)
        if any((p.getContactPoints(robot, obs) for obs in obstacles)):
            # print("Collision")
            break  
            
        # Estimate the state using KF
        state, obs = get_noisy_state(robot)
        x_est, P = x_hat(robot, x_est, np.r_[steering], obs, P)
        theta = x_est[2]
        num+=1
    return goal_reached


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


def calc_nearest_index(state, traj):
    '''
    Calcunp.linalgte the closest index of trajectory path.
    '''
    d = np.linalg.norm(state[:2] - traj[:, :2], axis=1)
    ind = np.argmin(d)

    mind = np.sqrt(d[ind])

    dxl = traj[ind, 0] - state[0]
    dyl = traj[ind, 1] - state[1]

    angle = pi_2_pi(traj[ind, 2] - np.arctan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def est_check_collision(x, y, theta):
    '''
    Check if an estimated state is in collision
    :param x: The x position of robot
    :param y: The y position of robot
    :param theta: The theta of robot
    '''
    quat = p.getQuaternionFromEuler((0, 0, theta))
    # Reset base position and orientation
    p2.resetBasePositionAndOrientation(car_2, np.r_[x, y, 0.05074242991633105], quat)
    distance = min(
        (
            min(p2.getClosestPoints(bodyA=obs, bodyB=car_2, distance=100), key=lambda x:x[8])[8]
            for obs in obstacles_2
        )
    )
    return distance<0


def execute_path_LQR(robot, traj, obstacles):
    '''
    Returns true if a trajectory executed using LQR controller
    without collision.
    :param robot: The pybullet id of the robot
    :param traj: The numpy array of trajectory
    :param obstacles: A list of pybullet id of obstacles
    :return bool: True if the goal is reached.
    '''
    done, _, _ = execute_path_LQR_data(robot, traj, obstacles)
    return done


def execute_path_LQR_data(robot, traj, obstacles, get_log=False):
    '''
    Executes a trajectory using LQR controller and returns
    path log if required.
    :param robot: The pybullet id of the robot
    :param traj: The numpy array of trajectory
    :param obstacles: A list of pybullet id of obstacles
    :param get_log: A bool, which records the estimated and true path of the robot
    :return (bool, np.array, np.array): A tuple of done flag, true trajectory,
    and estimated trajectory.
    '''
    reset(robot, traj[0,0], traj[0,1], traj[0,2])
    x, y, theta, _, _ =  get_state(robot)
    x_est = np.r_[x, y, theta, 1e-3, 0.0]

    goal_reached = False
    Q = np.eye(4)*5
    R = np.eye(1)
    P = np.eye(5)*0
    goal= traj[-1,:]
    num = 0
    true_traj = []
    est_traj = []
    while not goal_reached and num<1e4:
        ind, e = calc_nearest_index(x_est, traj)
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

        K, _, eigvals = dlqr(A, B, Q, R)

        x = np.zeros((4, 1))
        th_e = pi_2_pi(x_est[2] - traj[ind, 2])
        x[0, 0] = x_est[0]-traj[ind, 0]
        x[1, 0] = x_est[1]-traj[ind, 1]
        x[2, 0] = th_e
        x[3, 0] = 0#20-v
        
        fb = pi_2_pi((-K @ x)[0, 0])
        delta = np.clip(fb, -np.pi*0.35, np.pi*0.35)
        
        # TODO : Explain why the control has to be negative??
        step(robot, 20, -delta)
        state, obs =  get_noisy_state(robot)
        x_est, P =  x_hat(robot, x_est, np.r_[-delta], obs, P)
        x, y, theta, phi, speed = get_state(robot)
        if get_log:
            true_traj.append(x_est[:3])
            est_traj.append(np.r_[x, y, theta])
    
        # if any((p.getContactPoints(robot, obs) for obs in obstacles)):
        #     # print("Collision")
        #     break
        
        if est_check_collision(x_est[0], x_est[1], x_est[2]):
            # print("Collision")
            break

        if np.linalg.norm(x_est[:2]-goal[:2])<0.25:
            goal_reached = True
        num+=1
    return goal_reached, true_traj, est_traj


def del_all():
    '''
    Delete all the pybullet connections
    '''
    p.__del__()
    p2.__del__()