'''Define a LTI point system
'''
import pybullet as p
import pybullet_data
import numpy as np

from config import box_length, box_width, cir_radius

np.random.seed(5)

# physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")

p.resetDebugVisualizerCamera(
    cameraDistance=10, 
    cameraYaw=0, 
    cameraPitch=-89.9, 
    cameraTargetPosition=[5,5,0])

geomBox = p.createCollisionShape(p.GEOM_BOX, halfExtents=[box_length/2, box_width/2, 0.2])
geomCircle = p.createCollisionShape(p.GEOM_CYLINDER, radius=cir_radius, height = 0.4)
geomRobot = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.2)
# Initialize the position of obstacles
# xy = [np.r_[2,2], np.r_[2, 8], np.r_[5,5], np.r_[8, 2], np.r_[8, 8]]

# Randomly generate boxes
num_boxes = 10
xy = np.random.rand(num_boxes, 2)*8.5 + 0.5

# Randomly generate circles
num_circles = 10
xy_circle = np.random.rand(num_circles, 2)*8.5 + 0.5

def set_env():
    '''
    Set the environment up with the obstacles and robot.
    :return tuple: The pybullet ID of obstacles and robot
    '''
    obstacles_box = [
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=geomBox,
            basePosition=np.r_[xy_i, 0.2]
        ) 
        for xy_i in xy
        ]

    obstacles_circle = [
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=geomCircle,
            basePosition=np.r_[xy_i, 0.2]
        ) 
        for xy_i in xy_circle
    ]
    obstacles = obstacles_box + obstacles_circle
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

M = np.eye(2)*1e-2 # Motion Noise
N = np.eye(2)*1e-3 # Observation Noise
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
    M = np.eye(2)*1e-2 # Motion Noise
    N = np.eye(2)*1e-3 # Observation Noise

    return A, B, M, N
