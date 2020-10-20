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
def get_dyn():
    '''
    Return the dynamics of the LTI system
    '''
    A = np.eye(3)
    B = np.zeros((3,3))
    B[0, 0] = 2
    B[1, 1] = 2
    return A,B