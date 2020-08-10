import pybullet as p
import time
import pybullet_data

import numpy as np


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")

p.resetDebugVisualizerCamera(
    cameraDistance=10, 
    cameraYaw=0, 
    cameraPitch=-89.9, 
    cameraTargetPosition=[5,5,0])

geomBox = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2,0.2, 0.2])
geomRobot = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.2)
# Initialize the position of obstacles
xy = [np.r_[2,2], np.r_[2, 8], np.r_[5,5], np.r_[8, 2], np.r_[8, 8]]

x = np.r_[0.0, 0.0, 0.1]
obstacles = [
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=geomBox,
        basePosition=np.r_[xy_i, 0.2]
    ) 
    for xy_i in xy
    ]

robot = p.createMultiBody(
    baseMass=0, 
    baseCollisionShapeIndex=geomRobot, 
    basePosition=[0.0, 0.0, 0.1])

pts_distance = min((p.getClosestPoints(bodyA=obs, bodyB=robot, distance=100)[0][8] for obs in obstacles))
print("Distance :{}".format(pts_distance))
A = np.eye(3)
B = np.zeros((3,3))
B[0, 0] = 0.1
B[1, 1] = 0.1
robotOrientation = p.getQuaternionFromEuler([0., 0., 0.])

for _ in range(50):
    x = A@x + B@np.random.rand(3)
    p.resetBasePositionAndOrientation(robot,x, robotOrientation)
    pts_distance = min((p.getClosestPoints(bodyA=obs, bodyB=robot, distance=100)[0][8] for obs in obstacles))
    print("Distance :{}".format(pts_distance))
    time.sleep(0.25)


