''' Create a random environment
'''

import pybullet_utils.bullet_client as bc
import pybullet as pyb
import pybullet_data

import numpy as np


from config import box_length, box_width, cir_radius

# ENVIRONMENT
np.random.seed(10)

# Randomly generate boxes
num_boxes = 8
xy = np.random.rand(num_boxes, 2)*9 + 0.5

# Randomly generate circles
num_circles = 5
xy_circle = np.random.rand(num_circles, 2)*9 + 0.5

def set_obstacles(client_obj):
    '''
    A function to set obstacles in the environment
    :param client_obj: A pybullet_utils.BulletClient object
    '''
    rgba = [0.125, 0.5, 0.5, 1]
    geomBox = client_obj.createCollisionShape(pyb.GEOM_BOX, halfExtents=[box_length/2, box_width/2, 0.2])
    visualBox = client_obj.createVisualShape(pyb.GEOM_BOX, halfExtents=[box_length/2, box_width/2, 0.2], rgbaColor=rgba)
    geomCircle = client_obj.createCollisionShape(pyb.GEOM_CYLINDER, radius=cir_radius, height = 0.4)
    visualCircle = client_obj.createVisualShape(pyb.GEOM_CYLINDER, radius=cir_radius, length = 0.4, rgbaColor=rgba)


    obstacles_box = [
        client_obj.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=geomBox,
            baseVisualShapeIndex=visualBox,
            basePosition=np.r_[xy_i, 0.2]
        ) 
        for xy_i in xy
        ]

    obstacles_circle = [
        client_obj.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=geomCircle,
            baseVisualShapeIndex=visualCircle,
            basePosition=np.r_[xy_i, 0.2]
        ) 
        for xy_i in xy_circle
    ]
    obstacles = obstacles_box + obstacles_circle
    return obstacles
