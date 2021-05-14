''' Create a random environment
'''

import pybullet_utils.bullet_client as bc
import pybullet as pyb
import pybullet_data

import numpy as np
import matplotlib.pyplot as plt
from config import box_length, box_width, cir_radius

class RandomWorld:
    seed = 10

    # Randomly generate boxes
    num_boxes = 8
    xy = []

    # Randomly generate circles
    num_circles = 5
    xy_circle = []

def set_obstacles(client_obj, **kwargs):
    '''
    A function to set obstacles in the environment
    :param client_obj: A pybullet_utils.BulletClient object
    :param kwargs: Additional environment parameters.
        seed: The random seed used to create environment
        num_boxes: Number of square obstacles
        num_circles: Number of circular obstacles
    '''
    rgba = [0.125, 0.5, 0.5, 1]
    geomBox = client_obj.createCollisionShape(pyb.GEOM_BOX, halfExtents=[box_length/2, box_width/2, 0.2])
    visualBox = client_obj.createVisualShape(pyb.GEOM_BOX, halfExtents=[box_length/2, box_width/2, 0.2], rgbaColor=rgba)
    geomCircle = client_obj.createCollisionShape(pyb.GEOM_CYLINDER, radius=cir_radius, height = 0.4)
    visualCircle = client_obj.createVisualShape(pyb.GEOM_CYLINDER, radius=cir_radius, length = 0.4, rgbaColor=rgba)

    if 'seed' in kwargs.keys():
        np.random.seed(kwargs['seed'])
        RandomWorld.seed = kwargs['seed']
    else:
        np.random.seed(RandomWorld.seed)

    if 'num_boxes' in kwargs.keys():
        RandomWorld.num_boxes = kwargs['num_boxes']

    if 'num_circles' in kwargs.keys():
        RandomWorld.num_circles = kwargs['num_circles']

    RandomWorld.xy = np.random.rand(RandomWorld.num_boxes, 2)*9 + 0.5
    RandomWorld.xy_circle = np.random.rand(RandomWorld.num_circles, 2)*9 + 0.5

    obstacles_box = [
        client_obj.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=geomBox,
            baseVisualShapeIndex=visualBox,
            basePosition=np.r_[xy_i, 0.2]
        ) 
        for xy_i in RandomWorld.xy
        ]

    obstacles_circle = [
        client_obj.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=geomCircle,
            baseVisualShapeIndex=visualCircle,
            basePosition=np.r_[xy_i, 0.2]
        ) 
        for xy_i in RandomWorld.xy_circle
    ]
    obstacles = obstacles_box + obstacles_circle
    return obstacles


def plot_env(ax, alpha=1):
    '''
    Plots the environment on matplotlib figure
    :param ax: The axis on which to plot the path.
    '''
    assert len(RandomWorld.xy)>0 or len(RandomWorld.xy_circle)>0, "Call the set_env first"

    # Initialize the position of obstacles
    dimensions = [box_length, box_width]
    rectangle_corner = np.r_[(-dimensions[0]/2, -dimensions[1]/2)]  
    
    # ax.set_xlim((-0.750, 10.5))
    # ax.set_ylim((-0.750, 10.5))
    ax.set_xlim((-2, 12))
    ax.set_ylim((-2, 12))


    for xy_i in RandomWorld.xy_circle:
        plt_cir = plt.Circle(xy_i, radius=cir_radius, color='r', alpha=alpha)
        ax.add_patch(plt_cir)

    for xy_i in RandomWorld.xy:
        plt_box = plt.Rectangle(xy_i+rectangle_corner, dimensions[0], dimensions[1], color='r', alpha=alpha)
        ax.add_patch(plt_box)