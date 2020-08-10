import numpy as np
import fcl

import matplotlib.pyplot as plt
import seaborn as sns

import time

class environment():
    '''
    A 2D environment with obstacles and a robot.
    '''
    def __init__(self):
        '''
        Initialze an environment with a Box and Cylinder
        '''
        # Initialize the Box
        self.box1 = fcl.Box(0.2, 0.2, 0.0)
        T_box = np.random.rand(2)*9 + 0.15

        # Initialie the Cylinder
        self.cylinder1 = fcl.Cylinder(0.3, 0.0)

        self.fig, self.ax = plt.figure()        



def box_collision_obj(dimensions, xy, angle):
    '''
    Returns a fcl-box collision object with the given dimension transformed to a 
    given location and orientation and the corresponding patch for matplotlib
    :param dimension: A tuple specifying the x,y and z of the obstacle  
    :param xy: A 1x2 numpy array specifying the xy translation of the obstacle
    :param angle: A float, specifying the orientation of the obstacle along x axis rotated clockwise
    :returns (fcl.CollisionObject, matplotlib.Rectangle): A box collision object specified with the given rotation and translation
    '''
    assert isinstance(dimensions, tuple), "dimensions has to be of type tuple"
    assert len(dimensions)==3, "dimensions has to be of length 3"
    
    g1 = fcl.Box(*dimensions)
    R_cir = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    R = np.eye(3)
    R[:2,:2] = R_cir
    t1 = fcl.Transform(R, np.r_[xy, 0.0])
    rectangle_corner = np.r_[(-dimensions[0]/2, -dimensions[1]/2)]
    return fcl.CollisionObject(g1, t1), plt.Rectangle(xy+rectangle_corner, dimensions[0], dimensions[1], angle=angle*180/np.pi, color='r')


if __name__=="__main__":
    fig, ax = plt.subplots(figsize=(10,10))
    sns.set()
    ax.set_xlim((0, 10.0))
    ax.set_ylim((0, 10.0))

    # Initialize the Robot
    radius = 0.1
    g2 = fcl.Cylinder(radius, 0.0)
    A = np.eye(2)
    B = np.eye(2)*0.2

    # T_cir = np.random.rand(2)*8 + 1
    T_cir = np.r_[2., 4.]
    T = np.array(np.r_[T_cir, 0])
    plt_cir = plt.Circle(T_cir, radius = radius)
    t2 = fcl.Transform(T)
    o2 = fcl.CollisionObject(g2, t2)

    # Initialize random obstacles
    # obstacles = [box_collision_obj((1., 1., 0.0), np.random.rand(2)*9 + 0.15, np.random.rand()*np.pi/2) for _ in range(5)]

    # Initialize the position of obstacles
    xy = [np.r_[2,2], np.r_[2, 8], np.r_[5,5], np.r_[8, 2], np.r_[8, 8]]
    theta = [0, 0, 0, 0, 0]
    obstacles = [box_collision_obj((1., 1., 0.0), xy_i , theta_i) for xy_i, theta_i in zip(xy,theta)]

    obstacle_grp = [obs[0] for obs in obstacles]

    obs_manager = fcl.DynamicAABBTreeCollisionManager()
    obs_manager.registerObjects(obstacle_grp)
    obs_manager.setup()
    req = fcl.DistanceRequest()

    request = fcl.DistanceRequest()
    result = fcl.DistanceResult()
    ret = fcl.distance(obstacle_grp[2], o2, request=request, result=result)
    print("Penetration Distance:{}".format(ret))

    for _, plt_box in obstacles:
        ax.add_patch(plt_box)

    for _ in range(25):
        # NOTE: This value has to be redefined for each iteration or the done method has to be reset to False
        # for more information refer the docs - https://github.com/BerkeleyAutomation/python-fcl
        ddata = fcl.DistanceData(request=req)  
        T = np.array(np.r_[T_cir, 0])
        plt_cir = plt.Circle(T_cir, radius = 0.2)
        t2 = fcl.Transform(T)
        o2 = fcl.CollisionObject(g2, t2)

        obs_manager.distance(o2, ddata, fcl.defaultDistanceCallback)
        print(ddata.result.min_distance)

        ax.add_patch(plt_cir)
        plt.draw()
        plt.pause(0.3)
        del ax.patches[-1]
        T_cir = A@T_cir + B@np.ones(2) + 0.1*np.random.rand(2)


