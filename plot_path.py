# Plot the path's with different probability threshold.
from config import xy, box_width, box_length

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from models import point

# TODO: Find a way to ensure that both environments used for 
# TODO: generating the path and plotting are the same.
# Set up the environment
A, B = point.set_env()

# Plot the obstacles
fig, ax = plt.subplots(figsize=(10,10))
sns.set()
ax.set_xlim((-0.2, 10.2))
ax.set_ylim((-0.2, 10.2))

# Initialize the position of obstacles
dimensions = [box_length, box_width]
rectangle_corner = np.r_[(-dimensions[0]/2, -dimensions[1]/2)]  

for xy_i in point.xy_circle:
    plt_cir = plt.Circle(xy_i, radius=0.2, color='r')
    ax.add_patch(plt_cir)

for xy_i in point.xy:
    plt_box = plt.Rectangle(xy_i+rectangle_corner, dimensions[0], dimensions[1], color='r')
    ax.add_patch(plt_box)

path_nums = [1, 2, 4 ,3]
plt_cir_goal = plt.Circle(np.r_[9.0, 9.0], radius=0.5, alpha=0.5)
ax.add_patch(plt_cir_goal)
for path_num in path_nums:
    # File saved as raw binary
    with open("path_{}.pkl".format(path_num),"rb") as f:
        data = pickle.load(f)
    ax.plot(data['Path'][:,0], data['Path'][:,1], label=str(data['ProbabilityThreshold']))

plt.legend()
