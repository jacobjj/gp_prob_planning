# Navigation through SST
from config import box_width, box_length, xy

import GPy
import numpy as np
import pybullet as p
from scipy import stats
from scipy import optimize
from sparse_rrt.systems.system_interface import BaseSystem

from models import point
from gp_model import get_model, return_collision_prob, return_collision_prob_line
from gp_model import return_collision_deterministic_line

obstacles, robot = point.set_env()

# Define gaussian model
m = get_model(robot, obstacles, point)
true_K = m.kern.K(m.X)
prior_var = 1e-6
K_inv = np.linalg.inv(true_K+np.eye(true_K.shape[0])*prior_var)
weights = K_inv@m.Y
k_x_x = m.kern.K(np.c_[0,0])

# Define LTI system
A,B = point.get_dyn()

from sparse_rrt.systems.system_interface import IDistance

class EucledianDistance(IDistance):
    '''
    Computes the Eucledian distance between the states
    '''

    def distance(self, point1, point2):
        '''
        Computes the eucledian distance between the two points.
        :param point1: A numpy.array
        :param point2: A numpy.array
        return float: The L2 distance between the points.
        '''
        return np.linalg.norm(point1[:2]-point2[:2])


class FreeDisc(BaseSystem):
    '''
    A simple 2D environment, that uses pybullet as it's backend to simulate the robot
    '''
    MIN_X, MAX_X = -0.1, 10.1
    MIN_Y, MAX_Y = -0.1, 10.1
    MIN_V, MAX_V = -1.0, 1.0

    def __init__(self, ProbabilityThreshold):
        self.ProbabilityThreshold = ProbabilityThreshold
        N = stats.norm(scale=np.sqrt(1/2))
        self.c = N.ppf(1-ProbabilityThreshold)
        super().__init__()


    def propagate(self, start_state, control, num_steps, integration_step):
        '''
        Propogates the robot forward and checks probabilistic collision by
        using deterministic bounds.
        :param start_state: numpy array with the start state for the integration
        :param control: numpy array with constant controls to be applied during integration
        :param num_steps: number of steps to integrate
        :param integration_step: dt of integration
        :return: new state of the system if valid or return None
        '''
        # TODO: 1. To check collision using the constraints equations
        x_mu = start_state[:3]
        x_sigma = start_state[3:].reshape((3,3))
        control = control*num_steps

        def G(x, *args):
            '''
            The ratio of mean by variance function.
            :param x: the state of the robot
            :returns float: E[g(x)]/sqrt(2var(g(x)))
            '''
            if x.ndim!=m.X.ndim:
                x = x[None, :]
            k_star = m.kern.K(x, m.X)
            var = k_x_x - k_star@K_inv@k_star.T
            return (k_star@weights/np.sqrt(2*var))[0]

        r0 = np.linalg.norm(A@x_mu.T+B@control)
        ri = r0*(np.exp(integration_step)-1)
        state_min, state_max = x_mu -ri, x_mu + ri
        sol = optimize.shgo(
            G,
            bounds = [(state_min[0], state_max[0]),(state_min[1], state_max[1])],
            constraints = [
                 {
                     'type':'ineq',
                     'fun': lambda x: ri - np.linalg.norm(x-x_mu[:2]),
                     'jac': lambda x: -(x-x_mu[:2])
                 }
            ]
        )
        if sol.fun<=self.c:
            return None

        # # Check empherically the safety criteria
        # c_hat = return_collision_deterministic_line(x_mu, x_sigma, control, m, A, B)
        # if c_hat<=self.c:
        #     return None

        x_t_1_mu = A@x_mu.T + (B@control)
        # check if bounds are reached
        if all(x_t_1_mu>=[10.1]*3) or all(x_t_1_mu<=[-0.1]*3):
            return None
        x_t_1_sigma = A@x_sigma@A.T
        return np.r_[np.squeeze(x_t_1_mu), np.ravel(x_t_1_sigma)]
        
    def visualize_point(self, state):
        '''
        Project the point onto a 3D-space
        :param state : numpy array of the state point
        :return x,y of visulaization co-ordinates for this state point.
        '''
        x = (state[0]-self.MIN_X)/(self.MAX_X-self.MIN_X)
        y = (state[1]-self.MIN_Y)/(self.MAX_Y-self.MIN_Y)
        return x,y

    def distance_computer(self):
        '''
        Use the redefined module to compute distance
        '''
        return EucledianDistance()

    
    def get_state_bounds(self):
        '''
        Return bounds for the state space
        :return: list of (min, max) bounds for each coordinate in the state space
        '''
        return [
            (self.MIN_X, self.MAX_X), 
            (self.MIN_Y, self.MAX_Y),
            (self.MIN_X, self.MAX_X)
            ] + [(0, 10)]*9

    def get_control_bounds(self):
        '''
        Return bounds for the control space
        :return: list of (min, max) bounds for each coordinate in the control space
        '''
        return [
            (self.MIN_V, self.MAX_V),
            (self.MIN_V, self.MAX_V),
            (self.MIN_V, self.MAX_V)
        ]

    def is_circular_topology(self):
        '''
        Indicate whether state system has planar or circular topology
        :return: boolean flag for each coordinate (False for planar topology)
        '''
        return [False]*12


from sparse_rrt.planners import SST
ProbabilityThreshold = 0.25
path_num = 4
system = FreeDisc(ProbabilityThreshold)

start_state = [0.0, 0.0, 0.1]
goal_state = [9.0, 9.0, 0.1]

# planner = SST(
#     state_bounds=system.get_state_bounds(),
#     control_bounds=system.get_control_bounds(),
#     distance=system.distance_computer(),
#     start_state=np.r_[start_state, np.eye(3).ravel()*0.1],
#     goal_state=np.r_[goal_state, np.eye(3).ravel()*0.1],
#     goal_radius=0.5,
#     random_seed=1,
#     sst_delta_near=0.4,
#     sst_delta_drain=0.2
# )

# for iteration in range(5000):
#     planner.step(system, 1, 10, 0.1)
#     if iteration%500 ==0:
#         solution = planner.get_solution()
#         if solution:
#             print("Solution Cost: {}, Number of nodes: {}".format(np.sum(solution[2]), planner.get_number_of_nodes()))
#         else:
#             print("Number of nodes: {}".format(planner.get_number_of_nodes()))
            
# # Save path
# import pickle
# if solution:
#     data = {
#         'ProbabilityThreshold': ProbabilityThreshold, 
#         'Path': solution[0],
#         'control': solution[1]
#     }
#     with open("path_{}.pkl".format(path_num),"wb") as f:
#         pickle.dump(data, f)
# else:
#     print("No solution found")

    
from sparse_rrt.experiments.experiment_utils import run_config

config = dict(
    system = system,
    number_of_iterations=5000,
    integration_step=0.1,
    min_time_steps=1.0,
    max_time_steps=2.0,
    start_state = np.r_[start_state, np.eye(3).ravel()*0.1],
    goal_state = np.r_[goal_state, np.eye(3).ravel()*0.1],
    goal_radius = 0.5,
    planner = 'sst',
    random_seed = 0,
    sst_delta_near = 0.4,
    sst_delta_drain=0.2,
    debug_period= 200,
    display_type=None
)

run_config(config)

