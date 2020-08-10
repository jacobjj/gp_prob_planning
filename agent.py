# Simulate an agent
import numpy as np

class DTS_agent():
    '''
    A discrete-time system, with random transition matrix.
    '''
    def __init__(self):
        '''
        Initialize the class with random A and B vectors
        '''
        self.A = np.random.uniform(size=(2,2))
        self.B = np.random.uniform(size=(2,2))
        self.x = np.random.uniform(size=(2,1))

    def step(self, u):
        '''
        Generate the next state.
        :param u: The control input
        :returns np.ndarray: Return the next state of the robot.
        '''
        # TODO: Add some noise to the moving
        self.x = self.A@self.x + self.B@u # + some noise.

        return self.x

    def reset(self):
        '''
        Reset the initial state of the robot
        :returns np.ndarray: Return the reset state of the robot.
        '''
        self.x = np.random.uniform(size=(2,1))
        return self.x
