#LEFKI MEIDI
# L3INFO G2
from enum import Enum
import gymnasium as gym
import numpy as np

class GameAgent:
        
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.01,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.01,
        final_epsilon: float = 0.1,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-acttion values (q_values), a learning rate and an epsilon.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
         - `state` is a tuple depending of the game
         - `action` is a number for an action
        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        if isinstance(env.action_space, gym.spaces.Box):
            nb_actions = env.action_space.shape[0]
        else:
            nb_actions = env.action_space.n
        print("action space=",nb_actions)
        print("state space=",env.observation_space)
        self.alpha = learning_rate
        self.gamma = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.q_values = dict()


    def set_hyperparameters(self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float):
        """Set hyperparameters

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
        """
        global nb_actions
        nb_actions = self.env.action_space.n

        self.alpha = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q_values`, return 0.
        """
        if (state,action) not in self.q_values:
            self.q_values[(state,action)] = 0.0
        
        return self.q_values[(state,action)]

    def update_q_value(self, state: tuple, action: int, reward: float, next_state: tuple):
        """
        Update the Q-value for the state `state` and the action `action`
        given a current reward `reward` and the next state
        and an estiamte of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and the discounted estimated future reward.
        """
        facteur = self.alpha * ( reward + ( self.gamma * self.best_future_reward(next_state) - self.get_q_value(state,action) )  )

        self.q_values[(state,action)] = self.q_values[(state,action)] + facteur

    def best_future_reward(self, state: tuple) -> float:
        """
        Given a state `state`, consider all possible '(state, action)'
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `state`, return 0.
        """
        max_future_reward = -float('inf')  
        for action in range(self.env.action_space.n):
            future_q_value = self.get_q_value(state, action)
            if future_q_value > max_future_reward:
                max_future_reward = future_q_value
        return max_future_reward


    def get_action(self, state: tuple) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.

        Function used by train.py
        """
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            max = -1000
            action = None

            for i in range(self.env.action_space.n):
                ancienne_qvalue = self.get_q_value(state,i)
                if (ancienne_qvalue > max):
                    action = i
                
        return action

    def get_determinist_action(self, state: np.ndarray) -> int:
        """
        Returns the best action for a state `state` using
        Q-values
        Function used by play.py
        """
        max = -1000
        action = None

        for i in range(self.env.action_space.n):
            ancienne_qvalue = self.get_q_value(state,i)
            if (ancienne_qvalue > max):
                max = ancienne_qvalue
                action = i
                
        return action


class Game(Enum):
    frozen = 'frozen_lake'
    frozen8 = 'frozen_lake8'
    taxi = 'taxi'
    cart = 'cart_pole'

    def __str__(self):
        return self.value
    
    def getEnv(s, show):
        """
        Create a Gymnasium environment associated with
        the game of name s
        
        Args:
            s: game name (as a string)
            show: displays the Game if show is True; does not show
            (computationnaly more efficient for training) otherwises
        """
        render_mode = None
        if show:
            render_mode = "human"

        if s == Game.cart:
            return gym.make('CartPole-v1', render_mode=render_mode)
        elif s == Game.frozen8:
            return gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode=render_mode)
        elif s == Game.taxi:
            return gym.make('Taxi-v3', render_mode=render_mode)
        else:
            return gym.make('FrozenLake-v1', is_slippery=False, render_mode=render_mode)
        
        
    def discretizeState(state):
        """
        Discretize states for Cart Pole following a grid defined by lower and upper
        bounds and the number of bins for each of the 4 input values
        Returns a 4-dimensional type defining the indices for each of the 4 input
        values

        Args:
            stage: numpy.ndarray with 4 entries:
               cart position, cart velocity, pole angle, and pole angular velocity 
        """

        # see value ranges for entries:
        # https://gymnasium.farama.org/environments/classic_control/cart_pole/
        cartPositionMin=-4.8
        cartPositionMax=4.8
        cartVelocityMin=-3
        cartVelocityMax=3
        poleAngleMin=-0.418
        poleAngleMax=0.418
        poleAngleVelocityMin=-10
        poleAngleVelocityMax=10
        numberOfBins=25

        cartPositionBin=np.linspace(cartPositionMin,cartPositionMax,numberOfBins)
        cartVelocityBin=np.linspace(cartVelocityMin,cartVelocityMax,numberOfBins)
        poleAngleBin=np.linspace(poleAngleMin,poleAngleMax,numberOfBins)
        poleAngleVelocityBin=np.linspace(poleAngleVelocityMin,poleAngleVelocityMax,numberOfBins)
         
        indexPosition=np.maximum(np.digitize(state[0],cartPositionBin)-1,0)
        indexVelocity=np.maximum(np.digitize(state[1],cartVelocityBin)-1,0)
        indexAngle=np.maximum(np.digitize(state[2],poleAngleBin)-1,0)
        indexAngularVelocity=np.maximum(np.digitize(state[3],poleAngleVelocityBin)-1,0)
         
        return tuple([indexPosition,indexVelocity,indexAngle,indexAngularVelocity])   
