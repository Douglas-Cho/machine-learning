import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense, LSTM, Input, Lambda, Add
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras import backend as K

EPISODES = 500


# DRQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DDDRQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DRQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        self.memory_size = 2000
        
        # create replay memory using deque
        self.memory = Memory(self.memory_size)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_dddrqn_per_bi.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        state_input = Input(shape=(self.state_size, 2))

        # state value tower - V
        state_value = LSTM(16, kernel_initializer='orthogonal', recurrent_initializer='zeros')(state_input)
        state_value = Dense(1, init='uniform')(state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(self.action_size,))(state_value)

        # action advantage tower - A
        action_advantage = LSTM(16, kernel_initializer='orthogonal', recurrent_initializer='zeros')(state_input)
        action_advantage = Dense(self.action_size)(action_advantage)
        action_advantage = Lambda(lambda a: a[:, :] - K.max(a[:, :], keepdims=True), output_shape=(self.action_size,))(action_advantage)

        # merge to state-action value function Q
        state_action_value = Add()([state_value, action_advantage])

        model = Model(input=state_input, output=state_action_value)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model
    
    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        if self.epsilon == 1:
            done = True

        # get TD-error and save in memory 
        target = self.model.predict(state)
        old_val = target[0][action]
        target_val = self.target_model.predict(next_state)
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * (
                np.amax(target_val[0]))
        error = abs(old_val - target[0][action])

        self.memory.add(error, (state, action, reward, next_state, done))

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        mini_batch = self.memory.sample(self.batch_size)

        errors = np.zeros(self.batch_size)
        update_input = np.zeros((self.batch_size, self.state_size, 2))
        update_target = np.zeros((self.batch_size, self.state_size, 2))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][1][0]
            action.append(mini_batch[i][1][1])
            reward.append(mini_batch[i][1][2])
            update_target[i] = mini_batch[i][1][3]
            done.append(mini_batch[i][1][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            old_val = target[i][action[i]]
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])
            # Save TD-error
            errors[i] = abs(old_val - target[i][action[i]])

        # priority update with TD-error
        for i in range(self.batch_size):
            idx = mini_batch[i][0]
            self.memory.update(idx, errors[i])

        # and do the model fit!      
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('CartPole-v1')
    
    # Total number of states to use
    number_of_states = 4
    
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    expanded_state_size = state_size * number_of_states
    action_size = env.action_space.n

    agent = DDDRQNAgent(expanded_state_size, action_size)

    scores, episodes = [], []
    
    step = 0
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        
        # expand the state with past states and initialize
        expanded_state = np.zeros(expanded_state_size)
        expanded_next_state = np.zeros(expanded_state_size)
        for h in range(state_size):
            expanded_state[(h + 1) * number_of_states -1] = state[h]
 
        # reshape states for LSTM input without embedding layer
        reshaped_state = np.zeros((1, expanded_state_size, 2))
        for i in range(expanded_state_size):
            for j in range(2):
                reshaped_state[0, i, j] = expanded_state[i]
        
        while not done:
            if agent.render:
                env.render()
            step += 1

            # get action for the current state and go one step in environment
            action = agent.get_action(reshaped_state)
            next_state, reward, done, info = env.step(action)
            
            # update the expanded next state with next state values
            for h in range(state_size):
                expanded_next_state[(h + 1) * number_of_states -1] = next_state[h]
            
            # reshape expanded next state for LSTM input without embedding layer
            reshaped_next_state = np.zeros((1, expanded_state_size, 2))
            for i in range(expanded_state_size):
                for j in range(2):
                    reshaped_next_state[0, i, j] = expanded_next_state[i]
                    
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(reshaped_state, action, reward, reshaped_next_state, done)
            
            # every time step do the training
            if step >= agent.train_start:
                agent.train_model()
                
            score += reward
            reshaped_state = reshaped_next_state
            
            # Shifting past state elements to the left by one
            expanded_next_state = np.roll(expanded_next_state, -1)

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_dddrqn_per_bi.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      step if step <= agent.memory_size else agent.memory_size, "  epsilon:", agent.epsilon)

            # if the mean of scores of last 10 episode is bigger than 490
            # stop training
            # revised to exit cleanly on Jupiter notebook
            if np.mean(scores[-min(10, len(scores)):]) > 490:
                #sys.exit()
                env.close()
                break

        # save the model
        if e % 50 == 0:
            agent.model.save_weights("./save_model/cartpole_dddrqn_per_bi.h5")
