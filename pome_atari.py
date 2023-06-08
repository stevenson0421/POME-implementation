import torch

import numpy as np
import scipy
import math

import gymnasium

import time
import cv2

'''
Functions
'''

'''
calclate discounted cumulated sum for a sequence (np array or tensor) and a discount factor
'''
def discounted_cumulated_sum(sequence, discount_factor):
    if isinstance(sequence, np.ndarray):
            return scipy.signal.lfilter([1], [1, float(-discount_factor)], sequence[::-1], axis=0)[::-1]
    elif isinstance(sequence, torch.Tensor):
            return torch.as_tensor(np.ascontiguousarray(scipy.signal.lfilter([1], [1, float(-discount_factor)], sequence.detach().numpy()[::-1], axis=0)[::-1]), dtype=torch.float32)
    else:
            raise TypeError

'''
process state from environment

current implementation is add a dimension for one channel since the state is grayscale image
'''
def process_state(state, new_size):
    state = cv2.resize(state, new_size, interpolation=cv2.INTER_AREA)
    state_expand = np.expand_dims(state, 0)
    state_normalized = state_expand / 255.0

    return state_normalized

'''
set random seeds for reproducibility
'''
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

'''
Data Buffer
'''

class Buffer:
    '''
    Buffer for storing information of trajectories experienced by POME agent,
    including state, action, qf (TD target), reward, reward-to-go (discounted cumulated reward), value (estimation), action probability in log
    '''
    def __init__(self, size, state_dimension, action_dimension, discount_factor=0.99):
        self.state_buffer = np.zeros(tuple([size]+list(state_dimension)), dtype=np.float32)
        self.action_buffer = np.zeros(tuple([size]+list(action_dimension)), dtype=np.float32)
        self.qf_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.reward_to_go_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.action_logprobability_buffer = np.zeros(size, dtype=np.float32)

        self.discount_factor = discount_factor
        self.pointer = 0
        self.start_index = 0
        self.max_size = size
    
    '''
    store informations of one state-action pair
    '''
    def store(self, state, action, reward, value, action_logprobability):
        assert self.pointer < self.max_size

        self.state_buffer[self.pointer] = state
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.action_logprobability_buffer[self.pointer] = action_logprobability

        self.pointer += 1

    '''
    call at the end of a trajectory, process further informations
    last_val should be 0 if the trajectory ended; otherwise value estimation for the last state
    '''
    def done(self, last_value=0):
        trajectory_slice = slice(self.start_index, self.pointer)
        reward_list = np.append(self.reward_buffer[trajectory_slice], last_value)
        value_list = np.append(self.value_buffer[trajectory_slice], last_value)

        # calculate Q_ft in paper
        self.qf_buffer[trajectory_slice] = reward_list[:-1] + self.discount_factor * value_list[1:]

        # calculate reward-to-go (discounted cumulated reward)
        self.reward_to_go_buffer[trajectory_slice] = discounted_cumulated_sum(reward_list, self.discount_factor)[:-1]

        # reset start index
        self.start_index = self.pointer

    '''
    get all data from the buffer
    '''
    def get(self):
        assert self.pointer == self.max_size

        # reset buffer
        self.pointer = 0
        self.start_index = 0

        data = dict(state=self.state_buffer,
                    action=self.action_buffer,
                    qf=self.qf_buffer,
                    reward=self.reward_buffer,
                    reward_to_go=self.reward_to_go_buffer,
                    value=self.value_buffer,
                    action_logprobability=self.action_logprobability_buffer)
        return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in data.items()}

'''
Policy Network

current implementation
input state 1x210x160
conv2d (output dim=16, kernel_size=8, stride=4, no padding) 16x51x39
conv2d (output dim=32, kernel_size=4, stride=2, no padding) 32x24x18
linear (output dim=256) ## note that output of this layer is also propogated to value network as input
linear (output dim=6) ## output one-hot action

output includes latent, policy and policy distribution
'''
class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dimension, action_dimension):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=state_dimension[0],
                                      out_channels=16,
                                      kernel_size=8,
                                      stride=4,
                                      padding='valid')
        conv1_dimension = (math.floor((state_dimension[1]-8)/4)+1, math.floor((state_dimension[2]-8)/4)+1)
        self.conv2 = torch.nn.Conv2d(in_channels=16,
                                      out_channels=32,
                                      kernel_size=4,
                                      stride=2,
                                      padding='valid')
        conv2_dimension = (math.floor((conv1_dimension[0]-4)/2)+1, math.floor((conv1_dimension[1]-4)/2)+1)
        self.fc1 = torch.nn.Linear(in_features=conv2_dimension[0]*conv2_dimension[1]*32,
                                      out_features=256)
        self.fc2 = torch.nn.Linear(in_features=256,
                                       out_features=action_dimension)
        self.activation = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()

    def forward(self, state):
        latent = self.activation(self.conv1(state))
        latent = self.activation(self.conv2(latent))
        latent = self.flatten(latent)
        latent = self.activation(self.fc1(latent))
        policy = self.activation(self.fc2(latent))

        policy_distribution = torch.distributions.Categorical(logits=policy)

        return latent, policy, policy_distribution


'''
Value Network

current implementation
input latent from policy network 1x256
linear (output dim=1) ## output value
'''
class ValueNetwork(torch.nn.Module):
    def __init__(self, latent_dimension=256):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_features=latent_dimension,
                                      out_features=1)
        self.activation = torch.nn.ReLU()

    def forward(self, latent):
        value = self.activation(self.fc1(latent))

        return value
        
'''
Reward Network

current implementation (not mentioned in paper)
input state + action 1x211x160 (action is one-hot encoded to dimension 1x1x160)
conv2d (output dim=16, kernel_size=8, stride=4, no padding) 16x51x39
linear (output dim=1) ## output reward
'''
class RewardNetwork(torch.nn.Module):
    def __init__(self, state_dimension):
        super().__init__()

        state_action_dimension = (state_dimension[0], state_dimension[1]+1, state_dimension[2])
        self.conv1 = torch.nn.Conv2d(in_channels=state_action_dimension[0],
                                      out_channels=16,
                                      kernel_size=8,
                                      stride=4,
                                      padding='valid')
        conv1_dimension = (math.floor((state_action_dimension[1]-8)/4)+1, math.floor((state_action_dimension[2]-8)/4)+1)
        self.fc1 = torch.nn.Linear(in_features=conv1_dimension[0]*conv1_dimension[1]*16,
                                      out_features=1)
        self.flatten = torch.nn.Flatten()
    
    def forward(self, state, action):
        # concatenate state and action
        action_one_hot = torch.nn.functional.one_hot(action.to(torch.int64), num_classes=state.shape[3])
        action_one_hot = action_one_hot.unsqueeze(1).unsqueeze(1)
        state_action = torch.cat((state, action_one_hot), dim=2)

        reward = self.conv1(state_action)
        reward = self.flatten(reward)
        reward = self.fc1(reward)

        return reward

'''
Transition Network

current implementation (parameters not mentioned in paper)
input state + action 1x211x160 (action is one-hot encoded to dimension 1x1x160)
conv2d (output dim=16, kernel_size=8, stride=4, no padding) 16x51x39
linear (output dim=210x160) output new state ## note that this is a flattened image
'''
class TransitionNetwork(torch.nn.Module):
    def __init__(self, state_dimension):
        super().__init__()

        state_action_dimension = (state_dimension[0], state_dimension[1]+1, state_dimension[2])
        self.conv1 = torch.nn.Conv2d(in_channels=state_action_dimension[0],
                                      out_channels=16,
                                      kernel_size=8,
                                      stride=4,
                                      padding='valid')
        conv1_dimension = (math.floor((state_action_dimension[1]-8)/4)+1, math.floor((state_action_dimension[2]-8)/4)+1)
        self.fc1 = torch.nn.Linear(in_features=conv1_dimension[0]*conv1_dimension[1]*16,
                                      out_features=state_dimension[1]*state_dimension[2])
        self.flatten = torch.nn.Flatten()
    
    def forward(self, state, action):
        # concatenate state and action
        action_one_hot = torch.nn.functional.one_hot(action.to(torch.int64), num_classes=state.shape[3])
        action_one_hot = action_one_hot.unsqueeze(1).unsqueeze(1)
        state_action = torch.cat((state, action_one_hot), dim=2)

        transition = self.conv1(state_action)
        transition = self.flatten(transition)
        transition = self.fc1(transition)

        return transition

'''
Network

combine four networks

the functions can be further optimized
'''
class Network(torch.nn.Module):
    def __init__(self, state_dimension, action_dimension):
        super().__init__()

        self.policy_network = PolicyNetwork(state_dimension, action_dimension)
        self.value_network = ValueNetwork()
        self.reward_network = RewardNetwork(state_dimension)
        self.transition_network = TransitionNetwork(state_dimension)

    '''
    get the log probability of an action
    '''
    def get_action_logprobability(self, policy_distribution, action):
        action_logprobability = policy_distribution.log_prob(action)
        
        return action_logprobability

    '''
    get action, log probability of the action and value estimation from a state
    '''
    def step(self, state):
        with torch.no_grad():
            latent, policy, policy_distribution = self.policy_network(state)
            
            action = policy_distribution.sample()
            action_logprobability = policy_distribution.log_prob(action)
            value = self.value_network(latent)
        return action.numpy(), action_logprobability.numpy(), value.numpy()
    
    '''
    get action only from a state

    this function is not used yet, maybe for evaluation
    '''
    def act(self, state):
        with torch.no_grad():
            latent, policy, policy_distribution = self.policy_network(state)
            action = policy_distribution.sample()
        return action.numpy()


'''
pome

main implementation of the method

input parameters

    environment: gym environment ex. gym.make('PongNoFrameskip-v4')

    state_new_size: states of environment are resized to this new size

    networkclass: combined network

    number_of_epoch: the number of epoch in training process

    steps_per_trajectory: steps of a trajectory (k in paper)

    max_steps_per_trajectory: max steps of a trajectory, basically the time limit of agent (truncated if set)

    pome_learning_rate: learning rate of pome object, consist of policy, value and transition

    reward_learning_rate: learning rate of reward estimation, ## note that this is not mentioned in paper

    train_pome_iterations: how many iterations to update pome parameters in one update ## note that this is not mentioned in paper
    
    train_reward_iterations: how many iterations to update reward parameters in one update ## note that this is not mentioned in paper

    discount_factor: discount factor (gamma) of computing discounted cumulated sum

    alpha: coefficient used in POME TD error

    clip_ratio: clip ratio for POME object before combined (l^POME in paper)

    value_loss_ratio: coefficient of value loss for combined POME object

    transition_loss_ratio: coefficient of transition loss for combined POME object

    seed: seed for randomization

'''
def pome(environment,
         state_new_size,
         networkclass,
         number_of_epoch,
         steps_per_trajectory,
         max_steps_per_trajectory,
         pome_learning_rate,
         reward_learning_rate,
         train_pome_iterations,
         train_reward_iterations,
         discount_factor,
         alpha,
         clip_ratio,
         value_loss_ratio,
         transition_loss_ratio,
         seed
         ):
    
    assert isinstance(environment.action_space, gymnasium.spaces.Discrete)

    set_seed(seed)
    
    # current implementation (210, 160)
    # state_space = environment.observation_space
    state_dimension = (1, state_new_size[0], state_new_size[1])
    # current implementation Discrete(6)
    action_dimension = environment.action_space.n
    
    network = networkclass(state_dimension, action_dimension)

    # state dimension is (1, 210, 160), action dimension is 0 (scalar)
    buffer = Buffer(steps_per_trajectory, state_dimension, environment.action_space.shape, discount_factor)

    '''
    pome loss before combined

    note that the KL divergence in paper is not included:   
        1. I think this is duplicated with the clip trick
        2. the value of coefficient Beta is not mentioned

    note that this should be modified to combined version
    '''
    def get_pome_loss(data):
        state = data['state']
        action = data['action']
        qf = data['qf']
        reward_to_go = data['reward_to_go']
        
        # pi_old, note that this is fixed during one epoch update
        action_logprobability_old = data['action_logprobability']

        # get policy, action probability in log and value
        latent, policy, policy_distribution = network.policy_network(state)
        value = network.value_network(latent)
        action_logprobability = policy_distribution.log_prob(action)

        # calculate Q_b_t in paper
        reward_hat = network.reward_network(state, action)
        transition = network.transition_network(state, action)
        transition_reshaped = transition.reshape(state.shape)
        latent_value, _, _ = network.policy_network(transition_reshaped)
        qb = reward_hat + discount_factor * network.value_network(latent_value)

        # calculate pi / pi_old
        policy_ratio = torch.exp(action_logprobability-action_logprobability_old)

        # calculate epsilon and epsilon_bar in paper
        epsilon = torch.abs(qf - qb)
        epsilon_median = torch.median(epsilon)
        delta_t = torch.abs(qf - value)
        delta_t_pome = qf + alpha * torch.clamp(epsilon-epsilon_median, -delta_t, delta_t) - value

        # calculate a_t_pome in paper
        a_t_pome = discounted_cumulated_sum(delta_t_pome, 1)

        # calculate pome loss in paper
        clip_a_t_pome = torch.clamp(policy_ratio, 1-clip_ratio, 1+clip_ratio) * a_t_pome
        pome_loss = -(torch.min(policy_ratio * a_t_pome, clip_a_t_pome)).mean()

        # calculate value loss in paper. Note that object is normalized
        value_object_core = a_t_pome + reward_to_go - value
        value_object_core = (value_object_core - value_object_core.mean()) / value_object_core.std()
        value_loss = ((value_object_core) ** 2).mean()

        # calculate transition loss in paper. note that state of 0:-1 are used for calculation, not sure if this is correct
        transition = network.transition_network(state[:-1], action[:-1])
        transition_reshaped = transition.reshape((state.shape[0]-1, state.shape[1], state.shape[2], state.shape[3]))

        transition_loss = (torch.norm(state[:-1]-transition_reshaped, p=2) ** 2) / torch.numel(state)

        return pome_loss, value_loss, transition_loss
    
    
    '''
    get reward loss

    note that this should be updated seperated with pome loss
    '''
    def get_reward_loss(data):
        state = data['state']
        action = data['action']
        reward = data['reward']

        reward_hat = network.reward_network(state, action)

        reward_object_core = reward - reward_hat

        # calculate reward loss in paper
        reward_loss = ((torch.sum(reward_object_core) / state.shape[0]) ** 2)

        return reward_loss
    
    '''
    get transition loss

    note that state of 0:-1 are used for calculation, not sure if this is correct
    '''
    def get_transition_loss(data):
        state = data['state']
        action = data['action']
        transition = network.transition_network(state[:-1], action[:-1])
        transition_reshaped = transition.reshape((state.shape[0]-1, state.shape[1], state.shape[2], state.shape[3]))

        # calculate transition loss in paper
        transition_loss = (torch.norm(state[:-1]-transition_reshaped, p=2) ** 2) / torch.numel(state)

        return transition_loss

    pome_optimizer = torch.optim.Adam([{'params':network.policy_network.parameters()},
                                       {'params':network.value_network.parameters()},
                                       {'params':network.transition_network.parameters()},], lr=pome_learning_rate)
    # learning rate is linearly annealed [1, 0]
    torch.optim.lr_scheduler.LinearLR(optimizer=pome_optimizer, start_factor=1., end_factor=0.)

    reward_optimizer = torch.optim.Adam(params=network.reward_network.parameters(), lr=reward_learning_rate)
    
    '''
    update all network parameters in paper. first update pome, then reward
    '''
    def update():
        # get all data from buffer, which should be a trajectory
        data = buffer.get()

        # update pome parameters
        for i in range(train_pome_iterations):
            pome_optimizer.zero_grad()

            pome_loss, value_loss, transition_loss = get_pome_loss(data)

            print(f'pome_loss: {pome_loss}, value_loss: {value_loss}, transition_loss: {transition_loss}')

            total_pome_loss = pome_loss + value_loss_ratio * value_loss + transition_loss_ratio * transition_loss
            total_pome_loss.backward()

            pome_optimizer.step()
        
        # update reward parameters
        for i in range(train_reward_iterations):
            reward_optimizer.zero_grad()

            reward_loss = get_reward_loss(data)

            print(f'reward_loss: {reward_loss}')
            reward_loss.backward()

            reward_optimizer.step()

    # main process
    trajectory_reward = 0
    trajectory_length = 0
    trajectory_rewards = []

    start_time = time.time()
    state, info = environment.reset()
    state = process_state(state, state_new_size)
    
    for epoch in range(number_of_epoch):
        for t in range(steps_per_trajectory):
            # get action, action probability in log, value from a state. note that state is unsqueezed as a batch with size 1
            action, action_logprobability, value = network.step(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            # get scalar
            action = action.item()
            action_logprobability = action_logprobability.item()
            value = value.item()
            # agent interaction
            next_state, reward, terminated, truncated, info = environment.step(action)
            trajectory_reward += reward
            trajectory_length += 1
            # store all data to buffer
            buffer.store(state, action, reward, value, action_logprobability)
            # update state. this is important!
            state = process_state(next_state, state_new_size)
            # set timeout for agent
            timeout = (trajectory_length == max_steps_per_trajectory)
            done = (terminated or truncated or timeout)
            # epoch is ended with full steps
            epoch_ended = (t == steps_per_trajectory-1)
            # end trajectory. note that timeout and epoch_ended are same if we set the same boundary
            if done or epoch_ended:
                if epoch_ended and not done:
                    print('Warning: trajectory cut off by epoch at %d steps.'%trajectory_length)

                if timeout or epoch_ended:
                    _, last_value, _ = network.step(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                else:
                    last_value = 0
                buffer.done(last_value)
                # reset the environment. this is important!
                state = environment.reset()
                state = process_state(next_state, state_new_size)
                # record total reward and reset
                print(f'Final step {t}: trajectory_reward: {trajectory_reward}, trajectory_length: {trajectory_length}')
                trajectory_rewards.append(trajectory_reward)
                trajectory_reward = 0
                trajectory_length = 0
        
        update()
    print(f'Train Time: {(time.time() - start_time):2f} seconds')
    # average reward of last 100 trajectories in paper
    print(f'Train Score: {np.mean(trajectory_rewards[-100:])}')

if __name__ == '__main__':
    pome(environment=gymnasium.make('PongNoFrameskip-v4', obs_type='grayscale'),
        state_new_size=(84, 84),
        networkclass=Network,
        number_of_epoch=1,
        steps_per_trajectory=10,
        max_steps_per_trajectory=100,
        pome_learning_rate=2.5*1e-4,
        reward_learning_rate=0.01,
        train_pome_iterations=5,
        train_reward_iterations=5,
        discount_factor=0.99,
        alpha=0.1,
        clip_ratio=0.1,
        value_loss_ratio=1,
        transition_loss_ratio=2,
        seed=24)