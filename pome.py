import torch

import numpy as np
import scipy
import math

import gymnasium

import time

'''
Functions
'''

def discounted_cumulated_sum(sequence, discount_factor):
    if isinstance(sequence, np.ndarray):
            return scipy.signal.lfilter([1], [1, float(-discount_factor)], sequence[::-1], axis=0)[::-1]
    elif isinstance(sequence, torch.Tensor):
            return torch.as_tensor(np.ascontiguousarray(scipy.signal.lfilter([1], [1, float(-discount_factor)], sequence.detach().numpy()[::-1], axis=0)[::-1]), dtype=torch.float32)
    else:
            raise TypeError
    
def process_state(state):
    return np.expand_dims(state, 0)

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
    Buffer for storing information of trajectories experienced by POME agent
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

        # calculate Q_ft in POME
        self.qf_buffer[trajectory_slice] = reward_list[:-1] + self.discount_factor * value_list[1:] - value_list[:-1]

        # calculate reward to go (discounted cumulated reward)
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



class ValueNetwork(torch.nn.Module):
    def __init__(self, latent_dimension=256):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_features=latent_dimension,
                                      out_features=1)
        self.activation = torch.nn.ReLU()

    def forward(self, latent):
        value = self.activation(self.fc1(latent))

        return value
        

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
    
class Network(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()

        assert isinstance(action_space, gymnasium.spaces.Discrete)

        state_dimension = (1, state_space.shape[0], state_space.shape[1])
        action_dimension = action_space.n

        self.policy_network = PolicyNetwork(state_dimension, action_dimension)
        self.value_network = ValueNetwork()
        self.reward_network = RewardNetwork(state_dimension)
        self.transition_network = TransitionNetwork(state_dimension)

    def get_action_logprobability(self, policy_distribution, action):
        action_logprobability = policy_distribution.log_prob(action)
        
        return action_logprobability

    def step(self, state):
        with torch.no_grad():
            latent, policy, policy_distribution = self.policy_network(state)
            
            action = policy_distribution.sample()
            action_logprobability = policy_distribution.log_prob(action)
            value = self.value_network(latent)
        return action.numpy(), action_logprobability.numpy(), value.numpy()
    
    def act(self, state):
        with torch.no_grad():
            latent, policy, policy_distribution = self.policy_network(state)
            action = policy_distribution.sample()
        return action.numpy()

def pome(environment,
         networkclass,
         number_of_epoch,
         steps_per_epoch,
         pome_learning_rate,
         reward_learning_rate,
         train_pome_iterations,
         train_reward_iterations,
         max_trajectory_length,
         discount_factor,
         alpha,
         clip_ratio,
         value_loss_ratio,
         transition_loss_ratio,
         seed
         ):
    
    set_seed(seed)
    
    state_space = environment.observation_space
    action_space = environment.action_space

    network = networkclass(state_space, action_space)

    buffer = Buffer(steps_per_epoch, tuple([1]+list(state_space.shape)), action_space.shape, discount_factor)

    def get_pome_loss(data):
        state = data['state']
        action = data['action']
        qf = data['qf']
        
        action_logprobability_old = data['action_logprobability']

        latent, policy, policy_distribution = network.policy_network(state)
        value = network.value_network(latent)
        action_logprobability = policy_distribution.log_prob(action)

        qb = network.reward_network(state, action)

        policy_ratio = torch.exp(action_logprobability-action_logprobability_old)

        epsilon = torch.abs(qf - qb)
        epsilon_median = torch.median(epsilon)
        delta_t = torch.abs(qf - value)
        delta_t_pome = qf + alpha * torch.clamp(epsilon-epsilon_median, -delta_t, delta_t) - value

        a_t_pome = discounted_cumulated_sum(delta_t_pome, 1)

        clip_a_t_pome = torch.clamp(policy_ratio, 1-clip_ratio, 1+clip_ratio) * a_t_pome
        pome_loss = -(torch.min(policy_ratio * a_t_pome, clip_a_t_pome)).mean()

        other_variable = {'a_t_pome': a_t_pome, 'value': value}

        return pome_loss, other_variable
    
    def get_value_loss(data, other_variable):
        reward_to_go = data['reward_to_go']
        value = other_variable['value']
        a_t_pome = other_variable['a_t_pome']

        value_loss = ((a_t_pome + reward_to_go - value) ** 2).mean()

        return value_loss
    
    def get_reward_loss(data):
        state = data['state']
        action = data['action']
        reward = data['reward']

        reward_hat = network.reward_network(state, action)

        reward_loss = (torch.sum(reward-reward_hat)**2)

        return reward_loss
    
    def get_transition_loss(data):
        state = data['state']
        action = data['action']
        transition = network.transition_network(state[:-1], action[:-1])

        state_reshaped = state[1:].reshape(state.shape[0]-1, state.shape[1], state.shape[2]*state.shape[3])
        transition_loss = (torch.norm(state_reshaped-transition, p=2) ** 2)

        return transition_loss

    pome_optimizer = torch.optim.Adam([{'params':network.policy_network.parameters()},
                                       {'params':network.value_network.parameters()},
                                       {'params':network.transition_network.parameters()},], lr=pome_learning_rate)
    torch.optim.lr_scheduler.LinearLR(optimizer=pome_optimizer, start_factor=1., end_factor=0.)

    reward_optimizer = torch.optim.Adam(params=network.reward_network.parameters(), lr=reward_learning_rate)
    
    def update():
        data = buffer.get()

        for i in range(train_pome_iterations):
            pome_optimizer.zero_grad()

            pome_loss, other_variable = get_pome_loss(data)
            value_loss = get_value_loss(data, other_variable)
            transition_loss = get_transition_loss(data)

            total_pome_loss = pome_loss + value_loss_ratio * value_loss + transition_loss_ratio * transition_loss
            total_pome_loss.backward()

            pome_optimizer.step()
        
        for i in range(train_reward_iterations):
            reward_optimizer.zero_grad()

            reward_loss = get_reward_loss(data)
            reward_loss.backward()

            reward_optimizer.step()


    start_time = time.time()
    state, info = environment.reset()
    state = process_state(state)
    
    trajectory_reward = 0
    trajectory_length = 0
    trajectory_rewards = []

    for epoch in range(number_of_epoch):
        for t in range(steps_per_epoch):
            action, action_logprobability, value = network.step(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            action = action.item()
            action_logprobability = action_logprobability.item()
            value = value.item()
            next_state, reward, terminated, truncated, info = environment.step(action)
            trajectory_reward += reward
            trajectory_length += 1

            buffer.store(state, action, reward, value, action_logprobability)

            state = process_state(next_state)

            timeout = (trajectory_length == max_trajectory_length)
            done = (terminated or truncated or timeout)
            epoch_ended = (t == steps_per_epoch-1)

            if done or epoch_ended:
                if epoch_ended and not done:
                    print('Warning: trajectory cut off by epoch at %d steps.'%trajectory_length)

                if timeout or epoch_ended:
                    _, last_value, _ = network.step(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                else:
                    last_value = 0
                buffer.done(last_value)

                state = environment.reset()
                state = process_state(next_state)

                print(f'Final step {t}: trajectory_reward: {trajectory_reward}, trajectory_length: {trajectory_length}')
                trajectory_rewards.append(trajectory_reward)
                trajectory_reward = 0
                trajectory_length = 0
        
        update()
    print(f'Train Time: {(time.time() - start_time):2f} seconds')
    print(f'Train Score: {np.mean(trajectory_rewards[-100:])}')

if __name__ == '__main__':
    pome(environment=gymnasium.make('PongNoFrameskip-v4', obs_type='grayscale'),
        networkclass=Network,
        number_of_epoch=1,
        steps_per_epoch=2,
        pome_learning_rate=2.5*1e-4,
        reward_learning_rate=0.01,
        train_pome_iterations=2,
        train_reward_iterations=2,
        max_trajectory_length=2,
        discount_factor=0.99,
        alpha=0.1,
        clip_ratio=0.1,
        value_loss_ratio=1,
        transition_loss_ratio=2,
        seed=24)