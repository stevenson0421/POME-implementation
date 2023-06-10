import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import scipy
import math

import gymnasium

import time
import cv2
from tqdm import trange

'''
Functions
'''

'''
calclate discounted cumulated sum for a sequence (np array or tensor) and a discount factor
'''
def discounted_cumulated_sum(sequence, discount_factor, device=None):
    if isinstance(sequence, np.ndarray):
            return scipy.signal.lfilter([1], [1, float(-discount_factor)], sequence[::-1], axis=0)[::-1]
    elif isinstance(sequence, torch.Tensor):
            return torch.as_tensor(np.ascontiguousarray(scipy.signal.lfilter([1], [1, float(-discount_factor)], sequence.detach().cpu().numpy()[::-1], axis=0)[::-1]), dtype=torch.float32, device=device)
    else:
            raise TypeError

'''
process state from environment

current implementation
    1. resize state
    2. add a dimension for one channel since the state is grayscale image
    3. scale values to [0, 1]
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
shuffle all data in a dictionary
'''
def shuffle_data(data, seed):
    k, v = list(data.items())[0]
    data_length = len(v)
    data_shuffled = {}
    torch.manual_seed(seed)
    random_indices = torch.randperm(data_length)
    for k, v in data.items():
        data_shuffled[k] = v[random_indices]
    
    return data_shuffled


'''
Data Buffer
'''

class Buffer:
    '''
    Buffer for storing information of trajectories experienced by POME agent,
    including state, action, qf (TD target), reward, reward-to-go (discounted cumulated reward), value (estimation), action probability in log
    '''
    def __init__(self, size, state_dimension, action_dimension, discount_factor, device):
        self.state_buffer = np.zeros(tuple([size]+list(state_dimension)), dtype=np.float32)
        self.action_buffer = np.zeros(tuple([size]+list(action_dimension)), dtype=np.float32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.reward_to_go_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.action_logprobability_buffer = np.zeros(size, dtype=np.float32)

        self.discount_factor = discount_factor
        self.pointer = 0
        self.start_index = 0
        self.max_size = size

        self.device = device
    
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
        self.advantage_buffer[trajectory_slice] = reward_list[:-1] + self.discount_factor * value_list[1:] - value_list[:-1]

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
                    advantage=self.advantage_buffer,
                    reward=self.reward_buffer,
                    reward_to_go=self.reward_to_go_buffer,
                    value=self.value_buffer,
                    action_logprobability=self.action_logprobability_buffer)
        return {key: torch.as_tensor(value, dtype=torch.float32, device=self.device) for key, value in data.items()}

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
        torch.nn.init.orthogonal_(self.conv1.weight, gain=math.sqrt(2))
        torch.nn.init.zeros_(self.conv1.bias)
        conv1_dimension = (math.floor((state_dimension[1]-8)/4)+1, math.floor((state_dimension[2]-8)/4)+1)

        self.conv2 = torch.nn.Conv2d(in_channels=16,
                                      out_channels=32,
                                      kernel_size=4,
                                      stride=2,
                                      padding='valid')
        torch.nn.init.orthogonal_(self.conv2.weight, gain=math.sqrt(2))
        torch.nn.init.zeros_(self.conv2.bias)
        conv2_dimension = (math.floor((conv1_dimension[0]-4)/2)+1, math.floor((conv1_dimension[1]-4)/2)+1)

        self.fc1 = torch.nn.Linear(in_features=conv2_dimension[0]*conv2_dimension[1]*32,
                                      out_features=256)
        torch.nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
        torch.nn.init.zeros_(self.fc1.bias)

        self.fc2 = torch.nn.Linear(in_features=256,
                                       out_features=action_dimension)
        torch.nn.init.orthogonal_(self.fc2.weight, gain=0.01)
        torch.nn.init.zeros_(self.fc2.bias)

        self.activation = torch.nn.ReLU()

        self.flatten = torch.nn.Flatten()

        self.policy_distribution = None

    def forward(self, state):
        latent = self.activation(self.conv1(state))
        latent = self.activation(self.conv2(latent))
        latent = self.flatten(latent)
        latent = self.activation(self.fc1(latent))
        policy = self.activation(self.fc2(latent))

        self.policy_distribution = torch.distributions.Categorical(logits=policy)

        return latent, policy


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
        torch.nn.init.orthogonal_(self.fc1.weight, gain=1)
        torch.nn.init.zeros_(self.fc1.bias)

    def forward(self, latent):
        value = self.fc1(latent)

        return value

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
            latent, policy = self.policy_network(state)
            
            action = self.policy_network.policy_distribution.sample()
            action_logprobability = self.policy_network.policy_distribution.log_prob(action)
            value = self.value_network(latent)
        return action.cpu().numpy().item(), action_logprobability.cpu().numpy().item(), value.cpu().numpy().item()
    
    '''
    get action only from a state

    this function is not used yet, maybe for evaluation
    '''
    def act(self, state):
        with torch.no_grad():
            latent, policy = self.policy_network(state)
            action = self.policy_network.policy_distribution.sample()
        return action.cpu().numpy()


'''
ppo

main implementation of the method

input parameters

    environment: gym environment ex. gym.make('PongNoFrameskip-v4')

    state_new_size: states of environment are resized to this new size

    networkclass: combined network

    number_of_epoch: the number of epoch in training process

    steps_per_epoch: steps of a epoch

    steps_per_subepoch: substeps of a epoch. note that this is for saving memory with long steps

    steps_per_trajectory: max steps of a trajectory, basically the time limit of agent (truncated if set)

    batch_size: minibatch size in learning process (k in paper)

    policy_learning_rate: learning rate of ppo object, consist of policy, value

    train_iterations: how many iterations to update ppo parameters in one update

    discount_factor: discount factor (gamma) of computing discounted cumulated sum

    clip_ratio: clip ratio for PPO object before combined (l^PPO in paper)

    value_loss_ratio: coefficient of value loss for combined PPO object

    seed: seed for randomization

    device: device for all torch tensors

'''
def ppo(environment,
        state_new_size,
        networkclass,
        number_of_epoch,
        steps_per_epoch,
        steps_per_subepoch,
        steps_per_trajectory,
        batch_size,
        policy_learning_rate,
        train_iterations,
        discount_factor,
        value_loss_ratio,
        clip_ratio,
        seed,
        device
        ):
    
    assert isinstance(environment.action_space, gymnasium.spaces.Discrete)

    set_seed(seed)
    
    # current implementation (210, 160)
    # state_space = environment.observation_space
    state_dimension = (1, state_new_size[0], state_new_size[1])
    # current implementation Discrete(6)
    action_dimension = environment.action_space.n
    
    network = networkclass(state_dimension, action_dimension).to(device)

    # state dimension is (1, 210, 160), action dimension is 0 (scalar)
    buffer = Buffer(steps_per_subepoch, state_dimension, environment.action_space.shape, discount_factor, device)

    policy_optimizer = torch.optim.Adam([{'params':network.policy_network.parameters()},
                                         {'params':network.value_network.parameters()},], lr=policy_learning_rate)
    # learning rate is linearly annealed [1, 0]
    torch.optim.lr_scheduler.LinearLR(optimizer=policy_optimizer, start_factor=1., end_factor=0., total_iters=steps_per_epoch*number_of_epoch)
    
    '''
    update all network parameters
    '''
    def update(current_step):
        '''
        current_step: the timestep of current update, recorded by tensorboard
        '''
        # get all data from buffer, which should be a trajectory
        data = buffer.get()

        # update ppo parameters
        for i in trange(train_iterations):
            data_shuffled = shuffle_data(data, seed+i)

            for batch_index in range(int(math.floor(steps_per_subepoch / batch_size))):
                data_minibatch = {k:v[batch_index*batch_size:(batch_index+1)*batch_size].to(device) for k, v in data_shuffled.items()}
                state = data_minibatch['state']
                action = data_minibatch['action']
                advantage = data_minibatch['advantage']
                # pi_old, note that this is fixed during one epoch update
                action_logprobability_old = data_minibatch['action_logprobability']
                reward_to_go = data_minibatch['reward_to_go']

                policy_optimizer.zero_grad()
                
                '''
                note that the KL divergence in paper is not included:
                    1. I think this is duplicated with the clip trick
                    2. the value of coefficient Beta is not mentioned
                '''
                
                # get policy, action probability in log and value
                latent, policy = network.policy_network(state)
                action_logprobability = network.policy_network.policy_distribution.log_prob(action)
                value = network.value_network(latent)
                value = value.squeeze()

                value_loss = ((value - reward_to_go)**2).mean()
                
                # calculate pi / pi_old
                policy_ratio = torch.exp(action_logprobability-action_logprobability_old)
                # calculate policy loss in paper
                clip_advantage = torch.clamp(policy_ratio, 1-clip_ratio, 1+clip_ratio) * advantage
                policy_loss = -(torch.min(policy_ratio * advantage, clip_advantage)).mean()

                total_loss = policy_loss + value_loss_ratio * value_loss
                total_loss.backward()
                policy_optimizer.step()


        print(f'policy loss: {policy_loss}, value loss: {value_loss}')
        # tensorboard recording loss
        writer.add_scalar("Policy loss", policy_loss, global_step=(current_step))
        writer.add_scalar("Value loss", value_loss, global_step=(current_step))

        # reset random seed
        torch.manual_seed(seed)

    # main process
    trajectory_reward = 0
    trajectory_length = 0
    trajectory_rewards = []

    start_time = time.time()
    state, info = environment.reset()
    state = process_state(state, state_new_size)

    total_steps = 0
    
    for epoch in range(number_of_epoch):
        number_of_subepoch = int(math.floor(steps_per_epoch / steps_per_subepoch))

        for subepoch in range(number_of_subepoch):
            
            print('interacting phase')
            network.to('cpu')

            current_step = (epoch*number_of_subepoch + subepoch)
            print('current_step: ', current_step)
            for step in range(steps_per_subepoch):
                # get action, action probability in log, value from a state. note that state is unsqueezed as a batch with size 1
                action, action_logprobability, value = network.step(torch.as_tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0))

                # agent interaction
                next_state, reward, terminated, truncated, info = environment.step(action)
                trajectory_reward += reward
                trajectory_length += 1
                # store all data to buffer
                buffer.store(state, action, reward, value, action_logprobability)
                # update state. this is important!
                state = process_state(next_state, state_new_size)
                # set timeout for agent
                timeout = (trajectory_length == steps_per_trajectory)
                done = (terminated or truncated or timeout)
                # epoch is ended with full steps
                epoch_ended = ((step+subepoch*steps_per_subepoch) == steps_per_epoch-1)
                # end trajectory. note that timeout and epoch_ended are same if we set the same boundary
                if done or epoch_ended:
                    if epoch_ended and not done:
                        print('Warning: trajectory cut off by epoch at %d steps.'%trajectory_length)

                    if timeout or epoch_ended:
                        _, last_value, _ = network.step(torch.as_tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0))
                    else:
                        last_value = 0
                    buffer.done(last_value)
                    # reset the environment. this is important!
                    state = environment.reset()
                    state = process_state(next_state, state_new_size)
                    # record total reward and reset
                    print(f'trajectory ends with step {step}: trajectory_reward: {trajectory_reward}, trajectory_length: {trajectory_length}')
                    trajectory_rewards.append(trajectory_reward)
                    
                    trajectory_reward = 0
                    trajectory_length = 0

                    if epoch_ended:
                        total_steps += 1
                        break
                    
                    total_steps += 1
            # record with tensorboard
            
            writer.add_scalar("Trajectory reward", np.mean(trajectory_rewards[-100:]), global_step=current_step)
                        
            print('training phase')
            network.to(device)

            update(current_step)
    print(f'Train Time: {(time.time() - start_time):2f} seconds')
    # average reward of last 100 trajectories in paper
    print(f'Train Score: {np.mean(trajectory_rewards[-100:])}')

    # testing
    state, info = environment.reset()
    state = process_state(state, state_new_size)
    screens = []

    network.to('cpu')

    while True:
        action, action_logprobability, value = network.step(torch.as_tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0))
        next_state, reward, terminated, truncated, info = environment.step(action)
        screens.append(environment.render())
        state = process_state(next_state, state_new_size)

        if (terminated or truncated):
            out = cv2.VideoWriter(f'./experiments/ppo/video/{environment.unwrapped.spec.id}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, screens[0].shape[0:2])
            for img in screens:
                out.write(img)
            out.release()        

            torch.save(network.state_dict(), f'./experiments/ppo/model/{environment.unwrapped.spec.id}.pth')



if __name__ == '__main__':
    for env_name in ['ALE/RoadRunner-v5', 'ALE/Kangaroo-v5', 'ALE/Alien-v5']:
        # Initialize a SummaryWriter for TensorBoard
        writer = SummaryWriter(log_dir=f'./experiments/ppo/runs/{time.strftime("%Y%m%d-%H%M%S")}')

        print(env_name)

        ppo(environment=gymnasium.make(env_name, obs_type='grayscale', render_mode='rgb_array'),
            state_new_size=(84, 84),
            networkclass=Network,
            number_of_epoch=10,
            steps_per_epoch=100000,
            steps_per_subepoch=1000,
            steps_per_trajectory=100000,
            batch_size=5,
            policy_learning_rate=2.5*1e-4,
            train_iterations=1,
            discount_factor=0.99,
            clip_ratio=0.1,
            value_loss_ratio=0.5,
            seed=24,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
        writer.close()