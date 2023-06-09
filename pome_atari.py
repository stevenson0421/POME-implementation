import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import scipy
import math

import gymnasium

import time
import cv2
from tqdm import trange
import os


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
        self.qf_buffer = np.zeros(size, dtype=np.float32)
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
        torch.nn.init.orthogonal_(self.conv1.weight, gain=math.sqrt(2))
        torch.nn.init.zeros_(self.conv1.bias)
        conv1_dimension = (math.floor((state_action_dimension[1]-8)/4)+1, math.floor((state_action_dimension[2]-8)/4)+1)

        self.fc1 = torch.nn.Linear(in_features=conv1_dimension[0]*conv1_dimension[1]*16,
                                      out_features=1)
        torch.nn.init.orthogonal_(self.fc1.weight, gain=1)
        torch.nn.init.zeros_(self.fc1.bias)
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
        torch.nn.init.orthogonal_(self.conv1.weight, gain=math.sqrt(2))
        torch.nn.init.zeros_(self.conv1.bias)
        conv1_dimension = (math.floor((state_action_dimension[1]-8)/4)+1, math.floor((state_action_dimension[2]-8)/4)+1)
        
        self.fc1 = torch.nn.Linear(in_features=conv1_dimension[0]*conv1_dimension[1]*16,
                                      out_features=state_dimension[1]*state_dimension[2])
        torch.nn.init.orthogonal_(self.fc1.weight, gain=0.01)
        torch.nn.init.zeros_(self.fc1.bias)

        self.flatten = torch.nn.Flatten()
        self.activation = torch.nn.Sigmoid()
    
    def forward(self, state, action):
        # concatenate state and action
        action_one_hot = torch.nn.functional.one_hot(action.to(torch.int64), num_classes=state.shape[3])
        action_one_hot = action_one_hot.unsqueeze(1).unsqueeze(1)
        state_action = torch.cat((state, action_one_hot), dim=2)

        transition = self.conv1(state_action)
        transition = self.flatten(transition)
        transition = self.activation(self.fc1(transition))

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
        # self.reward_network = RewardNetwork(state_dimension)
        self.transition_network = TransitionNetwork(state_dimension)

    '''
    get the log probability of an action
    '''
    def get_action_logprobability(self, policy_distribution, action):
        action_logprobability = policy_distribution.log_prob(action)
        
        return action_logprobability
    
    '''
    get action, log probability of the action and value estimation from a state
    return as tensor for training
    '''
    def train_step(self, state):
        latent, policy = self.policy_network(state)
            
        action = self.policy_network.policy_distribution.sample()
        action_logprobability = self.policy_network.policy_distribution.log_prob(action)
        value = self.value_network(latent)

        return action, action_logprobability, value.squeeze()

    '''
    get action, log probability of the action and value estimation from a state
    return as np array for environment interacting
    '''
    def eval_step(self, state):
        with torch.no_grad():
            latent, policy = self.policy_network(state)
            
            action = self.policy_network.policy_distribution.sample()
            action_logprobability = self.policy_network.policy_distribution.log_prob(action)
            value = self.value_network(latent)
        return action.cpu().numpy().item(), action_logprobability.cpu().numpy().item(), value.cpu().numpy().item()
    
    '''
    get action only from a state

    this function is used for evaluation
    '''
    def act(self, state):
        with torch.no_grad():
            latent, policy = self.policy_network(state)
            action = self.policy_network.policy_distribution.sample()
        return action.cpu().numpy().item()


'''
pome

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

    device: device for all torch tensors

'''
def pome(environment,
         state_new_size,
         networkclass,
         number_of_epoch,
         steps_per_epoch,
         steps_per_subepoch,
         steps_per_trajectory,
         batch_size,
         pome_learning_rate,
         reward_learning_rate,
         train_pome_iterations,
         train_reward_iterations,
         discount_factor,
         alpha,
         clip_ratio,
         value_loss_ratio,
         transition_loss_ratio,
         seed,
         device
         ):
    
    assert isinstance(environment.action_space, gymnasium.spaces.Discrete)

    number_of_subepoch = int(math.floor(steps_per_epoch / steps_per_subepoch))

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    env_name = environment.unwrapped.spec.id.replace('/', '_')
    
    # current implementation (1, 84, 84)
    # state_space = environment.observation_space
    state_dimension = (1, state_new_size[0], state_new_size[1])
    # current implementation Discrete(6)
    action_dimension = environment.action_space.n
    
    network = networkclass(state_dimension, action_dimension).to(device)

    # state dimension is (1, 210, 160), action dimension is 0 (scalar)
    buffer = Buffer(steps_per_subepoch, state_dimension, environment.action_space.shape, discount_factor, device)

    pome_optimizer = torch.optim.Adam([{'params':network.policy_network.parameters()},
                                       {'params':network.value_network.parameters()},
                                       {'params':network.transition_network.parameters()},], lr=pome_learning_rate)
    # learning rate is linearly annealed [1, 0]
    torch.optim.lr_scheduler.LinearLR(optimizer=pome_optimizer, start_factor=1., end_factor=0., total_iters=steps_per_epoch)

    # reward_optimizer = torch.optim.Adam(params=network.reward_network.parameters(), lr=reward_learning_rate)
    
    '''
    update all network parameters in paper. first update pome, then reward
    '''
    def update(current_step):
        '''
        current_step: the timestep of current update, recorded by tensorboard
        '''
        # get all data from buffer, which should be a trajectory
        data = buffer.get()

        # calculate Q_b_t in paper
        state = data['state']
        action = data['action']
        qf = data['qf']
        value = data['value']

        # reward_hat = network.reward_network(state, action)
        reward_hat = data['reward']
        transition = network.transition_network(state, action)
        transition_reshaped = transition.reshape(state.shape)
        _, _, value_t = network.train_step(transition_reshaped)
        qb = reward_hat + discount_factor * value_t
        data['qb'] = qb

        # calculate epsilon and epsilon_bar in paper
        alpha_a = alpha * (1 - current_step / number_of_subepoch)
        epsilon = torch.abs(qf - qb)
        epsilon_median = torch.median(epsilon)
        delta_t = torch.abs(qf - value)
        delta_t_pome = qf + alpha_a * torch.clamp(epsilon-epsilon_median, -delta_t, delta_t) - value
        # calculate a_t_pome in paper
        data['a_t_pome'] = discounted_cumulated_sum(delta_t_pome, 1, device)

        # update pome parameters
        for i in trange(train_pome_iterations):
            data_shuffled = shuffle_data(data, seed+i)

            for batch_index in range(int(math.floor(steps_per_subepoch / batch_size))):
                # note that these are fixed during one epoch update
                data_minibatch = {k:v[batch_index*batch_size:(batch_index+1)*batch_size].to(device) for k, v in data_shuffled.items()}
                state = data_minibatch['state']
                action = data_minibatch['action']
                reward_to_go = data_minibatch['reward_to_go']
                action_logprobability_old = data_minibatch['action_logprobability']
                a_t_pome = data_minibatch['a_t_pome']

                pome_optimizer.zero_grad()
                
                '''
                note that the KL divergence in paper is not included:
                    1. I think this is duplicated with the clip trick
                    2. the value of coefficient Beta is not mentioned
                '''
                
                # get policy, action probability in log and value
                action, action_logprobability, value = network.train_step(state)
                # calculate pome loss in paper
                # calculate pi / pi_old
                policy_ratio = torch.exp(action_logprobability-action_logprobability_old)

                a_t_pome_st = (a_t_pome - a_t_pome.mean()) / (a_t_pome.std()+1e-8)
                clip_a_t_pome = torch.clamp(policy_ratio, 1-clip_ratio, 1+clip_ratio) * a_t_pome_st
                pome_loss = -(torch.min(policy_ratio * a_t_pome_st, clip_a_t_pome)).mean()

                # calculate value loss in paper.
                value_object_core = a_t_pome + reward_to_go - value
                value_object_core = (value_object_core - value_object_core.mean()) / (value_object_core.std()+1e-8)
                value_loss = ((value_object_core) ** 2).mean()

                # calculate transition loss in paper. note that state of 0:-1 are used for calculation, not sure if this is correct
                transition = network.transition_network(state[:-1], action[:-1])
                transition_reshaped = transition.reshape((state.shape[0]-1, state.shape[1], state.shape[2], state.shape[3]))
                transition_loss = (torch.norm(state[:-1]-transition_reshaped, p=2) ** 2) / torch.numel(transition_reshaped)

                total_pome_loss = pome_loss + value_loss_ratio * value_loss + transition_loss_ratio * transition_loss
                total_pome_loss.backward()
                pome_optimizer.step()
        
        # update reward parameters
        # for i in trange(train_reward_iterations):
        #     data_shuffled = shuffle_data(data, seed+i)

        #     for batch_index in range(int(math.floor(steps_per_subepoch / batch_size))):
        #         data_minibatch = {k:v[batch_index*batch_size:(batch_index+1)*batch_size].to(device) for k, v in data_shuffled.items()}
        #         state = data_minibatch['state']
        #         action = data_minibatch['action']
        #         reward = data_minibatch['reward']

        #         reward_optimizer.zero_grad()
                
        #         # calculate reward loss in paper
        #         reward_hat = network.reward_network(state, action)
        #         reward_object_core = reward - reward_hat                
        #         reward_loss = (torch.mean(reward_object_core) ** 2)

        #         reward_loss.backward()
        #         reward_optimizer.step()


        print(f'pome loss: {pome_loss}, value loss: {value_loss}, transition loss: {transition_loss}')
        # tensorboard recording loss
        # writer.add_scalars(f'Loss summary', {
        #     'Pome loss': pome_loss,
        #     'Value loss': value_loss,
        #     'Transition loss': transition_loss,
        #     'Reward loss': reward_loss
        # }, current_step)
        writer.add_scalar("Pome loss", pome_loss, global_step=(current_step))
        writer.add_scalar("Value loss", value_loss, global_step=(current_step))
        writer.add_scalar("Transition loss", transition_loss, global_step=(current_step))
        # writer.add_scalar("Reward loss", reward_loss, global_step=(current_step))

        # reset random seed
        torch.manual_seed(seed)

    # main process
    trajectory_reward = 0
    trajectory_length = 0
    trajectory_rewards = []

    start_time = time.time()
    state, info = environment.reset(seed=seed)
    state = process_state(state, state_new_size)

    total_steps = 0
    
    for epoch in range(number_of_epoch):
        for subepoch in range(number_of_subepoch):
            
            print('interacting phase')
            network.to('cpu')

            current_step = (epoch*number_of_subepoch + subepoch)
            print('current_step: ', current_step)
            for step in range(steps_per_subepoch):
                # get action, action probability in log, value from a state. note that state is unsqueezed as a batch with size 1
                action, action_logprobability, value = network.eval_step(torch.as_tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0))

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
                        _, last_value, _ = network.eval_step(torch.as_tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0))
                    else:
                        last_value = 0
                    buffer.done(last_value)
                    # reset the environment. this is important!
                    state = environment.reset(seed=seed+current_step)
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

    if not os.path.exists('./experiments/pome/model/'):
        os.makedirs('./experiments/pome/model/')
    torch.save(network.state_dict(), f'./experiments/pome/model/{env_name}{postfix}.pth')

    # record video
    state, info = environment.reset(seed=seed)
    state = process_state(state, state_new_size)
    screens = []

    network.to('cpu')

    screens = []
    while True:
        action = network.act(torch.as_tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0))
        next_state, reward, terminated, truncated, info = environment.step(action)
        frame = environment.render()
        screens.append(frame)
        state = process_state(next_state, state_new_size)

        if (terminated or truncated):
            if not os.path.exists('./experiments/pome/video/'):
                os.makedirs('./experiments/pome/video/')
            out = cv2.VideoWriter(f'./experiments/pome/video/{env_name}{postfix}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, (screens[0].shape[1], screens[0].shape[0]))
            for img in screens:
                out.write(img)
            out.release()

            break



if __name__ == '__main__':

    env_name = 'ALE/RoadRunner-v5'

    postfix = '_annealalpha_long'

    if not os.path.exists('./experiments/pome/runs/'):
        os.makedirs('./experiments/pome/runs/')
    # Initialize a SummaryWriter for TensorBoard
    writer = SummaryWriter(log_dir=f'./experiments/pome/runs/{time.strftime("%Y%m%d-%H%M%S")}')

    print(env_name)

    pome(environment=gymnasium.make(env_name, obs_type='grayscale', render_mode='rgb_array'),
        state_new_size=(84, 84),
        networkclass=Network,
        number_of_epoch=10,
        steps_per_epoch=1000000,
        steps_per_subepoch=10000,
        steps_per_trajectory=100000,
        batch_size=100,
        pome_learning_rate=2.5*1e-4,
        reward_learning_rate=0.01,
        train_pome_iterations=1,
        train_reward_iterations=1,
        discount_factor=0.99,
        alpha=0.1,
        clip_ratio=0.1,
        value_loss_ratio=1,
        transition_loss_ratio=2,
        seed=24,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    writer.close()