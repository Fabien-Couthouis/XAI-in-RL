import numpy as np
import random
from collections import namedtuple, deque

from model import Critic, Actor
import torch
from torch import autograd
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random_p as rm
from schedule import LinearSchedule

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
ACTOR_LR = 1e-3         # Actor network learning rate 
CRITIC_LR = 1e-4        # Actor network learning rate
UPDATE_EVERY = 20       # how often to update the network (time step)
#UPDATE_TIMES = 10       # how many times to update in one go

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents,seed,fc1=400,fc2=300,update_times=10,weight_decay=1.e-5):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.num_agents=num_agents
        self.n_seed=np.random.seed(seed)
        self.update_times=update_times
        self.n_step=0
        
        self.noise=[]
        for i in range(num_agents):
            self.noise.append(rm.OrnsteinUhlenbeckProcess(size=(action_size, ), std=LinearSchedule(0.2)))

        # critic local and target network (Q-Learning)
        self.critic_local = Critic(state_size, action_size,fc1,fc2, seed).to(device)
        
        self.critic_target = Critic(state_size, action_size,fc1,fc2, seed).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        
        # actor local and target network (Policy gradient)
        self.actor_local=Actor(state_size, action_size,fc1,fc2, seed).to(device)
        self.actor_target=Actor(state_size, action_size,fc1,fc2, seed).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        
        # optimizer for critic and actor network
        self.optimizer_critic = optim.Adam(self.critic_local.parameters(), lr=CRITIC_LR,weight_decay=1.e-5)
        self.optimizer_actor = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.a_step = 0

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        for i in range(self.num_agents):
            all_state=self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        if self.t_step == 0:
            
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE: 
                for i in range(self.update_times):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state,training=True):
        """Returns continous actions values for all action for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """
        
        state = torch.from_numpy(state).float().detach().to(device)
        #print(state.shape,"act")
        
        self.actor_local.eval()
        with torch.no_grad():
            actions=self.actor_local(state)
        self.actor_local.train()
        
        self.n_step+=1
        dec=max((500-self.n_step)/500,.05)
        noise=[]
        #for i in range(self.num_agents):
        #    noise.append(self.noise[i].sample())
        
        return np.clip(actions.cpu().data.numpy()+np.random.uniform(-1,1,(self.num_agents,self.action_size))*dec,-1,1)
        #np.clip(actions.cpu().data.numpy()+np.array(noise),-1,1)

    def learn(self, experiences, gamma):
        
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences

        next_actions=self.actor_target(next_states)
        with torch.no_grad():
            Q_target_next = self.critic_target(next_states,next_actions)
        Q_targets= rewards +(gamma * Q_target_next * (1-dones))
        
        Q_expected = self.critic_local(states,actions)
        
        #critic loss
        loss=F.mse_loss(Q_expected, Q_targets.detach())
        
        self.optimizer_critic.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.optimizer_critic.step()
        
        #actor loss
        
        
        
        action_pr = self.actor_local(states)
        p_loss=-self.critic_local(states,action_pr).mean()

        
        
        self.optimizer_actor.zero_grad()
        p_loss.backward()
        
        self.optimizer_actor.step()

        # ------------------- update target network ------------------- #

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def reset_random(self):
        for i in range(self.num_agents):
            self.noise[i].reset_states()
        

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        #actions.requires_grad=True
        #print(actions.requires_grad,"grad")
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)