import numpy as np
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.normal import Normal



class GumbelSoftmax(OneHotCategorical):

    def __init__(self, logits, probs=None, temperature=0.1):
        super(GumbelSoftmax, self).__init__(logits=logits, probs=probs)
        self.eps = 1e-20
        self.temperature = temperature

    def sample_gumbel(self):
        U = self.logits.clone()
        U.uniform_(0, 1)
        return -torch.log( -torch.log( U + self.eps ) )

    def gumbel_softmax_sample(self):
        y = self.logits + self.sample_gumbel()
        return torch.softmax( y / self.temperature, dim=-1)

    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (torch.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()

    def hard_sample(self):
        return self.hard_gumbel_softmax_sample()



def normal_entropy(mean, std):
    return Normal(mean, std).entropy().sum()

def multinomial_entropy(logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).entropy().sum()

def normal_log_density(x, mean, std):
    return Normal(mean, std).log_prob(x)

def multinomials_log_density(actions, logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).log_prob(actions)

def select_action(args, logits, status='train', exploration=True, info={}):
    if args.continuous:
        act_mean = logits
        act_std = cuda_wrapper(torch.ones_like(act_mean), args.cuda)
        if status is 'train':
            return Normal(act_mean, act_std).sample()
        elif status is 'test':
            return act_mean
    else:
        if status is 'train':
            if exploration:
                if args.epsilon_softmax:
                    eps = info['softmax_eps']
                    p_a = (1 - eps) * torch.softmax(logits, dim=-1) + eps / logits.size(-1)
                    return OneHotCategorical(logits=None, probs=p_a).sample()
                elif args.gumbel_softmax:
                    return GumbelSoftmax(logits=logits).sample()
                else:
                    return OneHotCategorical(logits=logits).sample()
            else:
                if args.gumbel_softmax:
                    temperature = 1.0
                    return torch.softmax(logits/temperature, dim=-1)
                else:
                    return OneHotCategorical(logits=logits).sample()
        elif status is 'test':
            p_a = torch.softmax(logits, dim=-1)
            return  (p_a == torch.max(p_a, dim=-1, keepdim=True)[0]).float()

def translate_action(args, action, env):
    if not args.continuous:
        actual = [act.detach().squeeze().cpu().numpy() for act in torch.unbind(action, 1)]
        return action, actual
    else:
        actions = action.data[0].numpy()
        cp_actions = actions.copy()
        # clip and scale action to correct range
        for i in range(len(cp_actions)):
            cp_actions[i] = env.action_space[i].low
            cp_actions[i] = env.action_space[i].high
            cp_actions[i] = max(-1.0, min(cp_actions[i], 1.0))
            cp_actions[i] = 0.5 * (cp_actions[i] + 1.0) * (high - low) + low
        return actions, cp_actions

def prep_obs(state=[]):
    state = np.array(state)
    if len(state.shape) == 2:
        state = np.stack(state, axis=0)
    elif len(state.shape) == 4:
        state = np.concatenate(state, axis=0)
    else:
        raise RuntimeError('The shape of the observation is incorrect.')
    return torch.tensor(state).float()

def cuda_wrapper(tensor, cuda):
    if isinstance(tensor, torch.Tensor):
        return tensor.cuda() if cuda else tensor
    else:
        raise RuntimeError('Please enter a pytorch tensor, now a {} is received.'.format(type(tensor)))

def batchnorm(batch):
    if isinstance(batch, torch.Tensor):
        assert batch.size(-1) == 1
        return (batch - batch.mean()) / batch.std()
    else:
        raise RuntimeError('Please enter a pytorch tensor, now a {} is received.'.format(type(batch)))

def get_grad_norm(params):
    grad_norms = []
    for param in params:
        grad_norms.append(torch.norm(param.grad).item())
    return np.mean(grad_norms)

def merge_dict(stat, key, value):
    if key in stat.keys():
        stat[key] += value
    else:
        stat[key] = value

def unpack_data(args, batch):
    batch_size = len(batch.state)
    n = args.agent_num
    action_dim = args.action_dim
    cuda = torch.cuda.is_available() and args.cuda
    rewards = cuda_wrapper(torch.tensor(batch.reward, dtype=torch.float), cuda)
    last_step = cuda_wrapper(torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1), cuda)
    done = cuda_wrapper(torch.tensor(batch.done, dtype=torch.float).contiguous().view(-1, 1), cuda)
    actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.action))[0], axis=0), dtype=torch.float), cuda)
    last_actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.last_action))[0], axis=0), dtype=torch.float), cuda)
    state = cuda_wrapper(prep_obs(list(zip(batch.state))), cuda)
    next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), cuda)
    return (rewards, last_step, done, actions, last_actions, state, next_state)

def n_step(rewards, last_step, done, next_values, n_step, args):
    cuda = torch.cuda.is_available() and args.cuda
    returns = cuda_wrapper(torch.zeros_like(rewards), cuda=cuda)
    i = rewards.size(0)-1
    while i >= 0:
        if last_step[i]:
            next_return = 0 if done[i] else next_values[i].detach()
            for j in reversed(range(i-n_step+1, i+1)):
                returns[j] = rewards[j] + args.gamma * next_return
                next_return = returns[j]
            i -= n_step
            continue
        else:
            next_return = next_values[i+n_step-1].detach()
        for j in reversed(range(n_step)):
            g = rewards[i+j] + args.gamma * next_return
            next_return = g
        returns[i] = g.detach()
        i -= 1
    return returns
