from learning_algorithms.rl_algorithms import *
import torch
from utilities.util import *



class ActorCritic(ReinforcementLearning):

    def __init__(self, args):
        super(ActorCritic, self).__init__('Actor_Critic', args)

    def __call__(self, batch, behaviour_net):
        return self.get_loss(batch, behaviour_net)

    def get_loss(self, batch, behaviour_net, target_net=None):
        # TODO: fix policy params update
        batch_size = len(batch.state)
        n = self.args.agent_num
        # collect the transition data
        rewards, last_step, done, actions, state, next_state = behaviour_net.unpack_data(batch)
        # construct the computational graph
        action_out = behaviour_net.policy(state)
        values = behaviour_net.value(state, actions)
        if self.args.q_func:
            values = torch.sum(values*actions, dim=-1)
        values = values.contiguous().view(-1, n)
        if target_net == None:
            next_action_out = behaviour_net.policy(next_state)
        else:
            next_action_out = target_net.policy(next_state)
        next_actions = select_action(self.args, next_action_out, status='train')
        next_values = behaviour_net.value(next_state, next_actions)
        if self.args.q_func:
            next_values = torch.sum(next_values*next_actions, dim=-1)
        next_values = next_values.contiguous().view(-1, n)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        # calculate the advantages
        assert values.size() == next_values.size()
        assert returns.size() == values.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_values[i].detach()
            else:
                next_return = next_values[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
        deltas = returns - values
        advantages = values.detach()
        # advantages = advantages.contiguous().view(-1, 1)
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        # construct the action loss and the value loss
        log_prob_a = multinomials_log_density(actions, action_out).contiguous().view(-1,n)
        assert log_prob_a.size() == advantages.size()
        action_loss = -advantages * log_prob_a
        action_loss = action_loss.mean(dim=0)
        value_loss = deltas.pow(2).mean(dim=0)
        return action_loss, value_loss, action_out
