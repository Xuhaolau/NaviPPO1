import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Normal

from net import PolicyNet, ValueNet
from helper import log_normal_density


class ppoAgent:
    def __init__(self, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, loss_tensorboard):
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.gamma = gamma
        self.device = device
        self.tsbd = loss_tensorboard
        self.action_bound = [[0, -1], [1, 1]]
        # initialize actor and critic
        self.actor = PolicyNet().to(device)
        self.critic = ValueNet().to(device)
        # initialize optimizer
        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, scan, goal, speed):
        scan = torch.tensor([scan], dtype=torch.float, requires_grad=True).to(self.device)
        goal = torch.tensor([goal], dtype=torch.float, requires_grad=True).to(self.device)
        speed = torch.tensor([speed], dtype=torch.float, requires_grad=True).to(self.device)
        mean, std = self.actor(scan, goal, speed)
        action_dists = Normal(mean, std)
        action = action_dists.sample()
        action = action.data.cpu().numpy()
        scaled_action = np.clip(action, a_min=self.action_bound[0], a_max=self.action_bound[1])
        return action, scaled_action

    def compute_advantage(self, gamma, lmbda, td_delta, dones):
        td_delta = td_delta.detach().numpy()
        dones = dones.detach().numpy()
        advantage_list = []
        advantage = 0.0
        # for delta in td_delta[::-1]:
        #     advantage = gamma * lmbda * advantage + delta
        #     advantage_list.append(advantage)
        for t in range(256 - 1, -1, -1):
            advantage = gamma * lmbda * advantage * (1 - dones[t]) + td_delta[t]
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def update(self, transition_dict, count=0):
        loss_a = []
        loss_c = []

        scans = torch.tensor(transition_dict['scans'], dtype=torch.float).to(self.device)
        goals = torch.tensor(transition_dict['goals'], dtype=torch.float).to(self.device)
        speeds = torch.tensor(transition_dict['speeds'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)
        next_scans = torch.tensor(transition_dict['next_scans'], dtype=torch.float).to(self.device)
        next_goals = torch.tensor(transition_dict['next_goals'], dtype=torch.float).to(self.device)
        next_speeds = torch.tensor(transition_dict['next_speeds'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_scans, next_goals, next_speeds) * (1 - dones)
        td_delta = td_target - self.critic(scans, goals, speeds)
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).view(-1, 1)\
            .to(self.device)



        mean, std = self.actor(scans, goals, speeds)
        action_dists = Normal(mean.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)
        old_log_probs = torch.sum(old_log_probs, dim=-1)

        for _ in range(self.epochs):
            mean, std = self.actor(scans, goals, speeds)
            action_dists = Normal(mean, std)
            log_probs = action_dists.log_prob(actions)
            log_probs = torch.sum(log_probs, dim=-1)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage

            actor_loss = torch.mean(-torch.min(surr1, surr2))
            loss_a.append(actor_loss.item())

            critic_loss = torch.mean(F.mse_loss(self.critic(scans, goals, speeds), td_target.detach()))
            loss_c.append(critic_loss.item())

            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()

        average_a = np.mean(loss_a)
        average_c = np.mean(loss_c)
        self.tsbd.add_scalar('actor/loss', average_a, count)
        self.tsbd.add_scalar('critic/loss', average_c, count)
