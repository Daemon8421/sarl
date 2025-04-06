import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import random
from utils import plot_learning_curve
from tqdm.notebook import tqdm
# import ray
# import gymnasium as gym

# @ray.remote
# class RemoteEnv:
#     def __init__(self, env_name):
#         self.env = gym.make(env_name)
#         self.buffer = 
    
#     def run_episode(self, agent):
#         obs, _ = self.env.reset()
#         done = False
#         while not done:
#             action, prob, val = agent.choose_action(obs)
#             obs, reward, terminated, truncated, _ = self.env.step(action)
#             done = terminated or truncated
#         return episode_reward

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones, dtype=np.int8),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'ppo_actor.pth')
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
        th.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(th.load(checkpoint_file, weights_only=True))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'ppo_critic.pth')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        th.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(th.load(checkpoint_file, weights_only=True))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    
    def clear_memory(self):
        self.memory.clear_memory()

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self, model_path='./tmp/ppo'):
        print('... loading models ...')
        self.actor.load_checkpoint(model_path + '/ppo_actor.pth')
        self.critic.load_checkpoint(model_path + '/ppo_critic.pth')

    def choose_action(self, observation, deterministic=False):
        state = th.tensor(np.array([observation]), dtype=th.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        if not deterministic:
            action = dist.sample()
        else:
            action = th.argmax(dist.logits, dim=1, keepdim=True)

        probs = th.squeeze(dist.log_prob(action)).item()
        action = np.array(th.squeeze(action).cpu())
        value = th.squeeze(value).item()

        return action, probs, value
    
    def collect_rollout(self, env, seed=11, T=200):
        obs, _ = env.reset(seed=seed)
        done = False
        n_step = 0
        while not done:
            action, prob, val = self.choose_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.remember(obs, action, prob, val, reward, done)
            
            obs = next_obs
            n_step += 1
            
            # if n_step == T:
            #     break
        
        return n_step

    def learn(self, epochs=1):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1, 0, -1):
                if dones_arr[t]:
                    a_t = reward_arr[t] - values[t]
                else:
                    a_t = reward_arr[t] + self.gamma * values[t + 1] - values[t]
                    a_t += self.gamma * self.gae_lambda * advantage[t + 1]
                advantage[t] = a_t
                
            advantage = th.tensor(advantage).to(self.actor.device)

            values = th.tensor(values).to(self.actor.device)
            for _ in range(epochs):
                random.shuffle(batches)
                
                for batch in batches:
                    states = th.tensor(state_arr[batch], dtype=th.float).to(self.actor.device)
                    old_probs = th.tensor(old_prob_arr[batch]).to(self.actor.device)
                    actions = th.tensor(action_arr[batch]).to(self.actor.device)

                    dist = self.actor(states)
                    critic_value = self.critic(states)

                    critic_value = th.squeeze(critic_value)

                    new_probs = dist.log_prob(actions)
                    prob_ratio = new_probs.exp() / old_probs.exp()
                    #prob_ratio = (new_probs - old_probs).exp()
                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = th.clamp(prob_ratio, 1-self.policy_clip,
                            1+self.policy_clip)*advantage[batch]
                    actor_loss = -th.min(weighted_probs, weighted_clipped_probs).mean()

                    returns = advantage[batch] + values[batch]
                    critic_loss = (returns-critic_value)**2
                    critic_loss = critic_loss.mean()

                    total_loss = actor_loss + 0.5*critic_loss
                    self.actor.optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()
                    total_loss.backward()
                    self.actor.optimizer.step()
                    self.critic.optimizer.step()

        self.memory.clear_memory()

    def train(self, env, train_freq=20, batch_size=5, n_epochs=4, lr=3e-4, n_episodes=300, seed=11, N=5, T=200, K=1, eval_freq=1):
        best_score = env.spec.reward_threshold / 2 # threshold of cartpole is 475.0
        score_history = []
        avg_score = 0
        n_steps = 0

        # latest_msg = []
        for episode in tqdm(range(n_episodes)):
            observation, _ = env.reset(seed=seed)
            done = False
            score = 0

            # run N rollouts
            for _ in range(N):
                n_step = self.collect_rollout(env, seed, T)
                n_steps += n_step
            
            self.learn(K)
            
            if episode % eval_freq == 0:
                obs, _ = env.reset(seed=seed)
                score = 0
                n_step = 0
                done = False
                while not done:
                    with th.no_grad():
                        action, _, _ = self.choose_action(obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        score += reward
                        n_step += 1
                        
                # latest_msg.append(f'\033[92m [episode {episode}] exploit timesteps: {n_steps}\tscore: {score:.2f}\talive timesteps: {n_step}\033[0m')
                # if len(latest_msg) > 15:
                #     latest_msg.pop(0)
                # print('\033[F' * (len(latest_msg) + 1), end='')
                # print('\033[K', end='')
                # for msg in latest_msg:
                #     print(msg)
                print(f'\033[92m [episode {episode}] exploit timesteps: {n_steps}\tscore: {score:.2f}\talive timesteps: {n_step}\033[0m')
            
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                self.save_models()

        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history)