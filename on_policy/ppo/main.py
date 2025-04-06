from ppo import Agent
import gymnasium as gym

if __name__ == '__main__':
    # training
    env = gym.make('CartPole-v1', render_mode=None)
    agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
    agent.train(env=env, n_episodes=23)
    
    
    # test
    # agent.load_models()
    
    # obs, _ = env.reset(seed=11)
    # done = False
    # score = 0
    # n_steps = 0
    # while not done:
    #     action, _, _ = agent.choose_action(obs)
    #     obs, reward, terminated, truncated, _ = env.step(action)
    #     score += reward
    #     n_steps += 1
    #     done = terminated or truncated
    
    # print(f'alive timesteps: {n_steps}, score: {score}')