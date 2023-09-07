import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

env_name = 'LunarLander-v2'

env = gym.make(env_name,
               render_mode="rgb_array",
               continuous=True)

print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space)

vid = VideoRecorder(env, path=f"random_luna_lander.mp4")
observation = env.reset()[0]


def Random_games():
    # Each of this episode is its own game.
    env.reset()
    done = False
    while not done:

        env.render()
        vid.capture_frame()

        action = env.action_space.sample()  # np.array([main, lateral])

        observation, reward, done, info, _ = env.step(action)

        print(observation)  # ‘x’: 10 ‘y’: 6.666 ‘vx’: 5
        # ‘vy’: 7.5 ‘angle’: 1 ‘angular velocity’: 2.5

        print(reward, done, info, action)

    vid.close()
    env.close()


Random_games()