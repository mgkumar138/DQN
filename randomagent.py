import gym

env = gym.make('BreakoutDeterministic-v4')  # BreakoutDeterministic-v4
episodes = 10
time = 500

for e in range(episodes):
    state = env.reset()
    R = 0

    for t in range(time):
        env.render()
        new_state, reward, done, _ = env.step(env.action_space.sample())
        R += reward
        print(state)
        if done:
            print('Episode {} of {}, score = {}'.format(e,episodes,R))
            break
env.close()
