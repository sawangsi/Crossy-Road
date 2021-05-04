import gym
import gym_chrome_crossy_road
import warnings
warnings.filterwarnings("ignore")
from stable_baselines import DQN
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.vec_env import DummyVecEnv
import sys


environment_name = 'ChromeCrossyRoad-v0'
programing_type = int(sys.argv[1])

# Start to train the agent
if programing_type == 0:
	env = gym.make(environment_name)
	model = DQN("MlpPolicy", env, learning_rate=0.0001, gamma=0.7, batch_size=1024, prioritized_replay=False, verbose=1, tensorboard_log="./log/dqn_crossy_road_tensorboard_without_prioritized/")
	model.learn(total_timesteps=30000)
	model.save("../model/DQN_without_prioritized")
	env.close()

# Continue to train
elif programing_type == 1:
	myenv = gym.make(environment_name)
	env = DummyVecEnv([lambda: myenv])
	model = DQN.load('../model/DQN_without_prioritized', env=env)
	model.set_env(env)
	model.learn(total_timesteps=20000,callback=None, reset_num_timesteps=False)
	model.save("../model/DQN_without_prioritized")
	env.close()

# Test the agent
else:
	myenv = gym.make(environment_name)
	env = DummyVecEnv([lambda: myenv])
	model = DQN.load('../model/DQN_without_prioritized', env=env)
	result = {}

	mean_reward = []
	scores = []

	episodes = 1000
	with open("../result/DQN_without_prioritized.txt", "w") as txtfile:
	    for episode in range(1, episodes+1):
	        print(f"episode: {episode}")
	        state = env.reset()
	        done = False
	        temp_result = {}
	        score = 0 
	        while done!= True:
	        	action, _states = model.predict(state)
	        	n_state, reward, done, info = env.step(action)
	        	score+=reward
	        mean_reward.append(score)
	        scores.append(info[0]['score'])

	        temp = str(episode) + "," + str(score[0]) + "," + str(info[0]['score']) + "\n"
	        txtfile.write(temp)

	        mean = sum(mean_reward)/len(mean_reward)
	        mean_score = sum(scores)/len(scores)

	        print(f"The mean reward is {mean}")
	        print(f"The mean score reward is {mean_score}")
	        print(f"The max score is {max(scores)}")

	    myenv.close()


	mean = sum(mean_reward)/len(mean_reward)
	mean_score = sum(scores)/len(scores)

	print(f"The mean reward is {mean}")
	print(f"The mean score reward is {mean_score}")
	print(f"The max score is {max(scores)}")

	txtfile.close()