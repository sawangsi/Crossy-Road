import gym
import gym_chrome_crossy_road
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


programing_type = int(sys.argv[1])
environment_name = 'ChromeCrossyRoad-v0'

# Start to train the agent
if programing_type == 0:
	env = gym.make(environment_name)
	model = PPO("MlpPolicy", env, learning_rate=0.0001, gamma=0.7, batch_size=1024, verbose=1, tensorboard_log ="./log/ppo_crossy_road_tensorboard/")
	model.learn(total_timesteps=30000)
	model.save("../model/ppo")
	env.close()

# Continue to train
elif programing_type == 1:
	myenv = gym.make(environment_name)
	env = DummyVecEnv([lambda: myenv])
	model = PPO.load('../model/ppo', env=env)
	model.set_env(env)
	model.learn(total_timesteps=150000,callback=None, reset_num_timesteps=False)
	model.save("../model/ppo")
	env.close()

# Test the agent
else:
	myenv = gym.make(environment_name)
	env = DummyVecEnv([lambda: myenv])
	model = PPO.load('../model/ppo', env=env)
	result = {}

	mean_reward = []
	scores = []

	episodes = 100
	with open("../result/PPO.txt", "w") as txtfile:
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

# tensorboard --logdir ./log/ppo_crossy_road_tensorboard
