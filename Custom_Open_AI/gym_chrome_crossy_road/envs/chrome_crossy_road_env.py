from cv2 import cv2 as cv
import numpy as np
import time
from PIL import Image, ImageGrab, ImageTk
from io import BytesIO
import base64
import gym
from gym import spaces
from gym_chrome_crossy_road.game import CrossyRoadGame
from PIL import Image

# model = "DQN_ERM"
# model = "DQN_PER"
# model = "A2C"
model = "PPO"
# model = "PPO_test"

class ChromeCrossyRoadEnv(gym.Env):
    def __init__(self):
        self.game = CrossyRoadGame()

        n_actions = 5
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 80), dtype=np.uint8)

        self.current_frame = self.observation_space.low
        self._action_set = [0, 1, 2, 3, 4]
        
    
    def get_score(self):
        return self.game.get_score()

    def _observe(self):
        image = self.game.get_canvas()
        image = np.array(Image.open(BytesIO(base64.b64decode(image))))
        image = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
        image = cv.Canny(image, threshold1 = 100, threshold2 = 200)
        image = cv.resize(image, (80,80))
        self.current_frame = image
        return self.current_frame
    
        
    def step(self, action):
        reward = 0

        if action == 0:
            pass
        if action == 1:
            self.game.press_up()
            tree = self.game.font_tree()
        if action == 2:
            self.game.press_left()
            tree = self.game.left_tree()
        if action == 3:
            self.game.press_right()
            tree = self.game.right_tree()
        if action == 4:
            self.game.press_down()
            tree = self.game.back_tree()

        game_score_after_action = self.get_score()
        observation = self._observe()
        done = False
        info = {}
        # game_score_after_action = 
        crashed = self.game.is_crashed()

        if crashed == False:
            if action == 0:
                reward -= 10 
            if action == 1 and tree == 0:
                reward += 7
            if action == 1 and tree == 1:
                reward -=2
            if action == 2 and tree == 0:
                reward += 0
            if action == 2 and tree == 1:
                reward -=1.5
            if action == 3 and tree == 0:
                reward += 0
            if action == 3 and tree == 1:
                reward -=1.5
            if action == 4:
                reward -=10

        if crashed == True:
            reward -= 200
            game_score = game_score_after_action
            info['score'] = game_score
            done = True
            with open ("score_result/score_" + model + ".txt", "a") as txt:
                txt.write(str(game_score_after_action)) 
                txt.write(",") 
            txt.close()


        if action == 0:
            walk = "Stay"
        elif action == 1:
            walk = "Forward"
        elif action == 2:
            walk = "Left"
        elif action == 3:
            walk = "Right"
        else:
            walk = "Backward"

        print(f"Step:{walk}, Reward:{reward}, Game Score:{game_score_after_action}")
        return observation, reward, done, info
    
    def reset(self):
        self.game.restart()
        return self._observe()
    
    def close (self):
        self.game.close()
        
    def render(self, mode='human'):
        pass
        
    
    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    
ACTION_MEANING = {
    0 : "STAY STILL",
    1 : "FORWARD",
    2 : "LEFT",
    3 : "RIGHT",
    4 : "BACKWARD"
}
        
        