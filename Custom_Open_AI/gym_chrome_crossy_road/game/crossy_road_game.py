from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from cv2 import cv2 as cv
import os, pickle, shutil, json, sys
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import numpy as np
import random, time
from collections import deque
import pandas as pd
from IPython.display import clear_output
from IPython.core.display import display
import matplotlib.pyplot as plt
from PIL import Image, ImageGrab, ImageTk
import keras.applications as kapp

game_url = "file:///Users/Namtarn/Documents/Spring2021/csci527/CSCI527_CrossyRoadProject/web/index.html"
# game_url = "http://hausai.com/crossyroad/"
chrome_driver_path = "gym_chrome_crossy_road/chrome_driver/chromedriver"

class CrossyRoadGame():
    def __init__(self):
        self._driver = webdriver.Chrome(chrome_driver_path)
        self._driver.set_window_position(0,0)
        self._driver.set_window_size(850,700)
        self._driver.get(game_url)        
        
    def press_up(self):
        self._driver.find_element_by_id("forward").send_keys(Keys.UP)

    def press_down(self):
        self._driver.find_element_by_id("backward").send_keys(Keys.DOWN)

    def press_left(self):
        self._driver.find_element_by_id('left').send_keys(Keys.LEFT)

    def press_right(self):
        self._driver.find_element_by_id('right').send_keys(Keys.RIGHT)

    def is_crashed(self):
        time.sleep(0.2)
        self.element = self._driver.find_element_by_id('retry')
        return self.element.is_displayed()

    def font_tree(self):
        return int(self._driver.find_element_by_id('font_tree').text)

    def left_tree(self):
        return int(self._driver.find_element_by_id('left_tree').text)

    def right_tree(self):
        return int(self._driver.find_element_by_id('right_tree').text)

    def back_tree(self):
        return int(self._driver.find_element_by_id('back_tree').text)
           
    def restart(self):
        time.sleep(0.2)
        self._driver.refresh()
    
    def close(self):
        self._driver.close()
        
    def get_score(self):
        time.sleep(0.2)
        return int(self._driver.find_element_by_id('counter').text)
    
    def get_canvas(self):
        canvas = self._driver.find_element_by_css_selector("#myCanvasId")
        canvas_base64 = self._driver.execute_script("return arguments[0].toDataURL().substring(21);", canvas)
        return canvas_base64