
import numpy as np
import cv2
import mss
import pyautogui
import time

from config import *

class SubwaySurfersEnv:

    # Initialize screen capture, game region, etc.
    def __init__(self):
        self.monitor = CAPTURE_REGION
        self.last_frame = None

    def capture_screen(self):
        with mss.mss() as sct:
            img = np.array(sct.grab(self.monitor))
        return img

    # Convert screenshot to grayscale
    def convert_to_grayscale(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # down-scale to 84 × 168 while preserving aspect
        return cv2.resize(gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    
    def perform_action(self, action):
        key = ACTION_KEY_MAPPING.get(action)
    
        if key in ["left", "right", "up", "down"]:
            pyautogui.press(key)
        
        time.sleep(ACTION_DELAY)

    # Reset game state and return initial preprocessed state
    def reset(self):
        # print("[Env] Resetting the game automatically...")
        time.sleep(0.75)

        # Click the play button on the death popup
        # Click twice because we want to skip animation and click play
        pyautogui.click(PLAY_BUTTON)
        time.sleep(0.05)
        pyautogui.click(PLAY_BUTTON)
        time.sleep(1.5)  # Wait for game to restart
         
        # Capture the initial frame after reset and return it
        frame = self.get_processed_frame()
        return frame

    # Take an action (simulate key), capture new frame, calculate reward, return (next_state, reward, done)
    def step(self, action):
        self.perform_action(action)
        time.sleep(0.1)
        next_state = self.get_raw_frame()
        done = self.is_game_over(next_state)
        reward = self.calculate_reward(done)
        processed = self.convert_to_grayscale(next_state)
        return processed, reward, done

    def calculate_reward(self, done):
        if done:
            return DYING_REWARD
        else:
            return STEP_REWARD

    # Check for game over condition. Get pixel RGB value at the game over location
    def is_game_over(self, state):
        frame = state
        x, y = GAME_OVER_PIXEL
        _, g, _ = frame[y, x][:3]
        _, expected_g, _ = GAME_OVER_RGB
        match1 = abs(g - expected_g) > GAME_OVER_TOLERANCE
        
        # print(f"[Env Debug] Pixel RGB at {x},{y}: ({g}) - Match expected? {match}")
        # if (match1):
        #     print("GAME OVER DETECTED")
        
        return match1
    
    def get_processed_frame(self):
        raw_frame = self.capture_screen()
        self.last_frame = self.convert_to_grayscale(raw_frame)
        return self.last_frame
    
    def get_raw_frame(self):
        return self.capture_screen()