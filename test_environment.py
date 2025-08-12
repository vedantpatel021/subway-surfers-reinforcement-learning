import cv2
import time
import os
import random
import keyboard  # Global keyboard listener
from environment import SubwaySurfersEnv
from config import *

# CONFIG
SAVE_IMAGES = SAVE_IMAGE_CAPTURES
SAVE_PATH = "imgs/logs/testing"

# Setup
if SAVE_IMAGES:
    os.makedirs(SAVE_PATH, exist_ok=True)

env = SubwaySurfersEnv()

# Initial reset
state = env.reset()
print("[Test] Environment reset. Initial frame captured.")

frame_count = 0

try:
    while True:
        action = random.choice(ACTIONS)
        print(f"[Test] Taking action: {action}")

        next_state, reward, done = env.step(action)

        print(f"[Test] Step {frame_count + 1} -> Reward: {reward}, Done: {done}\n")

        # if SAVE_IMAGES:
        #     filename = os.path.join(SAVE_PATH, f"frame_{frame_count:05d}.png")
        #     cv2.imwrite(filename, next_state)

        frame_count += 1

        if done:
            print("[Test] Game over detected! Resetting environment...")
            state = env.reset()
            frame_count = 0

except KeyboardInterrupt:
    print("[Test] Interrupted by Ctrl+C. Exiting...")

except Exception as e:
    print(f"[Test] Exception occurred: {e}")

finally:
    print("[Test] Cleanup complete. Goodbye.")
