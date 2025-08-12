
import random
import numpy as np
from collections import deque
from config import *

class ReplayBuffer:

    def __init__(self, capacity: int, frame_stack_size: int = FRAME_STACK_SIZE_DEFAULT):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.frame_stack_size = frame_stack_size

    def push(self, state_frames: np.ndarray, action: int, reward: float, next_state_frames: np.ndarray, done: bool):
        self.buffer.append((state_frames, action, reward, next_state_frames, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state_list, action_list, reward_list, next_state_list, done_list = zip(*batch)

        state_batch = np.stack(state_list, axis=0).astype(np.float32)
        action_batch = np.array(action_list, dtype=np.int64)
        reward_batch = np.array(reward_list, dtype=np.float32)
        next_state_batch = np.stack(next_state_list, axis=0).astype(np.float32)
        done_batch = np.array(done_list, dtype=np.float32)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)