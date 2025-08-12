
import numpy as np
from collections import deque
from config import FRAME_STACK_SIZE_DEFAULT

class FrameStack:

    def __init__(self, stack_size: int = FRAME_STACK_SIZE_DEFAULT):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, first_frame: np.ndarray):
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(first_frame)

    def push(self, new_frame: np.ndarray) -> np.ndarray:
        self.frames.append(new_frame)
        assert len(self.frames) == self.stack_size
        return np.stack(self.frames, axis=0)