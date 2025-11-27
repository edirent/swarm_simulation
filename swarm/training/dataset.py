import random
from dataclasses import dataclass
from typing import List


@dataclass
class Transition:
    obs: object
    action: object
    reward: float
    next_obs: object
    done: bool


class ExperienceDataset:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Transition] = []

    def add(self, transition: Transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, k=min(batch_size, len(self.buffer)))
