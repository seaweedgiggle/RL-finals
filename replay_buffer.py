import numpy as np

class ReplayBuffer():
    def __init__(self, max_items=1e6):
        self.buffer = []
        self.max_items = max_items
        self.curr_pos = 0   # 队尾指针

    def push(self, experience):
        # 如果当前经验池没有满，则直接加入经验
        if len(self.buffer) <= self.max_items:
            self.buffer.append(experience)
        
        # 如果满了，则替换队尾经验
        else:
            self.buffer[self.curr_pos] = experience
            self.curr_pos += 1
            self.curr_pos %= self.max_items

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        experiences = [self.buffer[i] for i in indexes]
        state, action, reward, next_state, done = [np.array(arr, copy=False) for arr in list(zip(*experiences))]
        return state, action, reward.reshape(-1, 1), next_state, done.reshape(-1, 1)


        