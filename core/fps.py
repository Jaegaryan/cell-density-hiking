import glfw


class FPS:
    def __init__(self, w=0.9):
        self.w = w  # exponential weight of last time step

        self.last_time = glfw.get_time()
        self.delta = 0
        self.running_delta = 0
        self.fps = 0

    def update(self):
        current_time = glfw.get_time()
        self.delta = current_time - self.last_time
        self.last_time = current_time
        self.running_delta = self.w * self.running_delta + (1 - self.w) * self.delta
        self.fps = 1 / self.running_delta

        return self.fps
