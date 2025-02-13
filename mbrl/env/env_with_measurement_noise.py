import numpy as np
import gymnasium as gym

class EnvWithMeasurementNoise(gym.Env):
    """ Environment which takes an environment and adds measurement noise to it.
    """
    def __init__(self, noise_std_dev, env):
        self.env = env
        self.noise_std_dev = noise_std_dev

    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space

    def step(self, *args, **kwargs):
        result = self.env.step(*args, **kwargs)
        result = (add_noise_to_state(result[0], self.noise_std_dev),) + result[1:]
        return result

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.env.close(*args, **kwargs)
        
def add_noise_to_state(state: np.array, std_dev):
    noise_samples = np.random.normal(loc=0, scale=std_dev, size=state.shape)
    noisy_state = noise_samples + state
    return noisy_state