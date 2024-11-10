# Assumption that the observation is a tensor of shape (batch_size, obs_dim)
def cartpole_reduce_fn(obs):
    """obs: cart position, cart velocity, pole angle, pole velocity
    """
    return obs[:,0].item()

def hopper_reduce_fn(obs):
    """obs: ?
    dim 11 should be in [-10,10]
    """
    return obs[:,11].item()