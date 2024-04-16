"""Registration code of Gym environments in this package."""
import gym


def _register_dk_env(id, is_random=False, **kwargs):
    """
    Register a Super Mario Bros. (1/2) environment with OpenAI Gym.

    Args:
        id (str): id for the env to register
        is_random (bool): whether to use the random levels environment
        kwargs (dict): keyword arguments for the SuperMarioBrosEnv initializer

    Returns:
        None

    """
    # if the is random flag is set
    if is_random:
        # set the entry point to the random level environment
        entry_point = 'gym_dk:DonkeyKongEnv'
    else:
        # set the entry point to the standard Super Mario Bros. environment
        entry_point = 'gym_dk:DonkeyKongEnv'
    # register the environment
    gym.envs.registration.register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=9999999,
        reward_threshold=9999999,
        kwargs=kwargs,
        nondeterministic=True,
    )


# Super Mario Bros.
_register_dk_env('DonkeyKong-v0')

# create an alias to gym.make for ease of access
make = gym.make
vector = gym.vector
vector.make = gym.vector.make

# define the outward facing API of this module (none, gym provides the API)
__all__ = [make.__name__]



