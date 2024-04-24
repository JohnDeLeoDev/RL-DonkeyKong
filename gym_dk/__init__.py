"""Registration code of Gym environments in this package."""
from .dk_env import DonkeyKongEnv
from ._registration import make, vector

# define the outward facing API of this package
__all__ = [
    make.__name__,
    vector.__name__,
    vector.make.__name__,
    DonkeyKongEnv.__name__,

]




