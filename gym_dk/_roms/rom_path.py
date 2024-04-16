"""A method to load a ROM path."""
import os


# a dictionary mapping ROM paths first by lost levels, then by ROM hack mode
_ROM_PATHS = {
    # the dictionary of lost level ROM paths
    True: {
        'vanilla': 'super-mario-bros-2.nes',
        'downsample': 'super-mario-bros-2-downsample.nes',
    },
    # the dictionary of Super Mario Bros. 1 ROM paths
    False: {
        'vanilla': 'super-mario-bros.nes',
        'pixel': 'super-mario-bros-pixel.nes',
        'rectangle': 'super-mario-bros-rectangle.nes',
        'downsample': 'super-mario-bros-downsample.nes',
    }
}


def rom_path():
    """
    Return the ROM filename for a game and ROM mode.

    Args:
        lost_levels (bool): whether to use the lost levels ROM
        rom_mode (str): the mode of the ROM hack to use as one of:
            - 'vanilla'
            - 'pixel'
            - 'downsample'
            - 'vanilla'

    Returns (str):
        the ROM path based on the input parameters

    """
    # load "DonkeyKong.nes" from the current directory
    rom = os.path.join(os.path.dirname(__file__), 'DonkeyKong.nes')
    

    return rom


# explicitly define the outward facing API of this module
__all__ = [rom_path.__name__]
