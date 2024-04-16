"""Methods for ROM file management."""
from .rom_path import rom_path


# explicitly define the outward facing API of this package
__all__ = [
    rom_path.__name__,
]
