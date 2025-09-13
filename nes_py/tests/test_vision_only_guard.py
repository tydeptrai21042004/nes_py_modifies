import unittest
from nes_py import NESEnv
from nes_py.wrappers.vision_only import VisionOnlyNES
from .rom_file_abs_path import rom_file_abs_path

class ShouldBlockRAMInVisionOnly(unittest.TestCase):
    def test(self):
        env = VisionOnlyNES(NESEnv(rom_file_abs_path('super-mario-bros-1.nes')))
        env.reset()
        with self.assertRaises(RuntimeError):
            _ = env.ram  # forbidden
        env.close()
