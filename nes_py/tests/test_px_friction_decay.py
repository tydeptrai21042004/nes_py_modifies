import unittest, numpy as np
from nes_py import NESEnv
from nes_py.wrappers.vision_only import VisionOnlyNES
from nes_py.wrappers.pixel_reward import PixelShiftReward
from .rom_file_abs_path import rom_file_abs_path

BTN_RIGHT, BTN_B, BTN_START = 1<<7, 1<<1, 1<<3

class ShouldDecayHorizontalMotionAfterRelease(unittest.TestCase):
    def test(self):
        env = PixelShiftReward(VisionOnlyNES(NESEnv(rom_file_abs_path('super-mario-bros-1.nes'))),
                               clip=None, shift_scale=1.0)
        try:
            env.reset(); env.step(BTN_START); [env.step(0) for _ in range(20)]
            # accelerate
            for _ in range(40): env.step(BTN_RIGHT | BTN_B)
            # measure decays
            dxs = []
            for _ in range(30):
                _, _, _, info = env.step(0)
                dxs.append(info.get("px_bg_dx", 0.0) if info.get("px_src")=="background"
                           else info.get("px_pl_dx", 0.0))
            dxs = np.array(dxs, float)
            # check monotone-ish decay (allow 4 violations)
            violations = sum(dxs[i] < dxs[i+1] - 0.2 for i in range(len(dxs)-1))
            self.assertLessEqual(violations, 4, f"too many increases: {violations}, dxs={dxs.tolist()}")
        finally:
            env.close()
