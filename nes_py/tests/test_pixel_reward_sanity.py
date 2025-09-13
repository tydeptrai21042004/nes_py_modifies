import unittest, numpy as np
from nes_py import NESEnv
from nes_py.wrappers.vision_only import VisionOnlyNES
from nes_py.wrappers.pixel_reward import PixelShiftReward
from .rom_file_abs_path import rom_file_abs_path

# Button bits (MSB→LSB): [Right, Left, Down, Up, Start, Select, B, A]
BTN_RIGHT = 1<<7; BTN_B = 1<<1  # NES bit order note: A,B,Select,Start,Up,Down,Left,Right. :contentReference[oaicite:5]{index=5}
BTN_START = 1 << 3  

def _step(env, a, n):  # helper
    for _ in range(n): env.step(a)
def _ensure_gameplay(env, taps=3, coast=30):
    """
    Tap START (don't hold) until we detect pixel motion (not paused/title),
    then do a short RIGHT+B warm-up so PixelShiftReward can calibrate sign.
    """
    env.reset()
    for _ in range(taps):
        _step(env, BTN_START, 1)  # one-frame tap
        moved = False
        for _ in range(coast):
            _, _, _, info = env.step(0)
            if abs(info.get("px_dx", 0.0)) > 0.1:
                moved = True
                break
        if moved:
            break
    # warm-up to produce clear scrolling motion
    _step(env, (1 << 7) | (1 << 1), 30)  # RIGHT | B
env = PixelShiftReward(
    VisionOnlyNES(NESEnv(rom_file_abs_path('super-mario-bros-1.nes'))),
    clip=None,            # <- do not cap per-step reward
    shift_scale=0.5       # <- scale down a bit to avoid huge values
)

def _settle(env, steps=10, action=0):
    for _ in range(steps):
        env.step(action)  # discard these rewards

class ShouldGiveHigherPixelRewardWhenMovingRight(unittest.TestCase):
    def test(self):
        try:
            # A) NOOP
            _ensure_gameplay(env)
            _settle(env, 10, 0)                # <- flush calibration/spikes
            r0 = 0.0
            for _ in range(60):                 # <- slightly longer horizon
                _, rew, _, _ = env.step(0)
                r0 += rew

            # B) RIGHT+B
            env.reset()
            _ensure_gameplay(env)
            _settle(env, 10, 0)
            r1 = 0.0
            for _ in range(60):
                _, rew, _, info = env.step(BTN_RIGHT | BTN_B)
                r1 += rew

            self.assertGreater(
                r1, max(r0 * 1.5, 1.5),        # <- relative + modest absolute floor
                f"pixel-reward didn’t rise with RIGHT: noop={r0}, move={r1}, sign={info.get('px_sign')}, dx={info.get('px_dx')}"
            )
        finally:
            env.close()
