"""Pixel-only physics test: verify parabolic vertical motion from RGB frames (no RAM)."""

import unittest
import numpy as np
from nes_py.nes_env import NESEnv
from .rom_file_abs_path import rom_file_abs_path
from unittest import TestCase, skip
import numpy as np

from .rom_file_abs_path import rom_file_abs_path
from nes_py.nes_env import NESEnv
# NES controller bitmap matches nes_env.get_keys_to_action():
# [right, left, down, up, start, select, B, A]  (right=bit7 ... A=bit0)
BTN_RIGHT = 1 << 7
BTN_LEFT  = 1 << 6
BTN_DOWN  = 1 << 5
BTN_UP    = 1 << 4
BTN_START = 1 << 3
BTN_SELECT= 1 << 2
BTN_B     = 1 << 1
BTN_A     = 1 << 0

def _step(env, action, n=1):
    for _ in range(n):
        env.step(action)

def _frames_change(env, frames=12, action=0):
    """Sum absolute pixel change over a few frames with given action."""
    s = 0
    prev = env.screen.copy()
    for _ in range(frames):
        env.step(action)
        s += np.abs(env.screen.astype(np.int16) - prev.astype(np.int16)).sum()
        prev = env.screen.copy()
    return int(s)

def _tap_start_until_frames_move(env, attempts=3):
    """Tap START (don’t hold) until frames change -> not paused/title."""
    env.reset()
    _step(env, 0, 20)  # let title settle
    for _ in range(attempts):
        _step(env, BTN_START, 1)  # tap
        moved = _frames_change(env, frames=30, action=0)
        if moved > 0:
            return
        _step(env, 0, 10)
    raise unittest.SkipTest("Could not leave title/pause using pixels only; frames stayed static.")

def _motion_template_from_diff(f0, f1, k=24):

    g0 = f0.mean(axis=2).astype(np.float32)
    g1 = f1.mean(axis=2).astype(np.float32)
    diff = np.abs(g1 - g0)
    H, W = diff.shape
    x_end = int(W * 0.60)
    y_start = int(H * 0.30)  # skip HUD

    best, best_yx = -1.0, None
    for y in range(y_start, H - k, 2):
        row = diff[y:y+k, :x_end]
        # stride columns to keep it light
        for x in range(0, x_end - k, 2):
            score = float(row[:, x:x+k].sum())
            if score > best:
                best = score
                best_yx = (y + k // 2, x + k // 2)
    if best_yx is None:
        raise RuntimeError("Failed to find motion-based template.")
    y, x = best_yx
    y0, x0 = max(0, y - k // 2), max(0, x - k // 2)
    y1, x1 = min(H, y0 + k), min(W, x0 + k)
    return y, x, f1[y0:y1, x0:x1].copy()

def _track_with_ncc(frames, init_y, init_x, templ, search=32):
    """
    Track (y,x) using normalized cross-correlation over a local search window.
    Return arrays ys, xs with one entry per frame.
    """
    templ = templ.astype(np.float32)
    kH, kW, _ = templ.shape
    H, W, _ = frames[0].shape
    ys, xs = [int(init_y)], [int(init_x)]

    # pre-normalize template
    t = templ - templ.mean()
    t_norm = np.sqrt((t**2).sum()) + 1e-6

    y, x = int(init_y), int(init_x)
    for idx in range(1, len(frames)):
        img = frames[idx].astype(np.float32)
        y0 = max(0, y - search); y1 = min(H - kH, y + search)
        x0 = max(0, x - search); x1 = min(W - kW, x + search)
        best, best_yx = -1e18, (y, x)

        for yy in range(y0, y1 + 1, 2):
            for xx in range(x0, x1 + 1, 2):
                win = img[yy:yy+kH, xx:xx+kW]
                w = win - win.mean()
                denom = (np.sqrt((w**2).sum()) + 1e-6) * t_norm
                ncc = float((w * t).sum() / denom)
                if ncc > best:
                    best = ncc
                    best_yx = (yy + kH // 2, xx + kW // 2)
        y, x = best_yx
        # light template update to handle animation
        patch = img[y - kH // 2:y - kH // 2 + kH, x - kW // 2:x - kW // 2 + kW]
        t = (0.8 * t + 0.2 * (patch - patch.mean())).astype(np.float32)
        t_norm = np.sqrt((t**2).sum()) + 1e-6

        ys.append(int(y)); xs.append(int(x))
    return np.array(ys), np.array(xs)

def _fit_parabola(y):
    """Least-squares fit y(t)=a t^2 + b t + c; return (a,b,c,R^2)."""
    t = np.arange(len(y), dtype=np.float32)
    X = np.stack([t**2, t, np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(X, y.astype(np.float32), rcond=None)
    y_hat = X @ coef
    ss_res = float(((y - y_hat)**2).sum())
    ss_tot = float(((y - y.mean())**2).sum()) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    a, b, c = coef.tolist()
    return a, b, c, r2

class ShouldShowParabolicJumpFromPixels(TestCase):
    """
    Pixel-only physical property: jump arc is approximately parabolic (a>0, high R^2).
    - A = jump (hold longer -> higher)
    - B = run (makes horizontal motion consistent)
    - START toggles pause (tap only)
    Sources: NES manual & guides. 
    """

    def test(self):
        env = NESEnv(rom_file_abs_path('super-mario-bros-1.nes'))
        try:
            # 1) Enter gameplay using pixel change (avoid title/pause)
            _tap_start_until_frames_move(env)

            # 2) Warm-up: move RIGHT+B for motion and predictable placement
            _step(env, BTN_RIGHT | BTN_B, 40)
            f0 = env.screen.copy()
            _step(env, BTN_RIGHT | BTN_B, 6)
            f1 = env.screen.copy()

            # 3) Motion-based template from diff (locks onto moving sprite, not background)
            y0, x0, templ = _motion_template_from_diff(f0, f1, k=24)

            # 4) Record frames around a jump (hold A longer for a clean arc)
            frames = [env.screen.copy()]
            _step(env, BTN_RIGHT | BTN_B | BTN_A, 8)   # hold A for 8 frames -> clear jump
            for _ in range(36):                        # total ~44 frames ~0.73s @60fps
                env.step(BTN_RIGHT | BTN_B)
                frames.append(env.screen.copy())

            # Basic sanity: frames changed
            if _frames_change(env, frames=5, action=0) == 0:
                raise unittest.SkipTest("Frames static after start; likely title/pause variant.")

            # 5) Track y(t) with NCC
            ys, xs = _track_with_ncc(frames, y0, x0, templ, search=32)

            span = int(ys.max() - ys.min())
            self.assertGreater(span, 8, f"Too little vertical motion from pixels (span={span}).")

            # 6) Parabolic fit: y grows downwards in image coords => curvature a > 0
            a, b, c, r2 = _fit_parabola(ys)
            self.assertGreater(a, 0.0, f"Quadratic curvature not positive (a={a}).")
            self.assertGreater(r2, 0.80, f"Parabolic fit too weak: R^2={r2:.3f} (ys span={span}).")

            # 7) Check 'up then down' around apex
            t_apex = int(np.argmin(ys))
            left_ok  = len(ys[:t_apex]) < 2 or np.all(np.diff(ys[:t_apex]) <= 1)
            right_ok = len(ys[t_apex:]) < 2 or np.all(np.diff(ys[t_apex:]) >= -1)
            self.assertTrue(left_ok and right_ok, f"Not 'up then down' around apex={t_apex}.")
        finally:
            env.close()
