import unittest, numpy as np
from nes_py import NESEnv
from nes_py.wrappers.vision_only import VisionOnlyNES
from .rom_file_abs_path import rom_file_abs_path

# NES bits: A(0), B(1), Select(2), Start(3), Up(4), Down(5), Left(6), Right(7)
BTN_A, BTN_B, BTN_START, BTN_RIGHT = 1<<0, 1<<1, 1<<3, 1<<7

# ---------- tiny NCC tracker (pixels only) ----------
def _gray(img): return img.mean(axis=2).astype(np.float32)

def _pick_template(prev, curr, k=18):
    g0, g1 = _gray(prev), _gray(curr)
    diff = np.abs(g1 - g0)
    H, W = diff.shape
    y0, y1 = int(0.35*H), int(0.95*H)   # avoid HUD/top
    x0, x1 = 0, int(0.80*W)             # avoid far-right
    best, by, bx = -1, None, None
    for y in range(y0, y1-k, 2):
        row = diff[y:y+k, x0:x1]
        # coarse x stride for speed
        for x in range(x0, x1-k, 2):
            s = float(row[:, x-x0:x-x0+k].sum())
            if s > best:
                best, by, bx = s, y + k//2, x + k//2
    if by is None:
        return None, None, None
    ys, xs = max(0, by-k//2), max(0, bx-k//2)
    templ = curr[ys:ys+k, xs:xs+k].copy()
    return (by, bx), templ

def _track(curr, templ, yx_prev, search=32):
    if templ is None or yx_prev is None: return None, -1e18
    kH, kW, _ = templ.shape
    H, W, _ = curr.shape
    y, x = map(int, yx_prev)
    y0, y1 = max(0, y-search), min(H-kH, y+search)
    x0, x1 = max(0, x-search), min(W-kW, x+search)
    if y0 >= y1 or x0 >= x1: return None, -1e18
    T = _gray(templ); T0 = T - T.mean(); tnorm = np.linalg.norm(T0) + 1e-6
    G = _gray(curr)
    best, best_yx = -1e18, (y, x)
    for yy in range(y0, y1+1, 2):
        for xx in range(x0, x1+1, 2):
            win = G[yy:yy+kH, xx:xx+kW]
            W0 = win - win.mean()
            denom = (np.linalg.norm(W0)*tnorm) + 1e-6
            ncc = float((W0*T0).sum()/denom)
            if ncc > best:
                best, best_yx = ncc, (yy + kH//2, xx + kW//2)
    return best_yx, best

class ShouldShowParabolicJumpFromPixels(unittest.TestCase):
    def test(self):
        env = VisionOnlyNES(NESEnv(rom_file_abs_path('super-mario-bros-1.nes')))
        try:
            # Unpause and coast
            env.reset(); env.step(BTN_START); [env.step(0) for _ in range(20)]
            # Warm up to make Mario visible & moving (no background requirement)
            prev = env.step(BTN_RIGHT | BTN_B)[0]
            for _ in range(15): prev = env.step(BTN_RIGHT | BTN_B)[0]
            curr = env.step(BTN_RIGHT | BTN_B)[0]
            yx, templ = _pick_template(prev, curr, k=18)
            self.assertIsNotNone(templ, "Failed to seed player template from pixels")

            # Record y(t) during an actual jump: HOLD A ~12 frames to ensure lift
            ys = []
            for i in range(2): env.step(BTN_RIGHT | BTN_B)  # small run-up
            for i in range(12):  # hold A for jump height
                obs, _, _, _ = env.step(BTN_RIGHT | BTN_B | BTN_A)
                yx, _ = _track(obs, templ, yx, search=28)
                ys.append(yx[0])
            for i in range(22):  # release A, keep forward motion
                obs, _, _, _ = env.step(BTN_RIGHT | BTN_B)
                yx, _ = _track(obs, templ, yx, search=28)
                ys.append(yx[0])

            # Fit y = a t^2 + b t + c in image coords (downward is +)
            t = np.arange(len(ys)); y = np.array(ys, float)
            A = np.vstack([t**2, t, np.ones_like(t)]).T
            a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
            y_hat = A @ np.array([a,b,c])
            r2 = 1 - ((y-y_hat)**2).sum()/max(1e-6, ((y-y.mean())**2).sum())

            self.assertGreater(a, 0.02, f"curvature too small: a={a}")
            self.assertGreater(r2, 0.85, f"parabola fit weak: R2={r2}")
        finally:
            env.close()
