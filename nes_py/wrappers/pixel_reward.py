# nes_py/wrappers/pixel_reward.py
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# ---------- utils ----------

def _to_small_gray(rgb, h=84, w=84):
    g = rgb.mean(axis=2).astype(np.float32)
    if _HAS_CV2:
        return cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    # NumPy fallback: block-average
    sh, sw = max(1, rgb.shape[0] // h), max(1, rgb.shape[1] // w)
    g = g[:h*sh, :w*sw]
    return g.reshape(h, sh, w, sw).mean(axis=(1, 3))

def estimate_x_shift(prev, curr, max_shift=12):
    """
    Row-wise NCC between downscaled grayscale frames.
    Returns signed shift s (pixels): scene LEFT -> s<0, RIGHT -> s>0.
    """
    p = _to_small_gray(prev); c = _to_small_gray(curr)
    rows = slice(int(0.35 * p.shape[0]), int(0.85 * p.shape[0]))  # avoid HUD/ground
    p, c = p[rows, :], c[rows, :]
    if np.allclose(p, c):
        return 0.0
    shifts = []
    for r in range(0, p.shape[0], 2):
        pr, cr = p[r], c[r]
        best, best_s = -1e18, 0
        for s in range(-max_shift, max_shift + 1):
            if s < 0:
                a, b = pr[-s:], cr[:pr.size + s]
            elif s > 0:
                a, b = pr[:-s], cr[s:]
            else:
                a, b = pr, cr
            if a.size < 12:
                continue
            a0 = a - a.mean(); b0 = b - b.mean()
            denom = (np.linalg.norm(a0) * np.linalg.norm(b0)) + 1e-6
            ncc = float((a0 * b0).sum() / denom)
            if ncc > best:
                best, best_s = ncc, s
        shifts.append(best_s)
    return float(np.median(shifts))

def _pick_motion_template(prev_rgb, curr_rgb, k=24):
    """
    Pick a kxk template from the area with strongest |curr - prev| motion,
    bias to lower-middle/left area (avoid HUD), return (y, x, template_rgb).
    """
    g0 = prev_rgb.mean(axis=2).astype(np.float32)
    g1 = curr_rgb.mean(axis=2).astype(np.float32)
    diff = np.abs(g1 - g0)
    H, W = diff.shape
    y0 = int(0.30 * H)      # skip HUD
    y1 = int(0.95 * H)
    x0 = 0
    x1 = int(0.70 * W)      # avoid far-right edge

    best, cy, cx = -1.0, None, None
    for y in range(y0, y1 - k, 2):
        row = diff[y:y+k, x0:x1]
        # stride x as well
        for x in range(x0, x1 - k, 2):
            score = float(row[:, x - x0:x - x0 + k].sum())
            if score > best:
                best = score
                cy, cx = y + k // 2, x + k // 2
    if cy is None:
        return None, None, None
    y_start = max(0, cy - k // 2)
    x_start = max(0, cx - k // 2)
    templ = curr_rgb[y_start:y_start + k, x_start:x_start + k].copy()
    return cy, cx, templ

def _track_ncc(curr_rgb, templ_rgb, yx_prev, search=32):
    """
    Track templ position in curr_rgb with NCC in a local window around yx_prev.
    Returns (y_new, x_new, ncc_score). Pixel coords in full-res image space.
    """
    if templ_rgb is None or yx_prev[0] is None:
        return None, None, -1e18
    kH, kW, _ = templ_rgb.shape
    H, W, _ = curr_rgb.shape
    y, x = int(yx_prev[0]), int(yx_prev[1])

    y0 = max(0, y - search); y1 = min(H - kH, y + search)
    x0 = max(0, x - search); x1 = min(W - kW, x + search)
    if y0 >= y1 or x0 >= x1:
        return None, None, -1e18

    # grayscale for NCC
    T = templ_rgb.mean(axis=2).astype(np.float32)
    T0 = T - T.mean()
    tnorm = np.sqrt((T0*T0).sum()) + 1e-6

    best, best_yx = -1e18, (y, x)
    G = curr_rgb.mean(axis=2).astype(np.float32)
    for yy in range(y0, y1 + 1, 2):
        for xx in range(x0, x1 + 1, 2):
            win = G[yy:yy+kH, xx:xx+kW]
            W0 = win - win.mean()
            denom = (np.sqrt((W0*W0).sum()) + 1e-6) * tnorm
            ncc = float((W0 * T0).sum() / denom)
            if ncc > best:
                best, best_yx = ncc, (yy + kH // 2, xx + kW // 2)
    return best_yx[0], best_yx[1], best

# ---------- main wrapper ----------

class PixelShiftReward:
    """
    Pixel-only progress reward:
      - background horizontal shift via NCC (global)
      - player local motion via NCC template tracking
    Uses player motion when bg shift ~ 0 (pre-scroll), else uses bg (with auto sign).
    """
    def __init__(self, base_env,
                 clip=None,                # default None: no per-step cap
                 shift_scale=1.0,
                 sign_threshold=0.7,       # bg shift needed to calibrate sign
                 settle_steps_after_calib=5,
                 player_dx_thresh=0.5,
                 bg_dx_thresh=0.5,
                 track_search=32,
                 templ_size=24):
        self._env = base_env
        self._prev = None
        self.clip = clip
        self.shift_scale = shift_scale

        self._sign = None
        self._sign_threshold = sign_threshold
        self._settle_steps_after_calib = settle_steps_after_calib
        self._settle_counter = 0

        self._templ = None
        self._yx = (None, None)
        self._track_search = track_search
        self._templ_size = templ_size

        self._player_dx_thresh = player_dx_thresh
        self._bg_dx_thresh = bg_dx_thresh

    def reset(self, *a, **k):
        obs = self._env.reset(*a, **k)
        self._prev = obs.copy()
        self._sign = None
        self._settle_counter = 0
        self._templ = None
        self._yx = (None, None)
        return obs

    def step(self, action):
        obs, _, done, info0 = self._env.step(action)

        # --- background shift (global) ---
        dx_bg = estimate_x_shift(self._prev, obs)
        if self._sign is None and abs(dx_bg) >= self._sign_threshold:
            self._sign = -1.0 if dx_bg < 0 else 1.0   # scrolling right -> bg dx < 0
            self._settle_counter = self._settle_steps_after_calib

        # --- player tracking (local) ---
        if self._templ is None:
            # seed template using motion between prev and obs
            y, x, templ = _pick_motion_template(self._prev, obs, k=self._templ_size)
            self._templ, self._yx = templ, (y, x)
            dx_pl = 0.0
            ncc = -1e18
        else:
            y_new, x_new, ncc = _track_ncc(obs, self._templ, self._yx, search=self._track_search)
            if y_new is None:
                dx_pl = 0.0
            else:
                dx_pl = float(x_new - self._yx[1])   # + to the right (image coords)
                # light template update
                k = self._templ_size
                ys = max(0, y_new - k // 2); xs = max(0, x_new - k // 2)
                patch = obs[ys:ys+k, xs:xs+k]
                if patch.shape == self._templ.shape:
                    self._templ = (0.8 * self._templ + 0.2 * patch).astype(np.uint8)
                self._yx = (y_new, x_new)

        # --- choose source ---
        use_player = (abs(dx_bg) < self._bg_dx_thresh) and (abs(dx_pl) >= self._player_dx_thresh)
        if self._settle_counter > 0:
            self._settle_counter -= 1
            rew_core = 0.0
            src = "settle"
        elif use_player:
            rew_core = max(dx_pl, 0.0)        # rightward local motion
            src = "player"
        else:
            if self._sign is None:
                rew_core = 0.0
            else:
                rew_core = max(self._sign * dx_bg, 0.0)  # rightward scroll
            src = "background"

        rew = rew_core * self.shift_scale
        if self.clip is not None:
            rew = float(np.clip(rew, self.clip[0], self.clip[1]))

        # info
        info = dict(info0 or {})
        info.update({
            "px_bg_dx": dx_bg,
            "px_pl_dx": dx_pl,
            "px_sign": self._sign,
            "px_src": src,
            "px_rew": rew,
        })

        self._prev = obs.copy()
        return obs, rew, done, info

    # passthroughs
    def render(self, *a, **k):  return self._env.render(*a, **k)
    def close(self):            return self._env.close()
    @property
    def action_space(self):     return self._env.action_space
    @property
    def observation_space(self):return self._env.observation_space
