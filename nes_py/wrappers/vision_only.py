# nes_py/wrappers/vision_only.py
class VisionOnlyNES:
    """
    Proxy NES env that forbids access to internal RAM (privileged state).
    Use this to guarantee 'from pixels' setting.
    """
    def __init__(self, base_env):
        self._env = base_env

    # forward everything except ram
    def __getattr__(self, name):
        if name == "ram":
            raise RuntimeError("RAM access disabled: vision-only mode")
        return getattr(self._env, name)

    # gym.Env interface passthroughs
    def reset(self, *a, **k):   return self._env.reset(*a, **k)
    def step(self, *a, **k):    return self._env.step(*a, **k)
    def render(self, *a, **k):  return self._env.render(*a, **k)
    def close(self):            return self._env.close()
    @property
    def action_space(self):     return self._env.action_space
    @property
    def observation_space(self):return self._env.observation_space
