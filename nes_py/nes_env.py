"""A CTypes interface to the C++ NES environment."""
import ctypes
import glob
import itertools
import os
import sys
import gym
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
from ._rom import ROM
from ._image_viewer import ImageViewer

# ---------------------------
# DEBUG CONTROLS (new)
# ---------------------------
# NESPY_DEBUG: "0"/"false"/"False" to disable, anything else enables
_DEBUG = os.environ.get("NESPY_DEBUG", "1").strip() not in ("0", "false", "False", "")
# Print stats for the first N steps
_DEBUG_STEPS = int(os.environ.get("NESPY_DEBUG_STEPS", "3"))

def _dbg(*args, **kwargs):
    """Print with a consistent NES-PY debug prefix if enabled."""
    if _DEBUG:
        print("[NES-PY DEBUG]", *args, **kwargs)

# the path to the directory this file is in
_MODULE_PATH = os.path.dirname(__file__)
# the pattern to find the C++ shared object library
_SO_PATH = 'lib_nes_env*'
# the absolute path to the C++ shared object library
_LIB_PATH = os.path.join(_MODULE_PATH, _SO_PATH)
# load the library from the shared object file
try:
    _LIB = ctypes.cdll.LoadLibrary(glob.glob(_LIB_PATH)[0])
    _dbg("Loaded native core:", glob.glob(_LIB_PATH)[0])
except IndexError:
    raise OSError('missing static lib_nes_env*.so library!')

# setup the argument and return types for Width
_LIB.Width.argtypes = None
_LIB.Width.restype = ctypes.c_uint
# setup the argument and return types for Height
_LIB.Height.argtypes = None
_LIB.Height.restype = ctypes.c_uint
# setup the argument and return types for Initialize
_LIB.Initialize.argtypes = [ctypes.c_wchar_p]
_LIB.Initialize.restype = ctypes.c_void_p
# setup the argument and return types for Controller
_LIB.Controller.argtypes = [ctypes.c_void_p, ctypes.c_uint]
_LIB.Controller.restype = ctypes.c_void_p
# setup the argument and return types for Screen
_LIB.Screen.argtypes = [ctypes.c_void_p]
_LIB.Screen.restype = ctypes.c_void_p
# setup the argument and return types for GetMemoryBuffer
_LIB.Memory.argtypes = [ctypes.c_void_p]
_LIB.Memory.restype = ctypes.c_void_p
# setup the argument and return types for Reset
_LIB.Reset.argtypes = [ctypes.c_void_p]
_LIB.Reset.restype = None
# setup the argument and return types for Step
_LIB.Step.argtypes = [ctypes.c_void_p]
_LIB.Step.restype = None
# setup the argument and return types for Backup
_LIB.Backup.argtypes = [ctypes.c_void_p]
_LIB.Backup.restype = None
# setup the argument and return types for Restore
_LIB.Restore.argtypes = [ctypes.c_void_p]
_LIB.Restore.restype = None
# setup the argument and return types for Close
_LIB.Close.argtypes = [ctypes.c_void_p]
_LIB.Close.restype = None

# height in pixels of the NES screen
SCREEN_HEIGHT = _LIB.Height()
# width in pixels of the NES screen
SCREEN_WIDTH = _LIB.Width()
# shape of the screen as 24-bit RGB (standard for NumPy)
SCREEN_SHAPE_24_BIT = SCREEN_HEIGHT, SCREEN_WIDTH, 3
# shape of the screen as 32-bit RGB (C++ memory arrangement)
SCREEN_SHAPE_32_BIT = SCREEN_HEIGHT, SCREEN_WIDTH, 4
# create a type for the screen tensor matrix from C++
SCREEN_TENSOR = ctypes.c_byte * int(np.prod(SCREEN_SHAPE_32_BIT))

# create a type for the RAM vector from C++
RAM_VECTOR = ctypes.c_byte * 0x800

# create a type for the controller buffers from C++
CONTROLLER_VECTOR = ctypes.c_byte * 1

# Early debug about shapes (printed at import)
_dbg(
    f"SCREEN_WIDTHxHEIGHT: {SCREEN_WIDTH}x{SCREEN_HEIGHT} | "
    f"32-bit shape {SCREEN_SHAPE_32_BIT} -> 24-bit {SCREEN_SHAPE_24_BIT} | "
    f"endianness={sys.byteorder}"
)

class NESEnv(gym.Env):
    """An NES environment based on the LaiNES emulator."""

    # relevant meta-data about the environment
    metadata = {
        'render.modes': ['rgb_array', 'human'],
        'video.frames_per_second': 60
    }

    # the legal range for rewards for this environment
    reward_range = (-float('inf'), float('inf'))

    # observation space for the environment is static across all instances
    observation_space = Box(
        low=0,
        high=255,
        shape=SCREEN_SHAPE_24_BIT,
        dtype=np.uint8
    )

    # action space is a bitmap of button press values for the 8 NES buttons
    action_space = Discrete(256)

    def __init__(self, rom_path):
        """
        Create a new NES environment.

        Args:
            rom_path (str): the path to the ROM for the environment
        """
        _dbg("Initializing NESEnv with ROM:", rom_path)
        # create a ROM file from the ROM path
        rom = ROM(rom_path)
        # check that there is PRG ROM
        if rom.prg_rom_size == 0:
            raise ValueError('ROM has no PRG-ROM banks.')
        # ensure that there is no trainer
        if rom.has_trainer:
            raise ValueError('ROM has trainer. trainer is not supported.')
        # try to read the PRG ROM and raise a value error if it fails
        _ = rom.prg_rom
        # try to read the CHR ROM and raise a value error if it fails
        _ = rom.chr_rom
        # check the TV system
        if rom.is_pal:
            raise ValueError('ROM is PAL. PAL is not supported.')
        # check that the mapper is implemented
        elif rom.mapper not in {0, 1, 2, 3}:
            msg = ('ROM has an unsupported mapper number {}. please see '
                   'https://github.com/Kautenja/nes-py/issues/28 for more information.')
            raise ValueError(msg.format(rom.mapper))
        # create a dedicated random number generator for the environment
        self.np_random = np.random.RandomState()
        # store the ROM path
        self._rom_path = rom_path
        # initialize the C++ object for running the environment
        self._env = _LIB.Initialize(self._rom_path)
        # setup a placeholder for a 'human' render mode viewer
        self.viewer = None
        # setup a placeholder for a pointer to a backup state
        self._has_backup = False
        # setup a done flag
        self.done = True
        # setup the controllers, screen, and RAM buffers
        self.controllers = [self._controller_buffer(port) for port in range(2)]
        self.screen = self._screen_buffer()
        self.ram = self._ram_buffer()

        # Debug: buffer addresses + shapes
        self._dbg_step_count = 0
        _dbg("observation_space:", self.observation_space)
        _dbg("screen buffer:",
             {"shape": self.screen.shape, "dtype": str(self.screen.dtype),
              "ptr": hex(self.screen.ctypes.data)})
        _dbg("ram buffer:",
             {"shape": self.ram.shape, "dtype": str(self.ram.dtype),
              "ptr": hex(self.ram.ctypes.data)})

    def _screen_buffer(self):
        """Setup the screen buffer from the C++ code."""
        # get the address of the screen
        address = _LIB.Screen(self._env)
        # create a buffer from the contents of the address location
        buffer_ = ctypes.cast(address, ctypes.POINTER(SCREEN_TENSOR)).contents
        # create a NumPy array from the buffer
        screen = np.frombuffer(buffer_, dtype='uint8')
        # reshape the screen from a column vector to a tensor
        screen = screen.reshape(SCREEN_SHAPE_32_BIT)
        # flip the bytes if the machine is little-endian (which it likely is)
        if sys.byteorder == 'little':
            # invert the little-endian BGRx channels to big-endian xRGB
            screen = screen[:, :, ::-1]
        # remove the 0th axis (padding from storing colors in 32 bit)
        out = screen[:, :, 1:]
        _dbg("_screen_buffer(): built observation view",
             {"raw32_shape": SCREEN_SHAPE_32_BIT,
              "final24_shape": out.shape,
              "dtype": str(out.dtype)})
        return out

    def _ram_buffer(self):
        """Setup the RAM buffer from the C++ code."""
        # get the address of the RAM
        address = _LIB.Memory(self._env)
        # create a buffer from the contents of the address location
        buffer_ = ctypes.cast(address, ctypes.POINTER(RAM_VECTOR)).contents
        # create a NumPy array from the buffer
        arr = np.frombuffer(buffer_, dtype='uint8')
        return arr

    def _controller_buffer(self, port):
        """
        Find the pointer to a controller and setup a NumPy buffer.
        """
        # get the address of the controller
        address = _LIB.Controller(self._env, port)
        # create a memory buffer using the ctypes pointer for this vector
        buffer_ = ctypes.cast(address, ctypes.POINTER(CONTROLLER_VECTOR)).contents
        # create a NumPy buffer from the binary data and return it
        return np.frombuffer(buffer_, dtype='uint8')

    def _frame_advance(self, action):
        """Advance a frame in the emulator with an action."""
        # set the action on the controller
        self.controllers[0][:] = action
        # perform a step on the emulator
        _LIB.Step(self._env)

    def _backup(self):
        """Backup the NES state in the emulator."""
        _LIB.Backup(self._env)
        self._has_backup = True
        _dbg("_backup(): state saved")

    def _restore(self):
        """Restore the backup state into the NES emulator."""
        _LIB.Restore(self._env)
        _dbg("_restore(): state restored")

    def _will_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        pass

    def seed(self, seed=None):
        """Set the seed for this environment's RNG."""
        if seed is None:
            return []
        self.np_random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None, return_info=None):
        """
        Reset the state of the environment and returns an initial observation.
        """
        # Set the seed.
        self.seed(seed)
        # call the before reset callback
        self._will_reset()
        # reset the emulator
        if self._has_backup:
            self._restore()
        else:
            _LIB.Reset(self._env)
        # call the after reset callback
        self._did_reset()
        # set the done flag to false
        self.done = False
        # return the screen from the emulator
        state = self.screen
        if _DEBUG:
            # small stats about the observation buffer at reset
            _dbg("reset(): obs",
                 {"shape": state.shape, "dtype": str(state.dtype),
                  "min": int(state.min()), "max": int(state.max()),
                  "ptr": hex(state.ctypes.data)})
            _dbg("reset(): ram",
                 {"len": int(self.ram.size), "min": int(self.ram.min()),
                  "max": int(self.ram.max())})
        return state

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        pass

    def step(self, action):
        """
        Run one frame of the NES and return the relevant observation data.
        """
        if self.done:
            raise ValueError('cannot step in a done environment! call `reset`')
        # set the action on the controller
        self.controllers[0][:] = action
        # pass the action to the emulator as an unsigned byte
        _LIB.Step(self._env)
        # get the reward for this step
        reward = float(self._get_reward())
        # get the done flag for this step
        self.done = bool(self._get_done())
        # get the info for this step
        info = self._get_info()
        # call the after step callback
        self._did_step(self.done)
        # bound the reward
        if reward < self.reward_range[0]:
            reward = self.reward_range[0]
        elif reward > self.reward_range[1]:
            reward = self.reward_range[1]

        state = self.screen
        # Debug the first few steps to understand observations
        self._dbg_step_count += 1
        if _DEBUG and self._dbg_step_count <= _DEBUG_STEPS:
            _dbg(f"step({self._dbg_step_count}): "
                 f"obs.shape={state.shape} dtype={state.dtype} "
                 f"min={int(state.min())} max={int(state.max())} "
                 f"reward={reward} done={self.done}")
        return state, reward, self.done, info

    def _get_reward(self):
        """Return the reward after a step occurs."""
        return 0

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        return False

    def _get_info(self):
        """Return the info after a step occurs."""
        return {}

    def _did_step(self, done):
        """Handle any RAM hacking after a step occurs."""
        pass

    def close(self):
        """Close the environment."""
        if self._env is None:
            raise ValueError('env has already been closed.')
        _LIB.Close(self._env)
        self._env = None
        if self.viewer is not None:
            self.viewer.close()

    def render(self, mode='human'):
        """
        Render the environment.
        """
        if mode == 'human':
            if self.viewer is None:
                if self.spec is None:
                    caption = self._rom_path.split('/')[-1]
                else:
                    caption = self.spec.id
                self.viewer = ImageViewer(
                    caption=caption,
                    height=SCREEN_HEIGHT,
                    width=SCREEN_WIDTH,
                )
            self.viewer.show(self.screen)
        elif mode == 'rgb_array':
            return self.screen
        else:
            render_modes = [repr(x) for x in self.metadata['render.modes']]
            msg = 'valid render modes are: {}'.format(', '.join(render_modes))
            raise NotImplementedError(msg)

    def get_keys_to_action(self):
        """Return the dictionary of keyboard keys to actions."""
        buttons = np.array([
            ord('d'),  # right
            ord('a'),  # left
            ord('s'),  # down
            ord('w'),  # up
            ord('\r'), # start
            ord(' '),  # select
            ord('p'),  # B
            ord('o'),  # A
        ])
        keys_to_action = {}
        values = 8 * [[0, 1]]
        for combination in itertools.product(*values):
            byte = int(''.join(map(str, combination)), 2)
            pressed = buttons[list(map(bool, combination))]
            keys_to_action[tuple(sorted(pressed))] = byte
        return keys_to_action

    def get_action_meanings(self):
        """Return a list of actions meanings."""
        return ['NOOP']

# explicitly define the outward facing API of this module
__all__ = [NESEnv.__name__]
