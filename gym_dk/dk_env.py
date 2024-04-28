"""An OpenAI Gym environment for Super Mario Bros. and Lost Levels."""

from collections import defaultdict
from nes_py import NESEnv
import numpy as np
from ._roms import rom_path

# dictionary mapping value of status register to string names
# $01=Horizontal, $02=Vertical, $04=Jump, $08=Falling, $0A=Has Mallet, $FF=Dead
_STATUS_MAP = defaultdict(
    lambda: "Unknown",
    {
        0x01: "Walking",
        0x02: "Climbing",
        0x04: "Jumping",
        0x08: "Falling",
        0x0A: "Has Mallet",
        0xFF: "Dead",
    },
)

# a set of state values indicating that Mario is "busy"
_BUSY_STATES = [0x01, 0x02, 0x04, 0x08, 0x0A]


class DonkeyKongEnv(NESEnv):
    """An environment for playing Donkey Kong with Gymnasium."""

    # the legal range of rewards for each step, -infinite to infinite
    reward_range = (-np.inf, np.inf)
    ladders = set()

    def __init__(self, target=None):
        rom = rom_path()
        # initialize the super object with the ROM path
        super(DonkeyKongEnv, self).__init__(rom)
        # self._time_last = 0
        # self._y_position_last = 41
        # self._max_y = 0
        self._platform_last = 1
        self._last_distance_to_princess = 500
        # self._explored_coords: set = set()
        # self._last_grounded_y = 41
        self._platform_start: set = set()
        self._actions = []
        # self.num_explored_coords = 0
        # self._reward_counter = 0
        self._first_y = 0
        # self._rewards_averages = []
        # self._rewards_counter = 0
        self._barrel_states = []
        self._hammer_position = []
        self._known_ladders = set()
        with open("ladders.csv", "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(",")
                line2 = line[2]
                if line2 == "True":
                    line2bool = True
                else:
                    line2bool = False
                self._known_ladders.add((int(line[0]), int(line[1]), line2bool))

        self._last_player_position = [0, 0]
        self._last_action = None
        # self._time_on_platforms = []
        self.reset()
        self._skip_start_screen()
        self._backup()

    # MARK: Memory access
    def _read_mem_range(self, address, length):
        return int("".join(map(str, self.ram[address : address + length])))

    """def check_if_explored(self, x, y):
        for coord in self._explored_coords:
            if abs(coord[0] - x) <= 5 and abs(coord[1] - y) <= 5:
                current_count = coord[2]
                self._explored_coords.remove(coord)
                self._explored_coords.add((x, y, current_count + 1))
                return True
        # add new coord to explored coords
        self._explored_coords.add((x, y, 1))
        self.num_explored_coords += 1
        return False"""

    """def get_count_visited_coords(self, x , y):
        for coord in self._explored_coords:
            if coord[0] == x and coord[1] == y:
                return coord[2]
        return 0"""

    """def check_broken_ladder(self, player_position, last_player_position, last_player_status, last_player_movement):        
        if self._known_ladders == None:
            self._known_ladders = set()
        
        if last_player_status == 'Climbing' and last_player_movement == 'Up':
            if last_player_position[1] - player_position[1] > 0:
                for ladder in self._known_ladders:
                    if ladder[0] == self._current_platform and ladder[1] == player_position[0] and not ladder[2]:
                        temp_ladder = ladder
                        self._known_ladders.remove(ladder)
                        self._known_ladders.add((temp_ladder[0], temp_ladder[1], True))
                        return True
        return False"""

    def log_ladder(self, ladder):
        # look at ladders.csv and see if ladder is already there
        with open("ladders.csv", "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(",")
                if int(line[0]) == ladder[0] and int(line[1]) == ladder[1]:
                    return
        # log ladder to ladders.csv
        with open("ladders.csv", "a") as f:
            f.write(f"{ladder[0]}, {ladder[1]}, {ladder[2]}\n")
        self._known_ladders.add(
            (
                ladder[0],
                ladder[1],
                ladder[2],
            )
        )

    def mark_ladder(self, player_position):
        if self._known_ladders == None:
            self._known_ladders = set()
        if self._player_status == "Climbing":
            for ladder in self._known_ladders:
                if (
                    ladder[0] == self._current_platform
                    or ladder[0] == self._current_platform - 1
                ):
                    if ladder[1] == player_position[0]:
                        return
            if self._current_platform > 0 and player_position[0] > 0:
                self.log_ladder(
                    (
                        self._current_platform,
                        player_position[0],
                        False,
                    )
                )

    def mark_successful_ladder(self, player_position):
        if self._platform_last != self._current_platform:
            for ladder in self._known_ladders:
                if ladder[0] == self._platform_last:
                    if ladder[1] == player_position[0]:
                        temp_ladder = ladder
                        self._known_ladders.remove(ladder)
                        self._known_ladders.add(
                            (temp_ladder[0], temp_ladder[1], False, True)
                        )
                        return
        return

    @property
    def _p1_score(self):
        # score is represented as a figure with 6 10's places
        return self._read_mem_range(0x0025, 6)

    @property
    def _time(self):
        # time is represented as a figure with 4 10's places
        return self._read_mem_range(0x002E, 4)

    @property
    def _p1_lives(self):
        """Return the number of remaining lives."""
        return self.ram[0x0404]

    @property
    def _princess_position(self):
        OAM_X = self.ram[0x0203]
        OAM_Y = self.ram[0x0200]

        Princess_OAM_Slot = 58
        # OAM_X is the base address for the OAM X coordinates
        Princess_OAM_X = OAM_X + (4 * Princess_OAM_Slot)
        # OAM_Y is the base address for the OAM Y coordinates
        Princess_OAM_Y = OAM_Y + (4 * Princess_OAM_Slot)
        if Princess_OAM_Y == 255:
            return (0, 0)
        return [270, 420]

    @property
    def player_position(self) -> list[int]:
        OAM_X = self.ram[0x0203]
        Jumpman_OAM_Slot = 0
        # OAM_X is the base address for the OAM X coordinates
        Jumpman_OAM_X = OAM_X + (4 * Jumpman_OAM_Slot)
        OAM_Y = self.ram[0x0200]
        # invert the y pixel into the distance from the bottom of the screen
        Jumpman_OAM_Y = OAM_Y + (4 * Jumpman_OAM_Slot)
        if Jumpman_OAM_Y == 255:
            return [0, 0]
        return [Jumpman_OAM_X, 240 - Jumpman_OAM_Y]

    @property
    def _fire_position(self):
        if self.ram[0xAD] == 0:
            return 0
        else:
            OAM_slot = 4
            OAM_x = self.ram[0x0203] + (4 * OAM_slot)
            OAM_y = 240 - self.ram[0x0200] + (4 * OAM_slot)
            if OAM_y < 0:
                return 0
            return [OAM_x, OAM_y]

    @property
    def _get_barrel_states(self):
        barrel_states = []
        for i in range(0, 4):
            barrel_x = self.ram[0x5D + (i * 9)]
            barrel_y = self.ram[0x5D + (i * 9) + 1]
            barrel_state = self.ram[0x5D + (i * 9) + 2]
            barrel_platform = self.ram[0x5D + (i * 9) + 3]
            barrel_gfx_frame = self.ram[0x5D + (i * 9) + 4]
            barrel_shift_down_flag = self.ram[0x5D + (i * 9) + 5]
            barrel_states.append(
                (
                    barrel_x,
                    barrel_y,
                    barrel_state,
                    barrel_platform,
                    barrel_gfx_frame,
                    barrel_shift_down_flag,
                )
            )
        return barrel_states

    @property
    def _hitbox(self):
        hitbox_a_x_left = self.ram[0x46]
        hitbox_a_y_top = self.ram[0x47]
        hitbox_a_x_right = self.ram[0x48]
        hitbox_a_y_bottom = self.ram[0x49]
        return [hitbox_a_x_left, hitbox_a_y_top, hitbox_a_x_right, hitbox_a_y_bottom]

    @property
    def _get_hammer_position(self):
        if self.ram[0xAD] == 0:
            return 0
        else:
            OAM_slot = 52
            OAM_x = self.ram[0x0203] + (4 * OAM_slot)
            OAM_y = 240 - self.ram[0x0200] + (4 * OAM_slot)
            if OAM_y < 0:
                return 0

            return [OAM_x, OAM_y]

    @property
    def entities_per_platform(self):
        entities_per_platform = []
        for i in range(0, 7):
            entities_per_platform.append(self.ram[0x7E + i])
        return entities_per_platform

    @property
    def _hitbox_x_distance(self):
        return self.ram[0x9C]

    @property
    def _hitbox_y_distance(self):
        return self.ram[0x9D]

    @property
    def _barrel_toss_position(self):
        barrel_x = self.ram[0x4D]
        barrel_y = self.ram[0x32]
        return [barrel_x, barrel_y]

    @property
    def _jumpman_on_platform_flag(self):
        return self.ram[0x5A]

    @property
    def _jumpman_climb_on_plat_anim_counter(self):
        return self.ram[0x5B]

    @property
    def _jumpman_climb_anim_counter(self):
        return self.ram[0x5C]

    @property
    def _platform_sprites(self):
        platform_sprites = []
        for i in range(0, 6):
            platform_tile = self.ram[0xA0 + (i * 2)]
            platform_prop = self.ram[0xA0 + (i * 2) + 1]
            platform_sprites.append((platform_tile, platform_prop))
        return platform_sprites

    @property
    def _fire_barrel_position(self):
        flame_x = self.ram[0x20]
        flame_y = self.ram[0xC0]
        flame_gfx_frame_frame1 = self.ram[0xFC]
        flame_gfx_frame_frame2 = self.ram[0xFE]
        return [flame_x, flame_y]

    @property
    def _entity_spawn_timer(self):
        return self.ram[0x36]

    @property
    def _flame_enemy_move_dir_update(self):
        return self.ram[0x3B]

    @property
    def _flame_state(self):
        flame_state = self.ram[0xAE]
        return flame_state

    @property
    def _flame_direction(self):
        flame_direction = self.ram[0xB3]
        return flame_direction

    @property
    def _flame_platform(self):
        return self.ram[0xE0]

    @property
    def _flame_follow_direction(self):
        return self.ram[0xEC]

    @property
    def _get_barrel_positions(self):
        barrel_positions = []

        return barrel_positions

    @property
    def _check_platform_start(self):
        time = self._time
        for platform in self._platform_start:
            if platform[0] == self._current_platform:
                return platform[1], platform[2]
        self._platform_start.add(
            (self._current_platform, self.player_position[0], 0, time)
        )
        return self.player_position[0]

    @property
    def _current_platform(self):
        # check what platform the player is on
        platform = self.ram[0x0059]
        platform_int = 1

        # convert scalar to 1 digit integer
        if platform == 0x01:
            platform_int = 1
        elif platform == 0x02:
            platform_int = 2
        elif platform == 0x03:
            platform_int = 3
        elif platform == 0x04:
            platform_int = 4
        elif platform == 0x05:
            platform_int = 5
        return platform_int

    @property
    def _player_status(self):
        return _STATUS_MAP[self.ram[0x0096]]

    @property
    def _player_state(self):
        """
        Return the current player state.

        Note:
            $01=Hor, $02=Ver, $04=Jump, $08=Falling, $0A=Has Mallet, $FF=Dead
        """
        return self.ram[0x0096]

    @property
    def _is_dead(self):
        return self._player_state == 0xFF

    @property
    def _is_grounded(self):
        """Return True if Mario is grounded, False otherwise."""
        return (
            self._player_state == 0x01 or self._player_state == 0x02
        ) and not self._is_dead

    @property
    def _is_jumping(self):
        """Return True if Mario is jumping and not dead, False otherwise."""
        return self._player_state == 0x04 and not self._is_dead

    @property
    def _grounded_y(self):
        if not self._is_jumping and self.player_position[1] > 0 and not self._is_dead:
            return self.player_position[1]
        return self._last_grounded_y

    @property
    def _is_game_over(self):
        """Return True if the game has ended, False otherwise."""
        return self._p1_lives == 1

    @property
    def _is_busy(self):
        return self._player_state in _BUSY_STATES

    @property
    def _is_stage_over(self):
        """Return a boolean determining if reach princess"""
        if self._distance_to_princess[0] < 5 and self._distance_to_princess[1] < 5:
            return True

        return False

    def _runout_prelevel_timer(self):
        """Force the pre-level timer to 0 to skip frames during a death."""
        self.ram[0x07A0] = 0

    def _skip_start_screen(self):
        """Press and release start to skip the start screen."""
        # press and release the start button
        self._frame_advance(8)
        self._frame_advance(0)
        # Press start until the game starts
        while self._time == 0:
            # press and release the start button
            self._frame_advance(8)
            self._frame_advance(0)
            # run-out the prelevel timer to skip the animation
            self._runout_prelevel_timer()
        # set the last time to now
        self._time_last = self._time
        # after the start screen idle to skip some extra frames
        while self._time >= self._time_last:
            self._time_last = self._time
            self._frame_advance(8)
            self._frame_advance(0)

    def _kill_mario(self):
        """Skip a death animation by forcing Mario to death."""
        # force Mario's state to dead
        self.ram[0x0406] = 0x01
        # step forward one frame
        self._frame_advance(0)

    # MARK: Reward Functions

    '''@property
    def _reward_at_ladder(self):
        """Return the reward earned by climbing a ladder."""
        if self._known_ladders == None:
            self._known_ladders = set()
        for ladder in self._known_ladders:
            if ladder[0] == self._current_platform:
                if ladder[1] == self.player_position[0] and not ladder[2]:
                    return 0.1
                if ladder[1] == self.player_position[0] and ladder[2]:
                    return -1     
        return -0.0001'''

    @property
    def _reward_closer_to_ladder(self):
        if self.player_position[0] == 0 and self.player_position[1] == 0:
            return 0

        if self._known_ladders == None:
            self._known_ladders = set()

        closest_ladder = None
        for ladder in self._known_ladders:
            if ladder[0] == self._current_platform and not ladder[2]:
                distance = abs(ladder[1] - self.player_position[0])
                if closest_ladder == None:
                    closest_ladder = ladder
                elif distance < abs(closest_ladder[1] - self.player_position[0]):
                    closest_ladder = ladder
        if closest_ladder != None:
            distance = abs(closest_ladder[1] - self.player_position[0])
            # check if distance is a number
            if distance == 0:
                return 0.1
            return 0.1 / distance
        return 0

    @property
    def _punish_down_ladder(self):
        if self.player_position[0] == 0 and self.player_position[1] == 0:
            return 0
        if self._player_status == "Climbing" and self._last_action == "Down":
            return -1
        return 0

    '''@property
    def _reward_platform_travel(self):
        """Return the reward earned by traveling on a platform."""
        platform_start = self._check_platform_start

        for platform in self._platform_start:
            if platform[0] == self._current_platform:
                temp_platform = platform
                self._platform_start.remove(platform)
                self._platform_start.add((temp_platform[0], temp_platform[1], abs(self.player_position[0] - platform[1])))
        total_platform_travel = 0
        for platform in self._platform_start:
            total_platform_travel += platform[2]
        return total_platform_travel / 100000'''

    '''@property
    def _reward_exploration(self):
        """Return the reward earned by exploring."""
        if self._get_info()['player_position'][1] > 0:
            if not self.check_if_explored(self._get_info()['player_position'][0], self._get_info()['player_position'][1]):
                return 0.0001 * self.num_explored_coords 
            else:
                # return a value between 0 and 1 based on the number of explored coords, decreasing as the number of explored coords increases
                return 0.0001 * self.num_explored_coords / self.get_count_visited_coords(self._get_info()['player_position'][0], self._get_info()['player_position'][1])
        return 0'''

    """@property
    def _reward_grounded_y(self):
        return (self._grounded_y - 41) / 100"""

    @property
    def _y_reward(self):
        if self._first_y == 0:
            self._first_y = self.player_position[1]
        """Return the reward based on up down movement between steps."""
        if self._is_dead:
            return 0
        if self._y_position_last == 0:
            self._y_position_last = self.player_position[1]

        if self.player_position:
            _reward = self.player_position[1] - self._first_y
            self._y_position_last = self.player_position[1]
            return _reward / 1000
        return 0

    """    
    @property
    def _new_max_y_reward(self):        
        if self._get_info()['player_position'][1] > self._max_y:
            self._max_y = self._get_info()['player_position'][1]
            return 0.1
        return 0
    """

    @property
    def _platform_reward(self):
        if self.player_position[0] == 0 and self.player_position[1] == 0:
            return 0
        """Return the reward based on the platform the player is on."""
        if self._current_platform < 5:
            reward = self._current_platform
        else:
            reward = 1
        return reward

    @property
    def _death_penalty(self):
        """Return the reward earned by dying."""
        if self._is_dead:
            return -0.1

        return 0

    @property
    def _reward_grounded(self):
        if self.player_position[0] == 0 and self.player_position[1] == 0:
            return 0
        """Return the reward earned by being grounded."""
        if self._is_grounded:
            return 0.1
        return 0

    @property
    def _climbing_reward(self):
        """Return the reward earned by climbing."""
        if self._player_status == "Climbing":
            return 0.01

        return 0

    @property
    def _score_reward(self):
        if self.player_position[0] == 0 and self.player_position[1] == 0:
            return 0
        """Return the reward earned by scoring."""
        return self._p1_score / 10000

    """@property
    def reward_y_improvement(self):
        reward = 0
        current_platform = self._current_platform
        current_ground_y = self._grounded_y
        if current_ground_y != self._last_grounded_y:
            reward = (current_ground_y - 41) * current_platform 
        self._last_grounded_y = current_ground_y
        return reward"""

    """@property
    def _time_reward(self):
        return (80000 - self._time) / 1000000
    """
    """
    @property
    def _distance_to_flame(self):
        if self._fire_position == 0:
            return np.inf
        flame_x, flame_y = self._fire_position
        player_x, player_y = self.player_position
        distance = np.sqrt((player_x - flame_x)**2 + (player_y - flame_y)**2)
        return distance"""

    @property
    def _distance_to_princess(self) -> list[float]:
        if self.player_position[0] == 0 and self.player_position[1] == 0:
            return [np.inf, np.inf]
        princess_y, princess_x = self._princess_position
        player_y, player_x = self.player_position
        if player_x == 0 or player_y == 0:
            return [np.inf, np.inf]
        distance_x = abs(player_x - princess_x)
        distance_y = abs(player_y - princess_y)

        return [distance_x, distance_y]

    @property
    def _reward_closer_to_princess(self):
        if self.player_position[0] == 0 and self.player_position[1] == 0:
            return 0
        distance_x = self._distance_to_princess[0]
        distance_y = self._distance_to_princess[1]
        actual_distance = np.sqrt(distance_x**2 + distance_y**2)

        return 1 / actual_distance + 4 / distance_y

    """@property
    def _time_on_platform_punishment(self):
        time = self._time
        for platform in self._platform_start:
            if platform[0] == self._current_platform:
                return -1000/(time - platform[2])
        return 0"""

    """@property
    def _punish_staying_still(self):
        if self.player_position[0] == 0 and self.player_position[1] == 0:
            return 0
        if self._last_action == 'No Action':
            return -0.1
        if self._last_player_position == self.player_position:
            return -0.1
        return 0"""

    """@property
    def _punish_at_broken_ladder(self):
        if self.player_position[0] == 0 and self.player_position[1] == 0:
            return 0
        if self._known_ladders == None:
            self._known_ladders = set()
        for ladders in self._known_ladders:
            if self._current_platform == ladders[0] and self.player_position[0] == ladders[1] and ladders[2] and self._player_status == 'Climbing':
                return -1
        return 0"""
    """
    @property
    def _reward_safety(self):
        if self._fire_position == 0:
            return 0
        
        fire_x, fire_y = self._fire_position
        player_x, player_y = self.player_position
        distance = np.sqrt((player_x - fire_x)**2 + (player_y - fire_y)**2)

        if distance < 10:
            return -1/distance   
        return 0
    """
    # MARK: nes-py API calls

    def _will_reset(self):
        """Handle and RAM hacking before a reset occurs."""
        self._time_last = 0
        self._x_position_last = 0
        self._y_position_last = 0
        self._last_grounded_y = 41

    def step(self, action):
        """Log the action taken by the agent."""
        self._last_action = action

        if action == 0:
            self._last_action = "No Action"
        elif action == 128:
            self._last_action = "Right"
        elif action == 64:
            self._last_action = "Left"
        elif action == 32:
            self._last_action = "Down"
        elif action == 1:
            self._last_action = "A"
        elif action == 129:
            self._last_action = "Right A"
        elif action == 65:
            self._last_action = "Left A"
        elif action == 17:
            self._last_action = "Up A"
        elif action == 33:
            self._last_action = "Down A"
        elif action == 16:
            self._last_action = "Up"

        self._last_player_status = self._player_status

        self.mark_ladder(self.player_position)
        self._last_player_position = self.player_position

        return super(DonkeyKongEnv, self).step(action)

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        self._time_last = self._time
        self._x_position_last = self.player_position[0]

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """

        # if done flag is set a reset is incoming anyway, ignore any hacking
        if done:
            return

    """def average_rewards(self, death_penalty, platform_reward, climbing_reward, score_reward, y_reward, reward_platform_travel, time_reward, total_reward, new_max_y_reward):
        # append the rewards to the rewards log
        
        if self._rewards_counter ==0:
            self._rewards_averages = [death_penalty, platform_reward, climbing_reward, score_reward, y_reward, reward_platform_travel, time_reward, total_reward, new_max_y_reward]
            self._rewards_counter += 1
            return
        self._rewards_counter += 1
        current_averages = self._rewards_averages
        self._rewards_averages = [
            (current_averages[0] + death_penalty) / self._rewards_counter,
            (current_averages[1] + platform_reward) / self._rewards_counter,
            (current_averages[2] + climbing_reward) / self._rewards_counter,
            (current_averages[3] + score_reward) / self._rewards_counter,
            (current_averages[4] + y_reward) / self._rewards_counter,
            (current_averages[5] + reward_platform_travel) / self._rewards_counter,
            (current_averages[7] + time_reward) / self._rewards_counter,
            (current_averages[8] + total_reward) / self._rewards_counter,
            (current_averages[6] + new_max_y_reward) / self._rewards_counter,
        ]"""

    def log_reward(self, reward):
        with open("rewards_map.csv", "a") as f:
            f.write(f"{self.player_position[0]}, {self.player_position[1]}, {reward}\n")

    def _get_reward(self):
        """Return the reward after a step occurs."""
        # flame = self._fire_position
        # safety_reward = self._reward_safety
        # score_reward = self._score_reward
        # reward_exploration = self._reward_exploration
        # new_max_y_reward = self._new_max_y_reward
        y_reward = self._y_reward
        y_reward = self._y_reward
        # reward_platform_travel = self._reward_platform_travel
        # time_reward = self._time_reward
        # grounded_y_reward = self._reward_grounded_y
        # punish_down_ladder = self._punish_down_ladder
        # reward_at_ladder = self._reward_at_ladder
        # time_on_platform_punishment = self._time_on_platform_punishment
        # y_improvement_reward = self.reward_y_improvement
        # climbing_reward = self._climbing_reward
        # punish_broken_ladder = self._punish_at_broken_ladder
        # climbing_reward = self._climbing_reward
        # punish_broken_ladder = self._punish_at_broken_ladder
        death_penalty = self._death_penalty
        platform_reward = self._platform_reward
        grounded_reward = self._reward_grounded
        princess_reward = self._reward_closer_to_princess
        # ladder_distance_reward = (self._reward_closer_to_ladder + platform_reward)/10

        rewards = princess_reward + y_reward + grounded_reward
        punishments = death_penalty
        total_reward = (platform_reward * rewards) + punishments
        # ladder_distance_reward = (self._reward_closer_to_ladder + platform_reward)/10

        rewards = princess_reward + y_reward + grounded_reward
        punishments = death_penalty
        total_reward = (platform_reward * rewards) + punishments

        return total_reward

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        return self._is_game_over

    def _get_info(self) -> dict:
        """Return the info after a step occurs"""
        return dict(
            lives=self._p1_lives,
            score=self._p1_score,
            platform=self._current_platform,
            time=self._time,
            player_position=self.player_position,
            princess_pos=self._princess_position,
            distance_to_princess=self._distance_to_princess,
            fire_position=self._fire_position,
            player_state=self._player_state,
            player_status=self._player_status,
            is_dead=self._is_dead,
            is_stage_over=self._is_stage_over,
            is_busy=self._is_busy,
            is_game_over=self._is_game_over,
            is_climbing=self._player_status == "Climbing",
            is_jumping=self._player_status == "Jumping",
            is_walking=self._player_status == "Walking",
            is_falling=self._player_status == "Falling",
            is_has_mallet=self._player_status == "Has Mallet",
            is_vertical=self._player_status == "Climbing",
            is_horizontal=self._player_status == "Walking",
            is_grounded=self._is_grounded,
            barrel_states=self._get_barrel_states,
            hammer_position=self._get_hammer_position,
            flame_state=self._flame_state,
            flame_direction=self._flame_direction,
            current_platform=self._current_platform,
            platform_reward=self._platform_reward,
            death_penalty=self._death_penalty,
            entities_per_platform=self.entities_per_platform,
            hitbox_x_distance=self._hitbox_x_distance,
            hitbox_y_distance=self._hitbox_y_distance,
            flame_platform=self._flame_platform,
            flame_follow_direction=self._flame_follow_direction,
            barrel_toss_position=self._barrel_toss_position,
            platform_sprites=self._platform_sprites,
            fire_barrel_position=self._fire_barrel_position,
            entity_spawn_timer=self._entity_spawn_timer,
            flame_enemy_move_dir_update=self._flame_enemy_move_dir_update,
            hit_box=self._hitbox,
            jumpman_on_platform_flag=self._jumpman_on_platform_flag,
            jumpman_climb_on_plat_anim_counter=self._jumpman_climb_on_plat_anim_counter,
            jumpman_climb_anim_counter=self._jumpman_climb_anim_counter,
            check_platform_start=self._check_platform_start,
            punish_down_ladder=self._punish_down_ladder,
            reward_closer_to_princess=self._reward_closer_to_princess,
            reward_grounded=self._reward_grounded,
            known_ladders=self._known_ladders,
            reward_closer_to_ladder=self._reward_closer_to_ladder,
            last_action=self._last_action,
            last_player_position=self._last_player_position,
            last_player_status=self._last_player_status,
            last_grounded_y=self._last_grounded_y,
            platform_start=self._platform_start,
            actions=self._actions,
            platform_last=self._platform_last,
            last_distance_to_princess=self._last_distance_to_princess,
        )
