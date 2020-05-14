from gym_minigrid.minigrid import *
from gym_minigrid.register import register

# Enumeration of possible actions
class FoodWorldActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2

    # Pick up an object
    pickup = 3


class FoodEnv(MiniGridEnv):
    """
    Environment in which the agent has to acquire enough nutritional resources to avoid
    starvation
    """

    def __init__(
        self,
        m=8,
        n=4,
        max_steps=None,
        min_nutrients=300
    ):
        self.numObjs = n
        self.target_protein = min_nutrients
        if max_steps == None:
            max_steps = m*2
        super().__init__(
            grid_size=m,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            actions=FoodWorldActions
        )



    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

        self.target_nutrient = 300.
        self.nutrient_accumulated = 0.

        objs = []

        # For each object to be generated
        while len(objs) < self.numObjs:
            objColor = self._rand_elem(COLOR_NAMES)

            obj = Ball(objColor)

            self.place_obj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        target = objs[self._rand_int(0, len(objs))]
        self.targetColor = target.color

        self.mission = 'acquire enough nutrients'

    def get_mission(self):
        return self.mission + ' ' + str(self.nutrient_accumulated)

    def grid_step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        if self.step_count >= self.max_steps:
            done = True
        obs = self.gen_obs()
        return obs, reward, done, {}

    def step(self, action):
        obs, _, done, info = self.grid_step(action)
        reward = 0
        if self.carrying:
            objColor = self.carrying.color
            self.nutrient_accumulated += 100.
            self.carrying = None

        if done:
            reward = self.nutrient_accumulated / self.target_nutrient
            reward = min(reward, 1.)
        return obs, reward, done, info



register(
    id='MiniGrid-FoodWorld-16x16-N3-v0',
    entry_point='gym_minigrid.envs:FoodEnv'
)
