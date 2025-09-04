import numpy as np
import pygame
import sys
from collections import OrderedDict
from src.environments.values_and_norms_bridge import Values, Norms, Instrumental
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.environments.bridge_person import Person_Moving, Person_Static, Status, Drowning_TimeFixed
import random

class Bridge_Map():
    def __init__(self, num_bridges,  width, height, dangerous_spots):
        self.directions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
        self.width = width  
        self.height = height
        self.dangerous_spots = dangerous_spots
        self.land_lines = 2
        self.bridge_lines = self.height - self.land_lines*2
        self.grid_types = {
            "water": 0,
            "land": 1
        }
        self.bridges = {}
        self.bridge_coordinates = [] 
        self.grid_list = []
        self.grid_array = []
        self.generate_map(num_bridges)
        self.land_tiles = []
        self.get_land_tiles()

        self.NOT_ON_MAP = [0,height]

    """
    generate list of land tiles
    """
    def get_land_tiles(self):
        for y, row in enumerate(self.grid_list):  
            for x, tile in enumerate(row):  
                if tile == self.grid_types["land"]:  
                    self.land_tiles.append((x, y))

    """
    calculates position of bridges
    """
    def calculate_bridge_positions(self, num_bridges):
        center = self.width // 2  
        
        if num_bridges == 1:
            return [center]  

        positions = []
        
        # odd number of bridges: one bridge in the middle
        if num_bridges % 2 == 1:
            positions.append(center)  
            num_bridges -= 1  

        # place bridges right and left of center
        if num_bridges == 2 or num_bridges == 3:
            positions.append(center - (center-1))  
            positions.append(center + (center-1))  

        return sorted(positions)  

    """
    generates map layout with land lines, water lines and bridges
    """
    def generate_map(self, num_bridges):
        # generates a row of land tiles
        vertical_land_line = [[self.grid_types["land"]] * self.width]  
        bridge_positions = self.calculate_bridge_positions(num_bridges)

        # generates tiles of water with bridges at the calculated positions
        middle_section = []
        self.bridge_coordinates = [] 

        for row in range(self.bridge_lines):  
            middle_row = []
            for x in range(self.width):
                if x in bridge_positions:
                    middle_row.append(self.grid_types["land"])
                    if x not in self.bridges:
                        self.bridges[x]=[]
                    self.bridges[x].append([x,row+self.land_lines])
                    self.bridge_coordinates.append([x,row+self.land_lines])  
                else:
                    middle_row.append(self.grid_types["water"])
            middle_section.append(middle_row)

        self.grid_list = vertical_land_line + vertical_land_line + middle_section + vertical_land_line + vertical_land_line 
        self.grid_array = np.array(self.grid_list)

    """
    returns grid type for coordinates
    """

    def get_grid_type(self, coordinates):
        if np.equal(coordinates, self.NOT_ON_MAP).all():
            grid_type = None
        else:
            grid_type = self.grid_list[coordinates[1]][coordinates[0]]
        return grid_type

    """
    checks if coordinates lie within the boundaries of the map
    """
    def location_in_grid(self, coordinates):
            if 0 <= coordinates[0] and coordinates[0]<self.width and 0 <= coordinates[1] and coordinates[1] <self.height:
                return True
            return False
        
    """
    checks which tiles lie next to the coordinates
    """
    def get_adjacent_tiles(self, coordinates):
        adjacent_tiles = []
        #check if tiles lie in the grid 
        for direction in self.directions:
            adjacent_tile = coordinates+direction
            if self.location_in_grid(adjacent_tile):
                adjacent_tiles.append(adjacent_tile)

        return adjacent_tiles
    
    """
    checks if coordinates lie next to a tile type
    """
    def next_to(self, coordinates, type):
        adjacent_tiles = self.get_adjacent_tiles(coordinates)
        for adjacent_tile in adjacent_tiles:
            if self.get_grid_type(adjacent_tile) == self.grid_types[type]:
                return True
        return False
    
    """
    checks if coordinates lie next to a land tiles
    """
    def next_to_land(self, coordinates):
        return self.next_to(coordinates, "land")
    
    """
    checks if coordinates lie next to water
    """
    def next_to_water(self, coordinates):
        return self.next_to(coordinates, "water")
    
    """
    checks if coordinates lie in water
    """
    def in_water(self, coordinates):
         if self.get_grid_type(coordinates) == self.grid_types["water"]:
             return True
         
    """
    checks if coordinates lie on briddge
    """    
    def position_on_bridge(self, coordinates):
        for _, bridge_tiles in self.bridges.items():
            for tile in bridge_tiles:
                if np.equal(coordinates, tile).all():
                    return True
        return False
    
    """
    checks if coordinates lie in front of a bridge
    """
    def in_front_of_bridge(self, coordinates):
        for _, bridge_tiles in self.bridges.items():
            spot_in_front_of_bridge =  (np.array(bridge_tiles[0]) + np.array([0,-1])).tolist()
            if np.equal(coordinates,spot_in_front_of_bridge).all():
                return True
        return False
    
    """
    checks if coordinates lie in behind a bridge
    """
    def behind_bridge(self, coordinates):
        for _, bridge_tiles in self.bridges.items():
            spot_behind_bridge = (np.array(bridge_tiles[-1]) + np.array([0,1])).tolist()
            if np.equal(coordinates,spot_behind_bridge).all():
                return True
        return False
    
    """
    returns trajcectory for persons that move across a bridge
    """
    def trajectory_across_bridge(self, bridge):
        #starting positon
        position = [1,self.height-1]
        trajectory = [position]

        #add path to bridge
        trajectory_to_bridge = []
        sorted_keys =  sorted(self.bridges.keys(), reverse=True)
        bridge_key = sorted_keys[bridge-1]
        first_tile = self.bridges[bridge_key][-1]
        while position[0] != first_tile[0]:
            position = [position[0]+1, position[1]]
            trajectory_to_bridge.append((position))
        position = [position[0],position[1]-1]
        trajectory_to_bridge.append(position)

        #add path for crossing the bridge
        trajectory += trajectory_to_bridge
        trajectory += self.bridges[bridge_key][::-1]
        position = trajectory[-1]
        position = [position[0],position[1]-1]
        trajectory.append(position)
        position = trajectory[-1]

        #add path for walking right till out of map
        while position[0] != self.width-1:
            position = [position[0]+1, position[1]]
            trajectory.append((position))

        return trajectory
    
    """
    returns trajcectory for persons that along the lower shore
    """
    
    def trajectory_land_line(self):
        #starting positon
        position = [0,self.height-2]
        trajectory = [position]

        while position[0] != self.width-1:
            position = [position[0]+1, position[1]]
            trajectory.append((position))

        position = [position[0], position[1]+1]
        trajectory.append((position))

        return trajectory

class Bridge(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 3} 

    def __init__(self,    
                #setting up the persons and their attributes
                ids_moving_persons, 
                pos_static_persons,
                drowning,
                drowning_time,
                #drowning_behavior, 
                slipping_prob, 
                pushed_off_bridge_prob,
                respawn_timer,
                #setting up map and rendering
                num_bridges,  
                width,
                height,
                dangerous_spots,
                target_location,
                reward_type = "instrumental",
                render_mode=None, 
                ):
        
        """
        setting parameters for person behavior
        """
        self.ids_moving_persons = ids_moving_persons
        self.pos_static_persons = pos_static_persons
        self.slipping_prob = slipping_prob
        self.pushed_off_bridge_prob = pushed_off_bridge_prob
        self.respawn_time = respawn_timer
        self.drowning = drowning
        self.drowning_time = drowning_time
        self.drowning_behavior = Drowning_TimeFixed(drowning, drowning_time)

        """
        setting map parameters 
        """
        self.num_bridges= num_bridges
        self.width = width
        self.height = height
        self.dangerous_spots = dangerous_spots
        self.reward_type = reward_type
        self._agent_location = np.array([0,0])
        self.render_mode=render_mode
        self.target_location = np.array(target_location)
        self.slipping_spots = []


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert num_bridges in [1, 2, 3], "Invalid number of bridges. Number of bridges must be 1,2 or 3."
        assert width % 2 != 0, "The width must be an odd number."
        if num_bridges == 3:
            assert width >= 7, "The width must be at least 7 for 3 bridges."
        elif num_bridges == 2:
            assert width >= 5, "The width must be at least 5 for 2 bridges."
        elif num_bridges == 1:
            assert width >= 3, "The width must be at least 3 for 1 bridge."
        assert all(dangerous_spot[0] < width and dangerous_spot[1] < height for dangerous_spot in dangerous_spots), "Dangerous spot out of map boundaries."
        assert all(pos[0] < width and pos[1] < height for pos in pos_static_persons), f"At least one postition (static persons) is out of map boundaries: {pos_static_persons}"
        self.bridge_map = Bridge_Map(num_bridges, width, height, dangerous_spots)
        assert all(any(self.bridge_map.next_to_water(tile) for tile in self.bridge_map.get_adjacent_tiles(dangerous_spot)) for dangerous_spot in dangerous_spots), "Each dangerous spot must be placed next to a water tile."
        
        """
        initialize persons on map 
        """
        self.persons = []
        self.initialize_persons(self.drowning_behavior)

        """
        the observation space:
        the observation is a one hot encoding of the agent position, the person positions and the water tiles flattened to a 1d vector (np.array)
        """
        self.size = self.bridge_map.width * self.bridge_map.height
        grid_width = self.bridge_map.width   # number of columns in the grid
        grid_height = self.bridge_map.height   # number of rows in the grid
        len_obs_dict = len(self.get_obs_dict())
        total_grid_cells = self.bridge_map.width  * self.bridge_map.height
        observation_size = len_obs_dict * total_grid_cells
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(observation_size,),
            dtype=np.float32
        )

        """cartesian coordinates for water_tiles"""
        self.water_tiles = [
            [x, y]  # cartesian coordinate of water tile
            for y in range(grid_height)
            for x in range(grid_width)
            if self.bridge_map.get_grid_type([x, y]) == self.bridge_map.grid_types["water"]
        ]

        """
        the action space (primtive actions the agent can take):
        0:  "right",
        1:  "down",
        2:  "left",
        3:  "up",
        4:  "pull out of water" 
        """
        self.action_space = spaces.Discrete(6)
        self.directions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]

        """
        mapping of primitive actions from `self.action_space` to
        the direction the agent will walk in if that action is taken.
        """
        self.action_to_direction = {
            0: self.directions[0], #right
            1: self.directions[1], #down
            2: self.directions[2], #left
            3: self.directions[3], #up
        }


        """
        setting parameters for rendering the environment
        """
        cell_width = 100  # width of each cell in pixels
        cell_height = 100  # height of each cell in pixels
        self.window_size_width = grid_width * cell_width  
        self.window_size_height = grid_height * cell_height
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def set_render_mode(self, render_mode):
        self.render_mode = render_mode

    def get_reward_type(self):
        return self.reward_type
    
    def set_reward_type(self, reward_type):
        self.reward_type = reward_type

    """
    initialize persons; sets trajectories of moving persons according to their ID
    """

    def initialize_persons(self, drowning_behavior):
        if 1 in self.ids_moving_persons:
            person_trajectory = self.bridge_map.trajectory_across_bridge(0)
            person = Person_Moving(person_id=1, position=person_trajectory[0], trajectory=person_trajectory, status=Status.NOT_IN_GAME, drowning_behavior=drowning_behavior, slipping_prob=self.slipping_prob, pushed_off_bridge_prob=self.pushed_off_bridge_prob, respawn_time=self.respawn_time)
            self.persons.append(person)
        if 2 in self.ids_moving_persons:
            person_trajectory = self.bridge_map.trajectory_across_bridge(1)
            person = Person_Moving(person_id=2, position=person_trajectory[0], trajectory=person_trajectory, status=Status.NOT_IN_GAME, drowning_behavior=drowning_behavior, slipping_prob=self.slipping_prob, pushed_off_bridge_prob=self.pushed_off_bridge_prob, respawn_time=self.respawn_time)
            self.persons.append(person)
        if 3 in self.ids_moving_persons:
            person_trajectory = self.bridge_map.trajectory_across_bridge(2)
            person = Person_Moving(person_id=3, position=person_trajectory[0], trajectory=person_trajectory, status=Status.NOT_IN_GAME, drowning_behavior=drowning_behavior, slipping_prob=self.slipping_prob, pushed_off_bridge_prob=self.pushed_off_bridge_prob, respawn_time=self.respawn_time)
            self.persons.append(person)
        if 4 in self.ids_moving_persons:
            person_trajectory = self.bridge_map.trajectory_land_line()
            person = Person_Moving(person_id=4, position=person_trajectory[0], trajectory=person_trajectory, status=Status.NOT_IN_GAME, drowning_behavior=drowning_behavior, slipping_prob=self.slipping_prob, pushed_off_bridge_prob=self.pushed_off_bridge_prob, respawn_time=self.respawn_time)
            self.persons.append(person)
        for person_pos in self.pos_static_persons:
            person = Person_Static(position=person_pos, status=Status.STANDING, drowning_behavior=drowning_behavior, slipping_prob=self.slipping_prob, pushed_off_bridge_prob=self.pushed_off_bridge_prob, respawn_time=self.respawn_time)
            self.persons.append(person)
            #person.respawn(bridge_map=self.bridge_map)

    """
    functions for returning information about the state 

    return of the observation function: the flattened and concatenated agent and person observations
    """
    def observation(self):
        obs_dict = self.get_obs_dict()

        # flatten 
        agent_flat = obs_dict["agent_window"].flatten()
        person_flat = obs_dict["person_window"].flatten()
        water_flat = obs_dict["water_window"].flatten()

        # concatenate into one 1D array
        nn_input = np.concatenate([agent_flat, person_flat, water_flat])

        return np.array(nn_input.astype(np.float32))
    
    def get_obs_dict(self):
            grid_width = self.bridge_map.width
            grid_height = self.bridge_map.height

            person_window = np.zeros((grid_height, grid_width))  
            water_window = np.zeros((grid_height, grid_width))   

            # loop through the grid
            for y in range(grid_height):
                for x in range(grid_width):
                    # check for water
                    if self.bridge_map.get_grid_type([x, y]) == self.bridge_map.grid_types["water"]:
                        water_window[y, x] = 1  

                    # Cceck for persons
                    for person in self.persons:
                        if np.array_equal(person.position, np.array([x, y])):
                            person_window[y, x] = 1  

            # agent's position (one-hot encoding)
            agent_window = np.zeros((grid_height, grid_width))
            agent_x, agent_y = self.get_agent_location()
            agent_window[agent_y, agent_x] = 1  

            observation = OrderedDict({
                "agent_window": agent_window,
                "person_window": person_window,
                "water_window": water_window
            })
            return observation
    
    """
    returns which morally relevant propositions hold true in the current state
    """
    def get_lables(self):
        labels = set()
        for person in self.persons:
            if person.status == Status.IN_WATER_AT_RISK:
                labels.add("D")
            for bridge, bridge_tiles in self. bridge_map.bridges.items():
                for tile in bridge_tiles:
                    if np.equal(person.position,tile).all(): 
                        labels.add("B")
        return labels
    
    """
    functions for handling positions
    """
    def get_agent_location(self):
        return self._agent_location

    def agent_on_bridge(self):
        agent_location = self.get_agent_location()
        return (self.bridge_map.position_on_bridge(agent_location))
    
    def agent_in_front_of_bridge(self):
        agent_location = self.get_agent_location()
        return (self.bridge_map.in_front_of_bridge(agent_location))
    
    def agent_behind_bridge(self):
        agent_location = self.get_agent_location()
        return (self.bridge_map.behind_bridge(agent_location))

    def get_coordinates_drowning(self):
        coordinates = []
        for person in self.persons:
            if np.array_equal(person.position, coordinates) and self.env.bridge_map.in_water(person.position):
                coordinates.append(person.position)  

    def to_2d_coordinates(self, index, width):
        row = index // width
        col = index % width
        return [col, row]
    
    def to_1d_index(self, position):
        position_1d = position[1] * self.bridge_map.width + position[0]
        return position_1d
    
    def get_static_person_position(self, state, person):
        return state[5+self.static_persons.index(person)]
    
    def random_agent_pos(self, random_gen):
        # choose the agent's location randomly 
        self._agent_location[0] = random_gen.integers(0, self.bridge_map.width)
        self._agent_location[1] = random_gen.integers(0, self.bridge_map.height)
        agent_grid_type = self.bridge_map.get_grid_type(self._agent_location)

        # ensure that the agent doesn't spawn in the water or at the goal position
        while agent_grid_type == self.bridge_map.grid_types["water"] or np.equal(self._agent_location, self.target_location).all():
            self._agent_location[0] = random_gen.integers(0, self.bridge_map.width)
            self._agent_location[1] = random_gen.integers(0, self.bridge_map.height)
            agent_grid_type = self.bridge_map.get_grid_type(self._agent_location)

    """
    functions for randomizing certain parts of the layout of the map
    """
    def set_random_target_location(self, random_gen):
        self.target_location[0] = random_gen.integers(0, self.bridge_map.width)
        self.target_location[1] = random_gen.integers(0, self.bridge_map.height)

    """
    functions for resetting the environment
    """
    def reset(self, seed=None, options=None, state=None, random_init="no randomness"):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        super().reset(seed=seed)
        self.initialize_at_reset(state, random_init, self.np_random)
        observation = self.observation()
        info = {"cost": 0}
        self.render()

        return observation, info
    
    def initialize_at_reset(self, state, random_init, random_gen):
        self.static_persons = [person for person in self.persons if isinstance(person, Person_Static)]
        if state is not None:
            self._agent_location = np.array(self.to_2d_coordinates((state[0]), self.bridge_map.width))
            for person in self.persons:
                person_position = list(person.position)
                if isinstance(person, Person_Moving):
                    person_position = state[person.person_id]
                elif isinstance(person, Person_Static):
                    person_position = self.get_static_person_position(state, person)
                person_position = np.array(self.to_2d_coordinates(person_position, self.bridge_map.width))
                person.position = person_position
                if np.equal(person_position,self.bridge_map.NOT_ON_MAP).all():
                    person.status = Status.NOT_IN_GAME
                elif self.bridge_map.get_grid_type(person.position) == self.bridge_map.grid_types["water"]:
                    person.status = Status.IN_WATER_AT_RISK
                else:
                    person.status = Status.MOVING
                person.respawn_timer = 0
            for person in self.persons:
                person.drowning_behavior.reset()

        elif random_init == "positions":
            self.random_agent_pos(random_gen)
            for person in self.persons:
                person.reset_random_pos(self.bridge_map, random_gen)

        #agent is set to the upper left corner and persons are set to the start positions of their trajectories
        elif random_init == "no randomness":
            for person in self.persons:
                person.reset(self.bridge_map)
            self._agent_location[0] = 0
            self._agent_location[1] = 0
            

    """
    update the envrionment according to what the agent does and subsequently update the environment according to how the persons move
    returns the reward according to the selected reward type (instrumental, rescueing, waiting or MO)
    also returns the output of a cost functions indiciating if the agent pushed a person off the bridge
    """
    def step(self, action):
        reward_waiting = 0
        reward_rescuing = 0
        reward_instrumental = 0
        constr_waiting = 0
        
        terminated = False

        # execute the action if it is taking a step in a direction
        if action in range(4):
            direction = self.action_to_direction[action]
            # ensures that the agent's position stays in the grid
            self._agent_location[0] = np.clip(
                self._agent_location[0] + direction[0], 0, self.bridge_map.width - 1
            )
            self._agent_location[1] = np.clip(
                self._agent_location[1] + direction[1], 0, self.bridge_map.height - 1
            )
            #if the agent runs into a person on the bridge, it pushes that person into the water (with a certain probability)
            for person in self.persons:
                if np.equal(self._agent_location, person.position).all() and self.bridge_map.position_on_bridge(person.position):
                    person.collision(self.bridge_map, self.np_random)
                    if self.bridge_map.in_water(person.position):
                        reward_waiting += Norms.ON_BRIDGE
                        constr_waiting += 1

        # exceute the action if it is rescuing a person 
        if action == 4:
            adjacent_tiles = self. bridge_map.get_adjacent_tiles(self._agent_location)
            for adjacent_tile in adjacent_tiles:
                #check if there is a person in reach that is in water and thus needs to be rescued
                for person in self.persons:
                    if np.all(person.position == adjacent_tile) and person.status == Status.IN_WATER_AT_RISK:
                        person.rescued(self.bridge_map)
                        if all(not self.bridge_map.in_water(person.position) for person in self.persons):
                            reward_rescuing += Values.RESCUING
                            if self.get_reward_type() == "rescuing":
                                terminated = True

        # update positions of persons
        for person in self.persons:
            # if there is a person at a dangerous spot they fall into the water (with a certain probability)
            for dangerous_spot in self.bridge_map.dangerous_spots:
                if np.equal(person.position,np.array(dangerous_spot)).all(): 
                    person.slipping(self.bridge_map, self.np_random)
            person.move(self.bridge_map)      

        # episode termiantes if the agent resaches its goal or falls into the water
        if np.array_equal(self._agent_location, self.target_location):
            #reward_instrumental += Instrumental.goal_reached
            if self.reward_type=="instrumental" or self.reward_type=="MO":
                terminated = True
                reward_instrumental += Instrumental.goal_reached
        elif self.bridge_map.get_grid_type(self._agent_location)==self.bridge_map.grid_types["water"]:
            terminated = True
        # if the reward type is rescuing, an episode termiantes if the agent rescues all persons from the water after at least one person has fallen into the water
        elif all(np.all(person.position == self.bridge_map.NOT_ON_MAP) for person in self.persons) and self.reward_type=="rescuing":
            terminated = True

        observation = self.observation()
        info = {"cost": constr_waiting}

        if self.get_reward_type() == "instrumental":
            reward = reward_instrumental
        elif self.get_reward_type() == "waiting":
            reward = reward_waiting
        elif self.get_reward_type() == "rescuing":
            reward = reward_rescuing
        elif self.get_reward_type() == "MO":
            reward = [reward_instrumental, reward_waiting, reward_rescuing]

        self.render()
        
        for person in self.persons:
            person.update_internals(self.bridge_map, self.np_random)

        return observation, reward, terminated, False, info
    
    """
    functions for rendering the envrionment 
    """
    
    def render(self):
        if self.render_mode == "human":
         return self.render_frame()
        
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Bridge Setting")
            self.window = pygame.display.set_mode(
                (self.window_size_width, self.window_size_height)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size_width, self.window_size_height))
        canvas.fill((0, 0, 255))
        pix_width_size = self.window_size_width / self.bridge_map.width  
        pix_height_size = self.window_size_height / self.bridge_map.height  

        y_coordinate = 0
        for row in self.bridge_map.grid_array:
            x_coordinate = 0
            for grid in row:
                if self.bridge_map.grid_types["water"] == grid:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 255),  # color for water (blue)
                        pygame.Rect(
                            (x_coordinate * pix_width_size, y_coordinate * pix_height_size),  
                            (pix_width_size, pix_height_size),  
                        ),
                    )
                elif self.bridge_map.grid_types["land"] == grid:
                    pygame.draw.rect(
                        canvas,
                        (139, 69, 19),  # color for land (brown)
                        pygame.Rect(
                            (x_coordinate * pix_width_size, y_coordinate * pix_height_size), 
                            (pix_width_size, pix_height_size),  
                        ),
                    )
                x_coordinate += 1
            y_coordinate += 1

         # render the target location in green
        target_pos = self.target_location * np.array([pix_width_size, pix_height_size])
        pygame.draw.rect(
            canvas,
            (0, 255, 0),  
            pygame.Rect(
                (target_pos[0], target_pos[1]),  
                (pix_width_size, pix_height_size),  
            ),
        )

        agent_pos = (self._agent_location + 0.5) * np.array([pix_width_size, pix_height_size])
        smaller_radius = min(pix_width_size, pix_height_size) / 6  

        for person in self.persons:
            if not person.status == Status.NOT_IN_GAME:
                # calculate person position
                person_pos = (np.array(person.position) + 0.5) * np.array([pix_width_size, pix_height_size])

                # if the agent and the person are on the same tile, draw two smaller circles next to each other
                if np.array_equal(self._agent_location, person.position):
                    left_pos = agent_pos - np.array([smaller_radius, 0])
                    right_pos = agent_pos + np.array([smaller_radius, 0])

                    # draw the agent
                    pygame.draw.circle(
                        canvas,
                        (255, 255, 255), 
                        left_pos,
                        smaller_radius,
                    )

                    # set color based on the person's status
                    if person.status == Status.MOVING or person.status == Status.STANDING:
                        color = (0, 0, 0)
                    elif person.status == Status.IN_WATER_AT_RISK:
                        color = (255, 0, 0)

                    # draw the person
                    pygame.draw.circle(
                        canvas,
                        color,
                        right_pos,
                        smaller_radius,
                    )
                else:
                    # draw the person 
                    if person.status == Status.MOVING or person.status == Status.STANDING:
                        color = (0, 0, 0)
                    elif person.status == Status.IN_WATER_AT_RISK:
                        color = (255, 0, 0)

                    pygame.draw.circle(
                        canvas,
                        color,
                        person_pos,
                        min(pix_width_size, pix_height_size) / 3, 
                    )

        persons_in_game = [person for person in self.persons if not person.status == Status.NOT_IN_GAME]

        # if no person is on the agent's location, draw the agent
        if not any(np.array_equal(self._agent_location, person.position) for person in persons_in_game):
            pygame.draw.circle(
                canvas,
                (255, 255, 255), #draws agent in white color
                agent_pos,
                min(pix_width_size, pix_height_size) / 3,  
            )


        # add the gridlines 
        for x in range(self.bridge_map.width + 1):
            # vertical lines
            pygame.draw.line(
                canvas,
                0,  # line color (black)
                (pix_width_size * x, 0),  
                (pix_width_size * x, self.window_size_height),  
                width=3,
            )

        for y in range(self.bridge_map.height + 1):
            # horizontal lines
            pygame.draw.line(
                canvas,
                0,  # line color (black)
                (0, pix_height_size * y),  
                (self.window_size_width, pix_height_size * y), 
                width=3,
            )

        font = pygame.font.Font(None, 36) 
        text = ""
        for label in self.get_lables():
            text = text + label + " "
        text_surface = font.render(text, True, (255, 255, 255))  # white color text
        canvas.blit(text_surface, (10, 10))

        # move drawings from canvas to the visible window
        self.window.blit(canvas, canvas.get_rect())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                pygame.quit()
                sys.exit()

        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.metadata["render_fps"])
