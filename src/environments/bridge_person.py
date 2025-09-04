import numpy as np
import random
from enum import Enum, auto
import random 

from abc import abstractmethod

"""
enum for the status of the persons; depending on the state of the bridge environment
"""
class Status(Enum):
    IN_WATER_AT_RISK = auto()
    MOVING = auto()
    NOT_IN_GAME = auto()
    NOT_MOVING = auto()
    STANDING = auto()

"""
class of persons in the bridge envrionment
"""
class Person():

    def __init__(self, position: np.array, status:Status, drowning_behavior, slipping_prob, pushed_off_bridge_prob, spawns_at_map_reset, respawn_time):
        self._status = status
        self.position = position
        self.respawn_time = respawn_time
        self.respawn_timer = 0
        self.drowning_behavior = drowning_behavior
        self.slipping_prob = slipping_prob
        self.pushed_off_bridge_prob = pushed_off_bridge_prob
        self.spawns_at_map_reset = spawns_at_map_reset
        self.slipping_spot = None

    @property
    def status (self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    def reset(self, bridge_map):
        self.respawn_timer = 0
        self.drowning_behavior.reset()

    """
    method alled to decide person is pushed into the water if the agent runs into it on the bridge
    add additional collision behavior as an extension of the transitions dynamics
    """        
    def collision(self, bridge_map, random_gen=None):
        for bridge, bridge_tiles in bridge_map.bridges.items():
            for tile in bridge_tiles:
                if np.equal(self.position,tile).all():
                    if (random_gen.random() if random_gen else random.random()) < self.pushed_off_bridge_prob:
                        self.to_water(bridge_map)
                        return "pushed to water"
        return None

    """
    method called to decide person falls into the water if it is at a dangerous spot
    """  
    def slipping(self, bridge_map, random_gen=None):
        random_number = random_gen.random() if random_gen else random.random()
        if random_number < self.slipping_prob:
            self.to_water(bridge_map)

    def to_water(self, bridge_map):
        for tile in bridge_map.get_adjacent_tiles(self.position):
            if bridge_map.get_grid_type(tile) == bridge_map.grid_types["water"]:
                self.slipping_spot = self.position
                self.position = tile
                break
        self.status = Status.IN_WATER_AT_RISK 

    """
    method called if person is rescued
    """      
    def rescued(self, bridge_map):
        self.position = bridge_map.NOT_ON_MAP
        self.status = Status.NOT_IN_GAME
        self.drowning_behavior.reset()

    """
    method for doing status updates at the end of each execution of the step method in the bridge envrionmnent
    """  
    def update_internals(self, bridge_map, random_gen=None):
        if self._status == Status.IN_WATER_AT_RISK:
            #delete person from map after drowning
            if self.drowning_behavior.drowning(random_gen):
                self.status = Status.NOT_IN_GAME
                self.position = bridge_map.NOT_ON_MAP
                self.drowning_behavior.reset()
        elif self._status == Status.NOT_IN_GAME:
            if self.respawn_timer < self.respawn_time:
                self.respawn_timer += 1
            #person reappears at the map 
            else:
                self.reset(bridge_map)
                self.respawn_timer = 0

"""
class for handling fixed-time drowning behavior in the bridge environment
"""  
class Drowning_TimeFixed:
    def __init__(self, drowns, drowning_threshold):
        self.drowning_threshold = drowning_threshold
        self.drowns = drowns
        self.time_in_water =0

    def reset(self): 
        self.time_in_water = 0
    
    def reset_time(self, time):
        self.time_in_water = time

    def drowning(self, random_gen=None):
        if self.drowns:
            if self.time_in_water >= self.drowning_threshold:
                self.reset()
                return True
            else: 
                self.time_in_water += 1
                return False
        else:
            return False
        
"""
class for random drowning behavior in the bridge environment
"""  
class Drowning_Random:
    def __init__(self, prob):
        self.prob = prob

    def reset(self):
        pass

    def reset_time(self):
        pass

    def drowning(self, random_gen=None):
        if (random_gen.random() if random_gen else random.random()) < self.prob:
            return True
        else:
            return False
        
"""
extends the base `Person` class by adding movement behavior;
a 'Person_Moving' has a trajectory (a list of positions), and moves 
along this path over time; once the final position is reached, the person 
is leaves the map
"""  
class Person_Moving(Person):
    def __init__(self, person_id, position: np.array, trajectory, status:Status, drowning_behavior, slipping_prob, pushed_off_bridge_prob, respawn_time):
        self.spawns_at_map_reset = True
        super().__init__(position, status, drowning_behavior=drowning_behavior, slipping_prob=slipping_prob, pushed_off_bridge_prob=pushed_off_bridge_prob, spawns_at_map_reset=self.spawns_at_map_reset, respawn_time=respawn_time)
        self.trajectory = trajectory
        self.person_id = person_id

    def check_path_finished(self):
        if np.equal(self.position,self.trajectory[-1]).all() and self.status == Status.MOVING:
            self._status = Status.NOT_IN_GAME

    def reset(self, bridge_map):
        super().reset(bridge_map=bridge_map)
        self.status = Status.MOVING
        self.position = self.trajectory[0]
    
    def reset_random_pos(self, bridge_map, random_gen=None):
        self.reset(bridge_map)
        if random_gen is not None:
            self.position = random_gen.choice(self.trajectory)
        else:
            self.position = random.choice(self.trajectory)
    
    def update_internals(self, bridge_map, random_gen=None):
        #self.check_path_finished()
        super().update_internals(bridge_map, random_gen)

    def next_index(self):
        for idx, pos in enumerate(self.trajectory):
            if np.equal(pos,self.position).all():  
                return idx
        raise LookupError("Person not on its trajectory.") 

    def move(self, bridge_map):
        if self.status == Status.MOVING and not np.equal(self.position,self.trajectory[-1]).all() and not np.equal(self.position,np.array(bridge_map.NOT_ON_MAP)).all():
            next_index = self.next_index()+1
            new_position = np.array(self.trajectory[next_index])
            self.position = new_position
        elif np.equal(self.position,self.trajectory[-1]).all():
            if self.status ==  Status.MOVING:
                self.position = bridge_map.NOT_ON_MAP
                self._status = Status.NOT_IN_GAME

"""
extends the base 'Person' class to enbale the placement of static persons on the map
a 'Person_Static' remains fixed at a designated position; a static person is always reset to their original spawn location and does 
not change position during the simulation.
""" 
class Person_Static(Person):
    def __init__(self, position: np.array, status:Status, drowning_behavior, slipping_prob, pushed_off_bridge_prob, respawn_time):
        super().__init__(position, status, drowning_behavior, slipping_prob, pushed_off_bridge_prob, spawns_at_map_reset=True, respawn_time=respawn_time)
        self.spawn_position = position
        self.position = position

    def reset(self, bridge_map):
        super().reset(bridge_map)
        self.status = Status.STANDING
        self.position = self.spawn_position
        if bridge_map.get_grid_type(self.position) == bridge_map.grid_types["water"]:
            self.status = Status.IN_WATER_AT_RISK
        else:
            self.status = Status.STANDING

    # static persons are always reset to their designated positions
    def reset_random_pos(self, bridge_map, random_gen=None):
        self.reset(bridge_map)

    def move(self, bridge_map):
        pass
