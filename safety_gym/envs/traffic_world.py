import numpy as np
import heapq
import random
from  more_itertools import unique_everseen


class RandomRoads:

    def __init__(self, grid_size):
        self.existing_roads = []
        self.grid_size = grid_size

    def create_road_start_and_ends(self, num_roads=5, num_tries=10):
        """
        Generate 'road_count' number of road start and end points.
        Args:
            road_count (int): number of road start and end point pairs
            num_tries (int): how many times we allow trying to create a road fail, if number is exceeded return False
        Return:
            list of Road object with length of 'road_count'

        """
        self.existing_roads = self.outside_roads(self.grid_size)  # create roads around our area
        failures = 0  # number of times it was tried to create a road, but it didn't fit the world

        while len(self.existing_roads) < num_roads + 3:  # plus 3 because we do not count the 4 outside roads
            new_road = self.choose_new_split(self.existing_roads, min_dist_between=2)

            # failed creating a new road
            if new_road is None:
                failures += 1
            # succeeded creating a new road
            else:
                self.existing_roads.append(new_road)
                print('adding a new road with coordinates', new_road.pt1, new_road.pt2)
                # too many failures
            if failures == num_tries:
                return False

        return self.existing_roads

    def choose_new_split(self, roads, min_dist_between, num_tries=10, random_state=np.random.RandomState()):
        '''
            Given a list of walls, choose a random road and draw a new road perpendicular to it.
            NOTE: Right now this O(n_walls^2). We could probably get this to linear if we did
                something smarter with the occupancy grid. Until n_walls gets way bigger this
                should be fine though.
            Args:
                roads (Wall list): walls to possibly draw a new road from
                min_dist_between (int): closest another parallel road can be to the new road in grid cells.
                num_tries (int): number of times before we can fail in placing a road before giving up
                random_state (np.random.RandomState): random state to use for sampling
        '''
        for i in range(num_tries):
            road1 = random_state.choice(roads)
            proposed_roads = [self.connect_roads(road1, road2, min_dist_between, random_state=random_state)
                              for road2 in roads if road2 != road1]
            proposed_roads = [road for road in proposed_roads
                              if road is not None
                              and not np.any([road.intersects(_wall) for _wall in roads])]
            if len(proposed_roads):
                new_road = random_state.choice(proposed_roads)
                for road in roads:
                    road.maybe_add_edge(new_road)
                    new_road.maybe_add_edge(road)
                return new_road
        return None

    def connect_roads(self, road1, road2, min_dist_between, random_state=np.random.RandomState()):
        '''
            Draw a random new wall connecting wall1 and wall2. Return None if
            the drawn wall was closer than min_dist_between to another wall
            or the wall wasn't valid.
            NOTE: This DOES NOT check if the created wall overlaps with any existing walls, that
                should be done outside of this function
            Args:
                wall1, wall2 (Wall): walls to draw a new wall between
                min_dist_between (int): closest another parallel wall can be to the new wall in grid cells.
                random_state (np.random.RandomState): random state to use for sampling
        '''
        if road1.is_vertical != road2.is_vertical:
            return None
        length = random_state.randint(1, road1.length + 1)
        if road1.is_vertical:
            pt1 = [road1.pt1[0], road1.pt1[1] + length]
            pt2 = [road2.pt1[0], road1.pt1[1] + length]
        else:
            pt1 = [road1.pt1[0] + length, road1.pt1[1]]
            pt2 = [road1.pt1[0] + length, road2.pt1[1]]

        # Make sure that the new wall actually touches both walls
        # and there is no wall close to this new wall
        road1_right_of_road2 = np.any(np.array(pt2) - np.array(pt1) < 0)
        if road1_right_of_road2:
            dists = np.array(pt1)[None, :] - np.array(road1.left_edges)
        else:
            dists = np.array(pt1)[None, :] - np.array(road1.right_edges)
        min_dist = np.linalg.norm(dists, axis=1).min()

        if road2.is_touching(pt2) and min_dist > min_dist_between and pt1 != pt2:
            return Road(pt1, pt2)
        return None

    def outside_roads(self, grid_size, rgba=(0, 1, 0, 0.1), use_low_wall_height=False):
        print('grid_size:', grid_size)
        height = 0.5 if use_low_wall_height else 4.0
        return [Road([0, 0], [0, grid_size - 1], height=height, rgba=rgba),
                Road([0, 0], [grid_size - 1, 0], height=height, rgba=rgba),
                Road([grid_size - 1, 0], [grid_size - 1, grid_size - 1], height=height, rgba=rgba),
                Road([0, grid_size - 1], [grid_size - 1, grid_size - 1], height=height, rgba=rgba)]


class Road:
    '''
        Defines a wall object which is essentially a pair of points on a grid
            with some useful helper functions for creating randomized rooms.
        Args:
            pt1, pt2 (float tuple): points defining the wall
            height (float): wall height
            rgba (float tuple): wall rgba
    '''

    def __init__(self, pt1, pt2, height=0.5, rgba=(0, 1, 0, 1)):
        assert pt1[0] == pt2[0] or pt1[1] == pt2[1], (
            "Currently only horizontal and vertical walls are supported")
        self.is_vertical = pt1[0] == pt2[0]
        # Make sure pt2 is top right of pt1
        if np.any(np.array(pt2) - np.array(pt1) < 0):
            self.pt1 = np.array(pt2)
            self.pt2 = np.array(pt1)
        else:
            self.pt1 = np.array(pt1)
            self.pt2 = np.array(pt2)
        self.length = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
        self.height = height
        self.rgba = rgba
        # Variables defining where other walls split from this wall on the left and right.
        # For horizontal walls, left means below, right means above
        self.left_edges = [self.pt1, self.pt2]
        self.right_edges = [self.pt1, self.pt2]

    def is_touching(self, pt):
        '''
            Is pt (tuple) touching this wall
        '''
        if self.is_vertical:
            return pt[0] == self.pt1[0] and pt[1] >= self.pt1[1] and pt[1] <= self.pt2[1]
        else:
            return pt[1] == self.pt1[1] and pt[0] >= self.pt1[0] and pt[0] <= self.pt2[0]

    def maybe_add_edge(self, road):
        '''
            Check if wall is originating from this wall. If so add it to the list of edges.
        '''
        if self.is_vertical == road.is_vertical:
            return
        if self.is_touching(road.pt1):
            self.right_edges.append(road.pt1)
        elif self.is_touching(road.pt2):
            self.left_edges.append(road.pt2)

    def intersects(self, road):
        '''
            Check if intersects with wall.
        '''
        if self.is_vertical == road.is_vertical:
            return False
        return np.all(np.logical_and(self.pt1 < road.pt2, road.pt1 < self.pt2))

    def split_for_doors(self, num_doors=1, door_size=1, all_connect=False,
                        random_state=np.random.RandomState()):
        '''
            Split this wall into many walls with 'doors' in between.
            Args:
                num_doors (int): upper bound of number of doors to create
                door_size (int): door size in grid cells
                all_connect (bool): create a door in every wall segment between pairs of points
                    where other walls connect with this wall
                random_state (np.random.RandomState): random state to use for sampling
        '''
        edges = np.unique(self.left_edges + self.right_edges, axis=0)
        edges = np.array(sorted(edges, key=lambda x: x[1] if self.is_vertical else x[0]))
        rel_axis = edges[:, 1] if self.is_vertical else edges[:, 0]
        diffs = np.diff(rel_axis)
        possible_doors = diffs >= door_size + 1

        # Door regions are stretches on the wall where we could create a door.
        door_regions = np.arange(len(edges) - 1)
        door_regions = door_regions[possible_doors]

        # The number of doors on this wall we want to/can create
        num_doors = len(edges) - 1 if all_connect else num_doors
        num_doors = min(num_doors, len(door_regions))
        if num_doors == 0 or door_size == 0:
            return [self], []

        # Sample num_doors regions to which we will add doors.
        door_regions = np.sort(random_state.choice(door_regions, num_doors, replace=False))
        new_roads = []
        doors = []
        new_road_start = edges[0]
        for door in door_regions:
            # door_start and door_end are the first and last point on the wall bounding the door
            # (inclusive boundary)
            door_start = random_state.randint(1, diffs[door] - door_size + 1)
            door_end = door_start + door_size - 1

            # Because door boundaries are inclusive, we add 1 to the door_end to get next wall
            # start cell and subtract one from the door_start to get the current wall end cell.
            if self.is_vertical:
                new_road_end = [edges[door][0], edges[door][1] + door_start - 1]
                next_new_road_start = [new_road_start[0], edges[door][1] + door_end + 1]
                door_start_cell = [edges[door][0], edges[door][1] + door_start]
                door_end_cell = [new_road_start[0], edges[door][1] + door_end]
            else:
                new_road_end = [edges[door][0] + door_start - 1, edges[door][1]]
                next_new_road_start = [edges[door][0] + door_end + 1, edges[door][1]]
                door_start_cell = [edges[door][0] + door_start, edges[door][1]]
                door_end_cell = [new_road_start[0] + door_end, edges[door][1]]

            # Store doors as inclusive boundaries.
            doors.append([door_start_cell, door_end_cell])
            # Check that the new wall isn't size 0
            if np.linalg.norm(np.array(new_road_start) - np.array(new_road_end)) > 0:
                new_roads.append(Road(new_road_start, new_road_end))
            new_road_start = next_new_road_start
        if np.linalg.norm(np.array(new_road_start) - np.array(edges[-1])) > 0:
            new_roads.append(Road(new_road_start, edges[-1]))
        return new_roads, doors


class Visualize:

    def __init__(self, roads, grid_size):
        self.roads = roads
        self.road_loc =  [(7.0, 3.0), (7.5, 3.0), (7.0, 3.5), (7.5, 3.5), (0.0, 7.0), (0.5, 7.0), (0.0, 7.5), (0.5, 7.5), (3.0, 7.0), (3.0, 7.5), (3.5, 7.0), (3.5, 7.5), (2.0, 4.5), (2.5, 4.0), (4.0, 0.0), (4.0, 0.5), (4.5, 0.0), (4.5, 0.5), (2.5, 4.5), (6.0, 7.0), (6.0, 7.5), (6.5, 7.0), (6.5, 7.5), (7.0, 6.0), (7.5, 6.0), (7.0, 6.5), (7.5, 6.5), (0.5, 4.0), (0.0, 4.5), (0.0, 4.0), (0.5, 4.5), (3.0, 2.0), (3.5, 2.0), (3.0, 2.5), (3.5, 2.5), (6.0, 0.0), (6.0, 0.5), (6.5, 0.0), (6.5, 0.5), (7.5, 5.5), (7.0, 5.0), (7.5, 5.0), (7.0, 5.5), (0.0, 1.0), (0.5, 1.0), (0.0, 1.5), (0.5, 1.5), (3.0, 1.0), (3.5, 1.0), (3.0, 1.5), (3.5, 1.5), (7.0, 2.0), (7.5, 2.0), (7.0, 2.5), (7.5, 2.5), (3.0, 6.0), (3.5, 6.0), (3.0, 6.5), (3.5, 6.5), (6.0, 4.0), (5.0, 4.0), (5.0, 4.5), (5.5, 4.0), (5.5, 4.5), (6.0, 4.5), (7.0, 1.0), (7.5, 1.0), (7.0, 1.5), (7.5, 1.5), (0.0, 5.0), (0.5, 5.0), (0.0, 5.5), (0.5, 5.5), (1.0, 0.0), (1.0, 0.5), (1.5, 0.0), (1.5, 0.5), (3.0, 5.0), (2.0, 7.0), (2.0, 7.5), (2.5, 7.0), (2.5, 7.5), (3.5, 5.0), (3.0, 5.5), (3.5, 5.5), (5.0, 7.0), (5.0, 7.5), (5.5, 7.0), (5.5, 7.5), (7.0, 4.5), (7.5, 4.5), (7.0, 4.0), (7.5, 4.0), (0.0, 2.5), (0.5, 2.5), (0.0, 2.0), (0.5, 2.0), (3.0, 0.0), (3.0, 0.5), (3.5, 0.0), (3.5, 0.5), (5.0, 0.0), (5.0, 0.5), (5.5, 0.0), (5.5, 0.5), (1.0, 4.0), (1.0, 4.5), (1.5, 4.0), (1.5, 4.5), (7.0, 0.0), (7.0, 0.5), (7.5, 0.0), (7.5, 0.5), (0.0, 6.0), (0.5, 6.0), (0.0, 6.5), (0.5, 6.5), (1.0, 7.0), (1.0, 7.5), (1.5, 7.0), (1.5, 7.5), (3.0, 4.0), (3.5, 4.0), (3.0, 4.5), (3.5, 4.5), (6.5, 4.0), (6.5, 4.5), (4.0, 7.0), (4.0, 7.5), (4.5, 7.0), (4.5, 7.5), (2.0, 4.0), (7.0, 7.0), (7.5, 7.0), (7.0, 7.5), (7.5, 7.5), (0.0, 3.5), (0.5, 3.5), (0.0, 3.0), (0.5, 3.0), (3.0, 3.0), (3.5, 3.0), (3.0, 3.5), (3.5, 3.5), (4.0, 4.0), (4.0, 4.5), (4.5, 4.0), (4.5, 4.5), (0.5, 0.5), (0.0, 0.5), (0.0, 0.0), (0.5, 0.0), (2.0, 0.0), (2.0, 0.5), (2.5, 0.0), (2.5, 0.5)]#[]
        self.ped_road_loc = []
        self.grid_size = grid_size

    def calc_coordinates_between_two_points(self, start, end):
        """
        Takes two points as numpy 2D array and returns coordinates that connect those two points. This is needed
        for vizualising the roads, because Road class only stores the start and end point

        Parameters:
        ends (numpy array): cooridnates of two points (e.g. start = [-1, 0], end = [2, 0]))

        Returns:
        numpy array: coordinates that connect those two points  (e.g. [[-1  0] [ 0  0] [ 1  0] [ 2  0]])
        """
        ends = np.array([start, end]) * 2
        d = np.diff(ends, axis=0)[0]  # gives the difference btw both arrays on an a vertical axis
        j = np.argmax(np.abs(d))  # returns the index of the max value in the previous vector (note we take abs)
        D = d[j]  # we get the value from the vector
        aD = np.abs(D)  # the absolute value of the maximum value
        # get's the slope in a way and floor divided it to get the change then add it to the first observation to get the path
        return (ends[0] + (np.outer(np.arange(aD + 1), d) + (
                aD >> 1)) // aD) / 2  # to have it with the 0.5 increment just multibly the cordinates by 2 and devide the result by 0.5 I guess is will work

    def calculate_road_locations(self):
        '''
        Calculate locations where to place roads based on road start and end points.
        We want the road to be wider than 0.5. Therefore for
            - vertical roads add 0.5 to x coordinate
            - horizontal roads add 0.5 to y coordinate

        Returns
            list containing tuples (e.g. [(1,2),(1,3),...])
        '''
        self.road_loc = set()  # only want unique coordinates

        for road in self.roads:
            road_path = self.calc_coordinates_between_two_points(road.pt1, road.pt2)

            # change the road path from list to tuple ([1,2] -> (1,2))
            for x, y in road_path:
                self.road_loc.add((x, y))
                # for vertical roads -> add one more road to the right
                if road.is_vertical:
                    self.road_loc.add((x + 0.5, y))
                # for horizontal roads -> add one more road above
                else:
                    self.road_loc.add((x, y + 0.5))

        # the upper corner road is always not added during previous process, add it now
        self.road_loc.add((self.grid_size - 0.5, self.grid_size - 0.5))
        return list(self.road_loc)

    def calculate_pedestrian_road_locations(self):
        '''
        Calculate locations where to place the pedestrian roads based on road start and end points.
        Take into account, that roads are made 0.5 thicker than described in start and end points.

        Returns
            list containing tuples (e.g. [(1,2),(1,3),...])
        '''
        self.ped_road_loc = set()  # only want unique coordinates

        # for every road point, surround it with pedestrian roads, overlapping is dealt with later
        for x, y in self.road_loc:
            self.ped_road_loc.add((x, y + 0.5))
            self.ped_road_loc.add((x, y - 0.5))
            self.ped_road_loc.add((x + 0.5, y))
            self.ped_road_loc.add((x - 0.5, y))

        # all the corners are missing pedestrian roads, add them now
        self.ped_road_loc.add((self.grid_size, self.grid_size))
        self.ped_road_loc.add((-0.5, -0.5))
        self.ped_road_loc.add((-0.5, self.grid_size))
        self.ped_road_loc.add((self.grid_size, -0.5))

        # make sure that the pedestrian roads do not overlap with normal roads
        self.ped_road_loc = set(self.ped_road_loc) - set(
            self.road_loc)  # [ped_loc for ped_loc in list(self.ped_road_loc) if ped_loc not in list(self.road_loc)]

        return list(self.ped_road_loc)

    def calculate_pedestrian_road_locations_old(self):
        '''
        Calculate locations where to place the pedestrian roads based on road start and end points.
        Take into account, that roads are made 0.5 thicker than described in start and end points.

        Returns
            list containing tuples (e.g. [(1,2),(1,3),...])
        '''
        self.ped_road_loc = set()  # only want unique coordinates

        for road in self.road_loc:
            road_path = self.calc_coordinates_between_two_points(road.pt1, road.pt2)

            # change the road path from list to tuple ([1,2] -> (1,2))
            for x, y in road_path:
                self.ped_road_loc.add((x, y))
                # add pedestrian roads to left and right
                if road.is_vertical:
                    self.ped_road_loc.add((x + 1, y))
                    self.ped_road_loc.add((x - 0.5, y))
                # add pedestrian roads below and above
                else:
                    self.ped_road_loc.add((x, y + 1))
                    self.ped_road_loc.add((x, y - 0.5))
        # all the corners are missing pedestrian roads, add it now
        # upper right
        self.ped_road_loc.add((self.grid_size, self.grid_size))
        self.ped_road_loc.add((self.grid_size - 0.5, self.grid_size))
        self.ped_road_loc.add((self.grid_size, self.grid_size - 0.5))

        # make sure that the pedestrian roads do not overlap with normal roads
        self.ped_road_loc = set(self.ped_road_loc) - set(
            self.road_loc)  # [ped_loc for ped_loc in list(self.ped_road_loc) if ped_loc not in list(self.road_loc)]

        return list(self.ped_road_loc)

    def calculate_house_locations(self):
        '''
        Everything that is NOT road will be covered with hazards.
        '''
        # calculate all the possible coordinates in the grid
        all_points_inside_grid = [(i, j) for j in np.arange(0.0, self.grid_size, 0.5) for i in
                                  np.arange(0.0, self.grid_size, 0.5)]
        # subtract the road coordinates
        return list(set(all_points_inside_grid) - set(self.road_loc) - set(self.ped_road_loc))


class GoalPath:
    '''
    Find path (coordinates) that connect start and goal locations.
    start and goal locations must be **numpy arrays** with shape (1,2) e.g.  np.array([3.5,1])
    '''

    def __init__(self, start_loc, goal_loc, road_locations):
        self.start_loc = np.around(start_loc * 2.0) / 2.0  # round to closest .5
        self.goal_loc = np.around(goal_loc * 2.0) / 2.0
        self.road_locations =  [(7.0, 3.0), (7.5, 3.0), (7.0, 3.5), (7.5, 3.5), (0.0, 7.0), (0.5, 7.0), (0.0, 7.5), (0.5, 7.5), (3.0, 7.0), (3.0, 7.5), (3.5, 7.0), (3.5, 7.5), (2.0, 4.5), (2.5, 4.0), (4.0, 0.0), (4.0, 0.5), (4.5, 0.0), (4.5, 0.5), (2.5, 4.5), (6.0, 7.0), (6.0, 7.5), (6.5, 7.0), (6.5, 7.5), (7.0, 6.0), (7.5, 6.0), (7.0, 6.5), (7.5, 6.5), (0.5, 4.0), (0.0, 4.5), (0.0, 4.0), (0.5, 4.5), (3.0, 2.0), (3.5, 2.0), (3.0, 2.5), (3.5, 2.5), (6.0, 0.0), (6.0, 0.5), (6.5, 0.0), (6.5, 0.5), (7.5, 5.5), (7.0, 5.0), (7.5, 5.0), (7.0, 5.5), (0.0, 1.0), (0.5, 1.0), (0.0, 1.5), (0.5, 1.5), (3.0, 1.0), (3.5, 1.0), (3.0, 1.5), (3.5, 1.5), (7.0, 2.0), (7.5, 2.0), (7.0, 2.5), (7.5, 2.5), (3.0, 6.0), (3.5, 6.0), (3.0, 6.5), (3.5, 6.5), (6.0, 4.0), (5.0, 4.0), (5.0, 4.5), (5.5, 4.0), (5.5, 4.5), (6.0, 4.5), (7.0, 1.0), (7.5, 1.0), (7.0, 1.5), (7.5, 1.5), (0.0, 5.0), (0.5, 5.0), (0.0, 5.5), (0.5, 5.5), (1.0, 0.0), (1.0, 0.5), (1.5, 0.0), (1.5, 0.5), (3.0, 5.0), (2.0, 7.0), (2.0, 7.5), (2.5, 7.0), (2.5, 7.5), (3.5, 5.0), (3.0, 5.5), (3.5, 5.5), (5.0, 7.0), (5.0, 7.5), (5.5, 7.0), (5.5, 7.5), (7.0, 4.5), (7.5, 4.5), (7.0, 4.0), (7.5, 4.0), (0.0, 2.5), (0.5, 2.5), (0.0, 2.0), (0.5, 2.0), (3.0, 0.0), (3.0, 0.5), (3.5, 0.0), (3.5, 0.5), (5.0, 0.0), (5.0, 0.5), (5.5, 0.0), (5.5, 0.5), (1.0, 4.0), (1.0, 4.5), (1.5, 4.0), (1.5, 4.5), (7.0, 0.0), (7.0, 0.5), (7.5, 0.0), (7.5, 0.5), (0.0, 6.0), (0.5, 6.0), (0.0, 6.5), (0.5, 6.5), (1.0, 7.0), (1.0, 7.5), (1.5, 7.0), (1.5, 7.5), (3.0, 4.0), (3.5, 4.0), (3.0, 4.5), (3.5, 4.5), (6.5, 4.0), (6.5, 4.5), (4.0, 7.0), (4.0, 7.5), (4.5, 7.0), (4.5, 7.5), (2.0, 4.0), (7.0, 7.0), (7.5, 7.0), (7.0, 7.5), (7.5, 7.5), (0.0, 3.5), (0.5, 3.5), (0.0, 3.0), (0.5, 3.0), (3.0, 3.0), (3.5, 3.0), (3.0, 3.5), (3.5, 3.5), (4.0, 4.0), (4.0, 4.5), (4.5, 4.0), (4.5, 4.5), (0.5, 0.5), (0.0, 0.5), (0.0, 0.0), (0.5, 0.0), (2.0, 0.0), (2.0, 0.5), (2.5, 0.0), (2.5, 0.5)]
        #road_locations

    def get_neighbours(self, l):
        '''
        l - locations e.g. np.array([3.5,1])
        returns locations right, down, left, up
        '''
        x, y = l
        return np.array([(x + 0.5, y), (x, y - 0.5), (x - 0.5, y), (x, y + 0.5)])

    def closest_point_to_goal(self, neighbours):
        '''
        Returns the closest neighbour location to the goal.

        1) Given four points, calculate which one is closest to the goal.
        2) Checks that chosen point is actually on car road and not on pedestrian road
        3) If true then return the location, else continue search
        '''
        goal_locs = np.repeat(np.array([self.goal_loc]), 4, axis=0)  # repeat goal location 4 times, for left right up and down
        dists = self.eucledian_distance(neighbours, goal_locs)  # calculate the distance between each neighbour and end goal
        dists = dists + np.random.uniform(0, 0.5, dists.shape)  # all distances need to be unique so add a little bit randomness to distances, this also helps for the algorithm to not get stuck between two states

        # return the closest point to the goal that is on car road
        for i in range(1, 5):
            min_idx = np.where(dists == heapq.nsmallest(i, dists)[-1])  # get the  idx of closest location to goal
            closest_point = tuple(neighbours[min_idx][0])  # get the cooridnates of the closest point to goal
            if closest_point in self.road_locations:  # check if the closest point exist on road
                return closest_point

        print('\nWARNING: None of the neighbours\n', neighbours, 'are on the road\n, (goal loc:', self.goal_loc, 'start_loc', self.start_loc, ')')
        return None

    def eucledian_distance(self, point_one, point_two):
        return np.sqrt(np.sum((point_one - point_two) ** 2, axis=1))

    def find_path_to_goal(self):
        '''
        Algorithm to find path to goal:
        1) get neighbours (up, down, left, right) of current location (start locations)
        2) pick the point x that is cloasest to the goal by eucledian distance and on car road
        3) append x to our path
        4) set current location to be our x
        5) repeat steps 1)-4) until we reach the goal
        '''
        path = []
        current_loc = self.start_loc
        it = 0
        # can finish when we have reached the goal location

        if all(self.start_loc == self.goal_loc):
            return [self.start_loc]

        while not all(current_loc == self.goal_loc):
            neigh = self.get_neighbours(current_loc)  # get neighbours of the current location
            loc_in_goal_path = self.closest_point_to_goal(neigh)  # get the closest suitable neighbour of our current location to end
            path.append(loc_in_goal_path)
            current_loc = loc_in_goal_path  # set the new location to current location

            it += 1
            if it > 1000:
                print('Warning: could not find path to goal, resampling')
                return None
        # the goal location should not be in the path, preserve all unique locations and their order
        return list(unique_everseen(path[:-1]))
