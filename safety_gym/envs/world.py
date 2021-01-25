#!/usr/bin/env python

import os
import xmltodict
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from mujoco_py import const, load_model_from_path, load_model_from_xml, MjSim, MjViewer, MjRenderContextOffscreen

import safety_gym
import sys

'''
Tools that allow the Safety Gym Engine to interface to MuJoCo.

The World class owns the underlying mujoco scene and the XML,
and is responsible for regenerating the simulator.

The way to use this is to configure a World() based on your needs 
(number of objects, etc) and then call `world.reset()`.

*NOTE:* The simulator should be accessed as `world.sim` and not just
saved separately, because it may change between resets.

Configuration is idiomatically done through Engine configuration,
so any changes to this configuration should also be reflected in 
changes to the Engine.

TODO:
- unit test scaffold
'''

# Default location to look for /xmls folder:
BASE_DIR = os.path.dirname(safety_gym.__file__)

def convert(v):
    ''' Convert a value into a string for mujoco XML '''
    if isinstance(v, (int, float, str)):
        return str(v)
    # Numpy arrays and lists
    return ' '.join(str(i) for i in np.asarray(v))


def rot2quat(theta):
    ''' Get a quaternion rotated only about the Z axis '''
    return np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)], dtype='float64')


class World:
    # Default configuration (this should not be nested since it gets copied)
    # *NOTE:* Changes to this configuration should also be reflected in `Engine` configuration
    DEFAULT = {
        'robot_base': 'xmls/cars/base_car/car1.xml',#"'xmls/car.xml', # 'xmls/cars/base_car/buddy.xml',   # Which robot XML to use as the base
        'robot_xy': np.zeros(2),  # Robot XY location
        'robot_rot': 0,  # Robot rotation about Z axis

        'floor_size': [3.5, 3.5, .1],  # Used for displaying the floor, overridden somewhere

        # Objects -- this is processed and added by the Engine class
        'objects': {},  # map from name -> object dict
        # Geoms -- similar to objects, but they are immovable and fixed in the scene.
        'geoms': {},  # map from name -> geom dict
        # Mocaps -- mocap objects which are used to control other objects
        'mocaps': {},

        # Determine whether we create render contexts
        'observe_vision': False,
    }

    def __init__(self, config={}, render_context=None):
        ''' config - JSON string or dict of configuration.  See self.parse() '''
        self.parse(config)  # Parse configuration
        self.first_reset = True
        self.viewer = None
        self.render_context = render_context
        self.update_viewer_sim = False
        self.robot = Robot(self.robot_base)

    def parse(self, config):
        ''' Parse a config dict - see self.DEFAULT for description '''
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

    @property
    def data(self):
        ''' Helper to get the simulation data instance '''
        return self.sim.data

    # TODO: remove this when mujoco-py fix is merged and a new version is pushed
    # https://github.com/openai/mujoco-py/pull/354
    # Then all uses of `self.world.get_sensor()` should change to `self.data.get_sensor`.
    def get_sensor(self, name):
        id = self.model.sensor_name2id(name)
        adr = self.model.sensor_adr[id]
        dim = self.model.sensor_dim[id]
        return self.data.sensordata[adr:adr + dim].copy()

    # here already have the floor
    def build(self):
        ''' Build a world, including generating XML and moving objects '''
        # Read in the base XML (contains robot, camera, floor, etc)
        self.robot_base_path = os.path.join(BASE_DIR, self.robot_base)
        with open(self.robot_base_path) as f:
            self.robot_base_xml = f.read()
        self.xml = xmltodict.parse(self.robot_base_xml)  # Nested OrderedDict objects

        # Convenience accessor for xml dictionary
        worldbody = self.xml['mujoco']['worldbody']

        # Move robot position to starting position
        worldbody['body']['@pos'] = convert(np.r_[self.robot_xy, self.robot.z_height])
        worldbody['body']['@quat'] = convert(rot2quat(self.robot_rot))

        # We need this because xmltodict skips over single-item lists in the tree
        worldbody['body'] = [worldbody['body']]
        if 'geom' in worldbody:
            worldbody['geom'] = [worldbody['geom']]
        else:
            worldbody['geom'] = []

        # Add equality section if missing
        if 'equality' not in self.xml['mujoco']:
            self.xml['mujoco']['equality'] = OrderedDict()
        equality = self.xml['mujoco']['equality']
        if 'weld' not in equality:
            equality['weld'] = []

        # Add asset section if missing
        if 'asset' not in self.xml['mujoco']:
            # old default rgb1: ".4 .5 .6"
            # old default rgb2: "0 0 0"
            # light pink: "1 0.44 .81"
            # light blue: "0.004 0.804 .996"
            # light purple: ".676 .547 .996"
            # med blue: "0.527 0.582 0.906"
            # indigo: "0.293 0 0.508"
            asset = xmltodict.parse('''
                <asset>
                    <texture type="skybox" builtin="gradient" rgb1="0.527 0.582 0.906" rgb2="0.1 0.1 0.35"
                        width="800" height="800" markrgb="1 1 1" mark="random" random="0.001"/>
                    <texture name="texplane" builtin="checker" height="100" width="100"
                        rgb1="0.7 0.7 0.7" rgb2="0.8 0.8 0.8" type="2d"/>
                    <material name="MatPlane" reflectance="0.1" shininess="0.1" specular="0.1"
                        texrepeat="10 10" texture="texplane"/>
                </asset>
                ''')
            self.xml['mujoco']['asset'] = asset['asset']


        # Add light to the XML dictionary
        light = xmltodict.parse('''<b>
            <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true"
                exponent="1" pos="0 0 0.5" specular="0 0 0" castshadow="false"/>
            </b>''')
        worldbody['light'] = light['b']['light']

        # Add floor to the XML dictionary if missing
        if not any(g.get('@name') == 'floor' for g in worldbody['geom']):
            floor = xmltodict.parse('''
                <geom name="floor" type="plane" condim="6" rgba="1, 0, 0, 1"/>
                ''')
            worldbody['geom'].append(floor['geom'])

        # Make sure floor renders the same for every world
        # FLOOR PARAMETERS CAN BE CHANGED HERE
        for g in worldbody['geom']:
            if g['@name'] == 'floor':
                # g.update({'@size': convert(self.floor_size), '@rgba': '108 122 137 1', '@material': 'MatPlane'})
                g.update({'@size': convert(self.floor_size), '@rgba': '0.38 0.38 0.38 1'})

        # Add cameras to the XML dictionary
        cameras = xmltodict.parse('''<b>
            <camera name="fixednear" pos="0 -2 2" zaxis="0 -1 1"/>
            <camera name="fixedfar" pos="0 -5 5" zaxis="0 -1 1"/>
            </b>''')
        worldbody['camera'] = cameras['b']['camera']

        # Build and add a tracking camera (logic needed to ensure orientation correct)
        theta = self.robot_rot
        xyaxes = dict(
                    x1=np.cos(theta), 
                    x2=-np.sin(theta),
                    x3=0,
                    y1=np.sin(theta),
                    y2=np.cos(theta),
                    y3=1
                    )
        pos = dict(
                xp=0*np.cos(theta) + (-2)*np.sin(theta),
                yp=0*(-np.sin(theta)) + (-2)*np.cos(theta),
                zp=2
                )
        track_camera = xmltodict.parse('''<b>
            <camera name="track" mode="track" pos="{xp} {yp} {zp}" xyaxes="{x1} {x2} {x3} {y1} {y2} {y3}"/>
            </b>'''.format(**pos, **xyaxes))
        worldbody['body'][0]['camera'] = [
            worldbody['body'][0]['camera'],
            track_camera['b']['camera']
            ]


        # Add objects to the XML dictionary
        for name, object in self.objects.items():
            assert object['name'] == name, f'Inconsistent {name} {object}'
            object = object.copy()  # don't modify original object
            object['quat'] = rot2quat(object['rot'])
            if name=='box':
                dim = object['size'][0]
                object['dim'] = dim
                object['width'] = dim/2
                object['x'] = dim
                object['y'] = dim
                body = xmltodict.parse('''
                    <body name="{name}" pos="{pos}" quat="{quat}">
                        <freejoint name="{name}"/>
                        <geom name="{name}" type="{type}" size="{size}" density="{density}"
                            rgba="{rgba}" group="{group}"/>
                        <geom name="col1" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="{x} {y} 0"/>
                        <geom name="col2" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="-{x} {y} 0"/>
                        <geom name="col3" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="{x} -{y} 0"/>
                        <geom name="col4" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="-{x} -{y} 0"/>
                    </body>
                '''.format(**{k: convert(v) for k, v in object.items()}))
            else:
                body = xmltodict.parse('''
                    <body name="{name}" pos="{pos}" quat="{quat}">
                        <freejoint name="{name}"/>
                        <geom name="{name}" type="{type}" size="{size}" density="{density}"
                            rgba="{rgba}" group="{group}"/>
                    </body>
                '''.format(**{k: convert(v) for k, v in object.items()}))
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])
        # Add mocaps to the XML dictionary
        for name, mocap in self.mocaps.items():
            # Mocap names are suffixed with 'mocap'
            assert mocap['name'] == name, f'Inconsistent {name} {object}'
            assert name.replace('mocap', 'obj') in self.objects, f'missing object for {name}'
            # Add the object to the world
            mocap = mocap.copy()  # don't modify original object
            mocap['quat'] = rot2quat(mocap['rot'])
            body = xmltodict.parse('''
                <body name="{name}" mocap="true">
                    <geom name="{name}" type="{type}" size="{size}" rgba="{rgba}"
                        pos="{pos}" quat="{quat}" contype="0" conaffinity="0" group="{group}"/>
                </body>
            '''.format(**{k: convert(v) for k, v in mocap.items()}))
            worldbody['body'].append(body['body'])
            # Add weld to equality list
            mocap['body1'] = name
            mocap['body2'] = name.replace('mocap', 'obj')
            weld = xmltodict.parse('''
                <weld name="{name}" body1="{body1}" body2="{body2}" solref=".02 1.5"/>
            '''.format(**{k: convert(v) for k, v in mocap.items()}))
            equality['weld'].append(weld['weld'])
        # Add geoms to XML dictionary
        for name, geom in self.geoms.items():
            assert geom['name'] == name, f'Inconsistent {name} {geom}'
            geom = geom.copy()  # don't modify original object
            geom['quat'] = rot2quat(geom['rot'])
            geom['contype'] = geom.get('contype', 1)
            geom['conaffinity'] = geom.get('conaffinity', 1)
            body = xmltodict.parse('''
                <body name="{name}" pos="{pos}" quat="{quat}">
                    <geom name="{name}" type="{type}" size="{size}" rgba="{rgba}" group="{group}"
                        contype="{contype}" conaffinity="{conaffinity}"/>
                </body>
            '''.format(**{k: convert(v) for k, v in geom.items()}))
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])

        # calculate how long is path to goal this time
        if 'goal_path0' in self.geoms.keys():
            gp_idxs = sorted([name.replace('goal_path', '') for name in self.geoms.keys() if name.startswith('goal_path')])
            length = int(gp_idxs[-1])
        else:
            length = 100

        # add goal path to XML dict, changes dynamically, therefore justa add a big number
        for i in range(length):
            name = 'goal_path' + str(i)
            geom = {'name': name, 'size': [0.25, 0.025, 0.01], 'pos': [6., 7.5, 0.02], 'rot': 0, 'type': 'cylinder', 'contype': 0, 'conaffinity': 0, 'group': 2, 'rgba': [0.3 , 0.7 , 0.3 , 0.25]}

            if name in self.geoms.keys():
                continue

            geom = geom.copy()  # don't modify original object
            geom['quat'] = rot2quat(geom['rot'])
            geom['contype'] = geom.get('contype', 1)
            geom['conaffinity'] = geom.get('conaffinity', 1)
            body = xmltodict.parse('''
                <body name="{name}" pos="{pos}" quat="{quat}">
                    <geom name="{name}" type="{type}" size="{size}" rgba="{rgba}" group="{group}"
                        contype="{contype}" conaffinity="{conaffinity}"/>
                </body>
            '''.format(**{k: convert(v) for k, v in geom.items()}))
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])

        # Instantiate simulator
        # print(xmltodict.unparse(self.xml, pretty=True))
        self.xml_string = xmltodict.unparse(self.xml)
        self.model = load_model_from_xml(self.xml_string)
        self.sim = MjSim(self.model)

        # Add render contexts to newly created sim
        if self.render_context is None and self.observe_vision:
            render_context = MjRenderContextOffscreen(self.sim, device_id=-1, quiet=True)
            render_context.vopt.geomgroup[:] = 1
            self.render_context = render_context

        if self.render_context is not None:
            self.render_context.update_sim(self.sim)

        # Recompute simulation intrinsics from new position
        self.sim.forward()

    def rebuild(self, config={}, state=True):
        ''' Build a new sim from a model if the model changed '''
        if state:
            old_state = self.sim.get_state()
        #self.config.update(deepcopy(config))
        #self.parse(self.config)
        self.parse(config)
        self.build()
        if state:
            self.sim.set_state(old_state)
        self.sim.forward()

    def reset(self, build=True):
        ''' Reset the world (sim is accessed through self.sim) '''
        if build:
            self.build()
        # set flag so that renderer knows to update sim
        self.update_viewer_sim = True

    def render(self, mode='human'):
        ''' Render the environment to the screen '''
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
            # Turn all the geom groups on
            self.viewer.vopt.geomgroup[:] = 1
            # Set camera if specified
            if mode == 'human':
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.type = const.CAMERA_FREE
            else:
                self.viewer.cam.fixedcamid = self.model.camera_name2id(mode)
                self.viewer.cam.type = const.CAMERA_FIXED
        if self.update_viewer_sim:
            self.viewer.update_sim(self.sim)
            self.update_viewer_sim = False
        self.viewer.render()

    def robot_com(self):
        ''' Get the position of the robot center of mass in the simulator world reference frame '''
        return self.body_com('robot')

    def robot_pos(self):
        ''' Get the position of the robot in the simulator world reference frame '''
        return self.body_pos('robot')

    def robot_mat(self):
        ''' Get the rotation matrix of the robot in the simulator world reference frame '''
        return self.body_mat('robot')

    def robot_vel(self):
        ''' Get the velocity of the robot in the simulator world reference frame '''
        return self.body_vel('robot')

    def body_com(self, name):
        ''' Get the center of mass of a named body in the simulator world reference frame '''
        return self.data.subtree_com[self.model.body_name2id(name)].copy()

    def body_pos(self, name):
        ''' Get the position of a named body in the simulator world reference frame '''
        return self.data.get_body_xpos(name).copy()

    def body_mat(self, name):
        ''' Get the rotation matrix of a named body in the simulator world reference frame '''
        return self.data.get_body_xmat(name).copy()

    def body_vel(self, name):
        ''' Get the velocity of a named body in the simulator world reference frame '''
        return self.data.get_body_xvelp(name).copy()



class Robot:
    ''' Simple utility class for getting mujoco-specific info about a robot '''
    def __init__(self, path):
        base_path = os.path.join(BASE_DIR, path)
        self.sim = MjSim(load_model_from_path(base_path))
        self.sim.forward()

        # Needed to figure out z-height of free joint of offset body
        self.z_height = self.sim.data.get_body_xpos('robot')[2]
        # Get a list of geoms in the robot
        self.geom_names = [n for n in self.sim.model.geom_names if n != 'floor']
        # Needed to figure out the observation spaces
        self.nq = self.sim.model.nq
        self.nv = self.sim.model.nv
        # Needed to figure out action space
        self.nu = self.sim.model.nu
        # Needed to figure out observation space
        # See engine.py for an explanation for why we treat these separately
        self.hinge_pos_names = []
        self.hinge_vel_names = []
        self.ballquat_names = []
        self.ballangvel_names = []
        self.sensor_dim = {}
        for name in self.sim.model.sensor_names:
            id = self.sim.model.sensor_name2id(name)
            self.sensor_dim[name] = self.sim.model.sensor_dim[id]
            sensor_type = self.sim.model.sensor_type[id]
            if self.sim.model.sensor_objtype[id] == const.OBJ_JOINT:
                joint_id = self.sim.model.sensor_objid[id]
                joint_type = self.sim.model.jnt_type[joint_id]
                if joint_type == const.JNT_HINGE:
                    if sensor_type == const.SENS_JOINTPOS:
                        self.hinge_pos_names.append(name)
                    elif sensor_type == const.SENS_JOINTVEL:
                        self.hinge_vel_names.append(name)
                    else:
                        t = self.sim.model.sensor_type[i]
                        raise ValueError('Unrecognized sensor type {} for joint'.format(t))
                elif joint_type == const.JNT_BALL:
                    if sensor_type == const.SENS_BALLQUAT:
                        self.ballquat_names.append(name)
                    elif sensor_type == const.SENS_BALLANGVEL:
                        self.ballangvel_names.append(name)
                elif joint_type == const.JNT_SLIDE:
                    # Adding slide joints is trivially easy in code,
                    # but this removes one of the good properties about our observations.
                    # (That we are invariant to relative whole-world transforms)
                    # If slide joints are added we sould ensure this stays true!
                    raise ValueError('Slide joints in robots not currently supported')

# TODO UNUSED
class Roads:
    """
    Roads are surrounded by hazards (getting negative reward for stepping on them).

    Algorithm 1:
    1. Specify the coordinates of the road intersections. The intersections should be on the same parallel or horizontal
    lines in order for the map to look more organized.
    2. Connect all the intersections with k closest intersections.
        - If k = 1, then it is a dead end.
        - If k = 2, then it is a road. Check the angle between the two closest points and our intersection.
        - If k = 3, then it is an intersection. Check the angle between the closest points.
    3. Fill the boarders of the roads with hazards (dangerous areas to avoid = pedestrian lanes).
    4. Fill the rest of the spaces with pillars (immobile obstacles = houses).

    Algorithm 2:
    If you have n different lines then those lines intersect in (n**2 - n)/2 different points ALMOST surely. Those are
    full intersections.
    ## In order to get other type of intersections, random points can be added and connected to our full intersections.

    1. Define two lines a1*x + b1*y + c1 = 0 and a2*x+ b2*y + c2 = 0 that intersect in our given world.
    2. Calculate the intersection of those lines
                    x0 = (b1*c2 - b2*c1)/(a1*b2 - a2*b1)
                    y0 = (a2*c1 - a1*c2)/(a1*b2 - a2*b1)
       and check that the intersection is inside of our defined world.
    3. Calculate the angle theta between those lines
                    tan(theta) = (a2-a1)/(1-a1*a2)
       and check that the angle is greater than some threshold (otherwise the curve is too sharp).
    4. Choose one of the lines defined in 1. and add a new random (and perpendicular for a more symmetric world)
      line a3*x + b3*y + c3 = 0 that intersect with the chosen line.
      Make sure that the constraints 2. and 3. are fulfilled.
    5. Repeat 4. until desired number certain intersections is reached in our world.

    Calculate all the intersections between all the lines an keep only those intersections (and lines) that are inside
    our defined world.
    """

    def __init__(self, start, end):
        self.start = start #[1, -3]
        self.end = end #[2, -1]

    def connect_two_points(self, start, end):
        """
        Takes two points as numpy 2D array and returns coordinates that connect those two points.

        Parameters:
        ends (numpy array): cooridnates of two points (e.g. start = [-1, 0], end = [2, 0]))

        Returns:
        numpy array: coordinates that connect those two points  (e.g. [[-1  0] [ 0  0] [ 1  0] [ 2  0]])
        """
        ends = np.array([start, end])
        d = np.diff(ends, axis=0)[0]  # gives the difference btw both arrays on an a vertical axis
        j = np.argmax(np.abs(d))  # returns the index of the max value in the previous vector (note we take abs)
        D = d[j]  # we get the value from the vector
        aD = np.abs(D)  # the absolute value of the maximum value
        # get's the slope in a way and floor divided it to get the change then add it to the first observation to get the path
        return ends[0] + (np.outer(np.arange(aD + 1), d) + (aD >> 1)) // aD

    def calculate_hazard_locations(self):
        path = self.connect_two_points(self.start, self.end)
        # requires a list of tuples
        hazard_loc = [(x, y) for x, y in path]
        return hazard_loc



class Graph(object):

    def __init__(self, graph_dict=None):
        """ initializes a graph object
            If no dictionary or None is given,
            an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1].append(vertex2)
        else:
            self.__graph_dict[vertex1] = [vertex2]

    def __generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one (a loop back to the vertex) or two
            vertices
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return

#
# if __name__ == "__main__":
#     g = {"a": ["d"],
#          "b": ["c"],
#          "c": ["b", "c", "d", "e"],
#          "d": ["a", "c"],
#          "e": ["c"],
#          "f": []
#          }
#
#     graph = Graph(g)
#
#     print("Vertices of graph:")
#     print(graph.vertices())
#
#     print("Edges of graph:")
#     print(graph.edges())
#
#     print("Add vertex:")
#     graph.add_vertex("z")
#
#     print("Vertices of graph:")
#     print(graph.vertices())
#
#     print("Add an edge:")
#     graph.add_edge({"a", "z"})
#
#     print("Vertices of graph:")
#     print(graph.vertices())
#
#     print("Edges of graph:")
#     print(graph.edges())
#
#     print('Adding an edge {"x","y"} with new vertices:')
#     graph.add_edge({"x", "y"})
#     print("Vertices of graph:")
#     print(graph.vertices())
#     print("Edges of graph:")
#     print(graph.edges())




