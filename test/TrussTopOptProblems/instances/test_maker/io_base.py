import os
import json
import datetime
import warnings
from collections import defaultdict, OrderedDict

'''length scale conversion to meter'''
LENGTH_SCALE_CONVERSION = {
    'millimeter': 1e-3,
    'meter': 1.0,
}

#############################################

def parse_point(json_point, scale=1.0):
    return [scale*json_point[0], scale*json_point[1], scale*json_point[2]]

def parse_nodes(node_data, scale=1.0):
    nodes = []
    for n in node_data:
        n['point'] = parse_point(n['point'], scale=scale)
        nodes.append(Node.from_data(n))
    return nodes

def extract_model_name_from_path(file_path):
    forename = file_path.split('.json')[0]
    return forename.split(os.sep)[-1]

#############################################

class Model(object):
    def __init__(self, nodes, elements, supports, joints, materials, crosssecs, unit='meter', model_name=None):
        self.model_name = model_name
        self.generate_time = str(datetime.datetime.now())
        assert unit in LENGTH_SCALE_CONVERSION
        self.unit = unit
        scale = LENGTH_SCALE_CONVERSION[unit]
        if scale != 1.0:
            warnings.warn('Model unit {}, scaled by {} and converted to Meter unit.'.format(unit, scale))
        for node in nodes:
            node.point = [scale*cp for cp in node.point]

        self.nodes = nodes
        self.elements = elements

        # turn lists into dicts
        self.supports = {}
        for support in supports:
            self.supports[support.node_ind] = support

        self.joints = {}
        if joints is not None:
            for joint in joints:
                for e_tag in joint.elem_tags:
                    if e_tag in self.joints:
                        warnings.warn('Multiple joints assigned to the same element tag |{}|!'.format(e_tag))
                    self.joints[e_tag] = joint

        self.materials = {}
        for mat in materials:
            for e_tag in mat.elem_tags:
                if e_tag in self.materials:
                    warnings.warn('Multiple materials assigned to the same element tag |{}|!'.format(e_tag))
                self.materials[e_tag] = mat

        self.crosssecs = {}
        for cs in crosssecs:
            for e_tag in cs.elem_tags:
                if e_tag in self.crosssecs:
                    warnings.warn('Multiple materials assigned to the same element tag |{}|!'.format(e_tag))
                self.crosssecs[e_tag] = cs

    @property
    def node_num(self):
        return len(self.nodes)

    @property
    def element_num(self):
        return len(self.elements)

    @classmethod
    def from_json(cls, file_path, verbose=False):
        assert os.path.exists(file_path) and "json file path does not exist!"
        with open(file_path, 'r') as f:
            json_data = json.loads(f.read())
        if 'model_name' not in json_data:
            json_data['model_name']  = extract_model_name_from_path(file_path)
        return cls.from_data(json_data, verbose)

    @classmethod
    def from_data(cls, data, verbose=False):
        model_name = data['model_name']
        unit = data['unit']
        # length scale for vertex positions
        scale = LENGTH_SCALE_CONVERSION[unit]
        unit = 'meter'
        nodes = parse_nodes(data['nodes'], scale=scale)
        elements = [Element.from_data(e) for e in data['elements']]
        element_inds_from_tag = defaultdict(list)
        for e in elements:
            element_inds_from_tag[e.elem_tag].append(e.elem_ind)
        supports = [Support.from_data(s) for s in data['supports']]
        joints = [Joint.from_data(j) for j in data['joints']]
        materials = [Material.from_data(m) for m in data['materials']]
        crosssecs = [CrossSec.from_data(c) for c in data['cross_secs']]
        # sanity checks
        if 'node_num' in data:
            assert len(nodes) == data['node_num']
        if 'element_num' in data:
            assert len(elements) == data['element_num']
        node_id_range = list(range(len(nodes)))
        for e in elements:
            assert e.end_node_inds[0] in node_id_range and \
                   e.end_node_inds[1] in node_id_range, \
                   'element end point id not in node_list id range!'
        num_grounded_nodes = 0
        for n in nodes:
            if n.is_grounded:
                num_grounded_nodes += 1
        # assert grounded_nodes > 0, 'The structure must have at lease one grounded node!'
        if num_grounded_nodes == 0:
            warnings.warn('The structure must have at lease one grounded node!')
        if verbose:
            print('Model: {} | Original unit: {} | Generated time: {}'.format(model_name, data['unit'], 
                data['generate_time'] if 'generate_time' in data else ''))
            print('Nodes: {} | Elements: {} | Supports: {} | Joints: {} | Materials: {} | Cross Secs: {} | Tag Ground: {} '.format(
                len(nodes), len(elements), len(supports), len(joints), len(materials), len(crosssecs), num_grounded_nodes))
        return cls(nodes, elements, supports, joints, materials, crosssecs, model_name=model_name)

    def to_data(self):
        data = OrderedDict()
        data['model_name'] = self.model_name
        data['unit'] = self.unit
        data['generate_time'] = self.generate_time
        data['node_num'] = len(self.nodes)
        data['element_num'] = len(self.elements)
        data['nodes'] = [n.to_data() for n in self.nodes]
        data['elements'] = [e.to_data() for e in self.elements]
        data['supports'] = [s.to_data() for s in self.supports.values()]
        data['joints'] = [j.to_data() for j in self.joints.values()]
        data['materials'] = [m.to_data() for m in self.materials.values()]
        data['cross_secs'] = [cs.to_data() for cs in self.crosssecs.values()]
        return data

class LoadCase(object):
    def __init__(self, point_loads=None, uniform_element_loads=None, gravity_load=None):
        self.point_loads = point_loads or []
        self.uniform_element_loads = uniform_element_loads or []
        self.gravity_load = gravity_load

    @classmethod
    def from_json(cls, file_path, lc_ind=0):
        assert os.path.exists(file_path) and "json file path does not exist!"
        with open(file_path, 'r') as f:
            json_data = json.loads(f.read())
        json_data = json_data['loadcases'] if 'loadcases' in json_data else json_data
        return cls.from_data(json_data[str(lc_ind)])
    
    @classmethod
    def from_data(cls, data):
        point_loads = [PointLoad.from_data(pl) for pl in data['ploads']]
        uniform_element_loads = [UniformlyDistLoad.from_data(el) for el in data['eloads']]
        gravity_load = None if 'gravity' not in data else data['gravity']
        return cls(point_loads, uniform_element_loads, gravity_load)

    def to_data(self):
        data = {'ploads' : [pl.to_data() for pl in self.point_loads],
                'eloads' : [el.to_data() for el in self.uniform_element_loads],
                'gravity' : self.gravity_load.to_data() if self.gravity_load is not None else None
                }
        return data

    def __repr__(self):
        return '{}(#pl:{},#el:{},gravity:{})'.format(self.__class__.__name__, len(self.point_loads), 
            len(self.uniform_element_loads), self.gravity_load)

##############################################

class Node(object):
    def __init__(self, point, node_ind, is_grounded):
        self.point = point
        self.node_ind = node_ind
        self.is_grounded = is_grounded

    @classmethod
    def from_data(cls, data):
        return cls(data['point'], data['node_ind'], data['is_grounded'])

    def to_data(self):
        data = {'point' : list(self.point), 'node_ind' : self.node_ind, 'is_grounded' : self.is_grounded}
        return data

    def __repr__(self):
        return '{}(#{},{},Grd:{})'.format(self.__class__.__name__, self.node_ind, self.point, self.is_grounded)

class Element(object):
    def __init__(self, end_node_inds, elem_ind, elem_tag='', bending_stiff=True):
        assert end_node_inds[0] != end_node_inds[1], 'zero length element not allowed!'
        self.end_node_inds = end_node_inds
        self.elem_tag = elem_tag
        self.elem_ind = elem_ind
        self.bending_stiff = bending_stiff

    @classmethod
    def from_data(cls, data):
        return cls(data['end_node_inds'], data['elem_ind'], data['elem_tag'], data['bending_stiff'])

    def to_data(self):
        data = {'end_node_inds' : self.end_node_inds,
                'elem_ind' : self.elem_ind,
                'elem_tag' : self.elem_tag,
                'bending_stiff' : self.bending_stiff,
                }
        return data

    def __repr__(self):
        return '{}(#{}({}),{},Bend:{})'.format(self.__class__.__name__, self.elem_ind, self.elem_tag, self.end_node_inds, self.bending_stiff)

class Support(object):
    def __init__(self, condition, node_ind):
        self.condition = condition
        self.node_ind = node_ind

    @classmethod
    def from_data(cls, data):
        return cls(data['condition'], data['node_ind'])

    def to_data(self):
        return {
            'condition' : self.condition,
            'node_ind' : self.node_ind,
        }

    def __repr__(self):
        return '{}(#{},{})'.format(self.__class__.__name__, self.node_ind, self.condition)

class Joint(object):
    def __init__(self, c_conditions, elem_tags):
        assert len(c_conditions) == 12
        self.c_conditions = c_conditions
        self.elem_tags = elem_tags

    @classmethod
    def from_data(cls, data):
        return cls(data['c_conditions'], data['elem_tags'])

    def to_data(self):
        return {
            'c_conditions' : self.c_conditions,
            'elem_tags' : self.elem_tags,
        }

# TODO: assumed to be converted to meter-based unit
class CrossSec(object):
    def __init__(self, A, Jx, Iy, Iz, elem_tags=None, family='unnamed', name='unnamed'):
        self.A = A
        self.Jx = Jx
        self.Iy = Iy
        self.Iz = Iz
        self.elem_tags = elem_tags if elem_tags else [None]
        self.family = family
        self.name = name

    @classmethod
    def from_data(cls, data):
        return cls(data['A'], data['Jx'], data['Iy'], data['Iz'], data['elem_tags'], data['family'], data['name'])

    def to_data(self):
        return {
            'A' : self.A,
            'Jx' : self.Jx,
            'Iy' : self.Iy,
            'Iz' : self.Iz,
            'elem_tags' : self.elem_tags,
            'family' : self.family,
            'name' : self.name,
        }

    def __repr__(self):
        return '{}(family:{} name:{} area:{}[m2] Jx:{}[m4] Iy:{}[m4] Iz:{}[m4] applies to elements:{})'.format(
            self.__class__.__name__, self.family, self.name, self.A, self.Jx, self.Iy, self.Iz, self.elem_tags)

def mu2G(mu, E):
    """compute shear modulus from poisson ratio mu and Youngs modulus E
    """
    return E/(2*(1+mu))

def G2mu(G, E):
    """compute poisson ratio from shear modulus and Youngs modulus E
    """
    return E/(2*G)-1

# TODO: assumed to be converted to kN/m2, kN/m3
class Material(object):
    def __init__(self, E, G12, fy, density, elem_tags=None, family='unnamed', name='unnamed', type_name='ISO', G3=None):
        self.E = E
        # in-plane shear modulus 
        self.G12 = G12
        # transverse shear modulus
        self.G3 = G3 or G12
        self.mu = G2mu(G12, E)
        # material strength in the specified direction (local x direction)
        self.fy = fy
        self.density = density
        self.elem_tags = elem_tags if elem_tags else [None]
        self.family = family
        self.name = name
        self.type_name = type_name

    @classmethod
    def from_data(cls, data):
        return cls(data['E'], data['G12'], data['fy'], data['density'], data['elem_tags'], data['family'], data['name'], data['type_name'], data.get('G3', None))

    def to_data(self):
        return  {
            'E' : self.E,
            'G12' : self.G12,
            'G3' : self.G3,
            'mu' : self.mu,
            'fy' : self.fy,
            'density' : self.density,
            'elem_tags' : self.elem_tags,
            'family' : self.family,
            'name' : self.name,
            'type_name' : self.type_name,
        }

    def __repr__(self):
        # G3:8076[kN/cm2]
        return '{}(|{}| E:{}[kN/m2] mu:{} G12:{}[kN/m2] G3:{}[kN/m2] density:{}[kN/m3] fy:{}[kN/m2] applies to elements:{})'.format(
            self.__class__.__name__, self.family+'-'+self.name, self.E, self.mu, self.G12, self.G3, self.density, self.fy, self.elem_tags)

class PointLoad(object):
    def __init__(self, force, moment, node_ind, loadcase=0):
        self.force = force
        self.moment = moment
        self.node_ind = node_ind
        self.loadcase = loadcase

    @classmethod
    def from_data(cls, data):
        lc_ind = 0 if 'loadcase' not in data else data['loadcase']
        return cls(data['force'], data['moment'], data['node_ind'], lc_ind)

    def to_data(self):
        return {
            'force' : self.force,
            'moment' : self.moment,
            'node_ind' : self.node_ind,
            'loadcase' : self.loadcase,
        }

    def __repr__(self):
        return '{}(node_ind {} | force {} | moment {} | lc#{})'.format(self.__class__.__name__, self.node_ind, self.force, self.moment, self.loadcase)

class UniformlyDistLoad(object):
    def __init__(self, q, load, elem_tags, loadcase=0):
        self.q = q
        # not sure if load is used in Karamba, we only use q here
        self.load = load
        self.elem_tags = elem_tags
        self.loadcase = loadcase

    @classmethod
    def from_data(cls, data):
        lc_ind = 0 if 'loadcase' not in data else data['loadcase']
        return cls(data['q'], data['load'], data['elem_tags'], lc_ind)

    def to_data(self):
        return {
            'q' : self.q,
            'load' : self.load,
            'elem_tags' : self.elem_tags,
            'loadcase' : self.loadcase,
        }

    def __repr__(self):
        return '{}(element_tags {} | q {} | load {} | lc#{})'.format(self.__class__.__name__, 
            self.elem_tags, self.q, self.load, self.loadcase)

class GravityLoad(object):
    def __init__(self, force=[0,0,-1], loadcase=0):
        self.force = force
        self.loadcase = loadcase

    @classmethod
    def from_data(cls, data):
        lc_ind = 0 if 'loadcase' not in data else data['loadcase']
        return cls(data['force'], lc_ind)

    def to_data(self):
        return {
            'force' : self.force,
            'loadcase' : self.loadcase,
        }

    def __repr__(self):
        return '{}({}, lc#{})'.format(self.__class__.__name__, self.force, self.loadcase)

##########################################

class AnalysisResult(object):
    def __init__(self):
        pass