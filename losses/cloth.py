import numpy as np
import tensorflow as tf

from utils import *


class Cloth(): 
    '''
    This class stores mesh and material information of the garment
    '''
    
    def __init__(self, path, material, dtype=tf.float32):
        self.dtype = dtype
        self.material = material

        v, f, vm, fm = load_obj(path, tex_coords=True)

        v = tf.convert_to_tensor(v, dtype)
        f = tf.convert_to_tensor(f, tf.int32)
        vm = tf.convert_to_tensor(vm, dtype)
        fm = tf.convert_to_tensor(fm, tf.int32)

        # Vertex attributes
        self.v_template = v
        self.v_mass = get_vertex_mass(v, f, self.material.density, dtype)
        self.v_velocity = tf.zeros((1, v.shape[0], 3), dtype) # Vertex velocities in global coordinates
        self.v = tf.zeros((1, v.shape[0], 3), dtype) # Vertex position in global coordinates
        self.v_psd = tf.zeros((1, v.shape[0], 3), dtype) # Pose space deformation of each vertex
        self.v_weights = None # Vertex skinning weights
        self.num_vertices = self.v_template.shape[0]
    
        # Face attributes
        self.f = f
        self.f_connectivity = get_face_connectivity(f) # Pairs of adjacent faces
        self.f_connectivity_edges = get_face_connectivity_edges(f) # Edges that connect faces
        self.f_area = tf.convert_to_tensor(get_face_areas(v, f), self.dtype)
        self.num_faces = self.f.shape[0]

        # Edge attributes
        self.e = get_vertex_connectivity(f) # Pairs of connected vertices
        self.e_rest = get_edge_length(v, self.e) # Rest lenght of the edges (world space)
        self.num_edges = self.e.shape[0]

        # Rest state of the cloth (computed in material space)
        tri_m = gather_triangles(vm, fm)
        self.Dm = get_shape_matrix(tri_m)
        self.Dm_inv = tf.linalg.inv(self.Dm)

    def compute_skinning_weights(self, smpl):
        self.closest_body_vertices = find_nearest_neighbour(self.v_template, smpl.template_vertices)
        self.v_weights = tf.gather(smpl.skinning_weights, self.closest_body_vertices).numpy()
        self.v_weights = tf.convert_to_tensor(self.v_weights, dtype=self.dtype)



def load_obj(filename, tex_coords=False):
    vertices = []
    faces = []
    uvs = []
    faces_uv = []

    with open(filename, 'r') as fp:
        for line in fp:
            line_split = line.split()
            
            if not line_split:
                continue

            elif tex_coords and line_split[0] == 'vt':
                uvs.append([line_split[1], line_split[2]])

            elif line_split[0] == 'v':
                vertices.append([line_split[1], line_split[2], line_split[3]])

            elif line_split[0] == 'f':
                vertex_indices = [s.split("/")[0] for s in line_split[1:]]
                faces.append(vertex_indices)

                if tex_coords:
                    uv_indices = [s.split("/")[1] for s in line_split[1:]]
                    faces_uv.append(uv_indices)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) - 1

    if tex_coords:
        uvs = np.array(uvs, dtype=np.float32)
        faces_uv = np.array(faces_uv, dtype=np.int32) - 1
        return vertices, faces, uvs, faces_uv

    return vertices, faces