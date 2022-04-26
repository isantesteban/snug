import numpy as np
import tensorflow as tf


def pairwise_distance(A, B):
    rA = np.sum(np.square(A), axis=1)
    rB = np.sum(np.square(B), axis=1)
    distances = - 2*np.matmul(A, np.transpose(B)) + rA[:, np.newaxis] + rB[np.newaxis, :]
    return distances


def find_nearest_neighbour(A, B, dtype=np.int32):
    nearest_neighbour = np.argmin(pairwise_distance(A, B), axis=1)
    return nearest_neighbour.astype(dtype)


def get_vertex_connectivity(faces, dtype=tf.int32):
    '''
    Returns a list of unique edges in the mesh. 
    Each edge contains the indices of the vertices it connects
    '''
    if tf.is_tensor(faces):
        faces = faces.numpy()

    edges = set()
    for f in faces:
        num_vertices = len(f)
        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            edges.add(tuple(sorted([f[i], f[j]])))

    return tf.convert_to_tensor(list(edges), dtype)


def get_face_connectivity(faces, dtype=tf.int32):
    '''
    Returns a list of adjacent face pairs
    '''
    if tf.is_tensor(faces):
        faces = faces.numpy()

    edges = get_vertex_connectivity(faces).numpy()

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]

    adjacent_faces = []
    for key in G:
        assert len(G[key]) < 3
        if len(G[key]) == 2:
            adjacent_faces += [G[key]]
   
    return tf.convert_to_tensor(adjacent_faces, dtype)


def get_face_connectivity_edges(faces, dtype=tf.int32):
    '''
    Returns a list of edges that connect two faces
    (i.e., all the edges except borders)
    '''
    if tf.is_tensor(faces):
        faces = faces.numpy()

    edges = get_vertex_connectivity(faces).numpy()

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]

    adjacent_face_edges = []
    for key in G:
        assert len(G[key]) < 3
        if len(G[key]) == 2:
            adjacent_face_edges += [list(key)]

    return tf.convert_to_tensor(adjacent_face_edges, dtype)


def get_vertex_mass(vertices, faces, density, dtype=tf.float32):
    '''
    Computes the mass of each vertex according to triangle areas and fabric density
    '''
    areas = get_face_areas(vertices, faces)
    triangle_masses = density * areas

    vertex_masses = np.zeros(vertices.shape[0])
    np.add.at(vertex_masses, faces[:,0], triangle_masses/3)
    np.add.at(vertex_masses, faces[:,1], triangle_masses/3)
    np.add.at(vertex_masses, faces[:,2], triangle_masses/3)

    return tf.convert_to_tensor(vertex_masses, dtype)


def get_face_areas(vertices, faces):
    if tf.is_tensor(vertices):
        vertices = vertices.numpy()

    if tf.is_tensor(faces):
        faces = faces.numpy()

    v0 = vertices[faces[:,0]]
    v1 = vertices[faces[:,1]]
    v2 = vertices[faces[:,2]]

    u = v2 - v0
    v = v1 - v0

    return np.linalg.norm(np.cross(u, v), axis=-1) / 2.0


def get_edge_length(vertices, edges):
    v0 = tf.gather(vertices, edges[:,0], axis=-2) 
    v1 = tf.gather(vertices, edges[:,1], axis=-2) 
    return tf.linalg.norm(v0 - v1, axis=-1)
 

def get_shape_matrix(x):
    if x.shape.ndims == 3:
        return tf.stack([x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], axis=-1)

    elif x.shape.ndims == 4:
        return tf.stack([x[:, :, 0] - x[:, :, 2], x[:, :, 1] - x[:, :, 2]], axis=-1)

    raise NotImplementedError
    

def gather_triangles(vertices, indices):
    if vertices.shape.ndims == (indices.shape.ndims + 1):
        indices = tf.repeat([indices], tf.shape(vertices)[0], axis=0)

    triangles = tf.gather(vertices, indices,
                          axis=-2,
                          batch_dims=vertices.shape.ndims - 2)

    return triangles


def finite_diff(x, h, keepdims=True):
    dx = (x[:, 1:] - x[:, :-1]) / h 

    if keepdims:
        shape = tf.shape(x)
        zeros = tf.zeros([shape[0], 1, shape[-1]], x.dtype)
        dx = tf.concat([zeros, dx], axis=1) 

    return dx
    

def finite_diff_np(x, h, diff=1):
    if diff == 0:
        return x

    v = np.zeros(x.shape, dtype=x.dtype)
    v[1:] = (x[1:] - x[0:-1]) / h

    return finite_diff_np(v, h, diff-1)


def compute_vertex_normals(vertices, faces):
    # Vertex normals weighted by triangle areas:
    # http://www.iquilezles.org/www/articles/normals/normals.htm

    normals = np.zeros(vertices.shape, dtype=vertices.dtype)
    triangles = vertices[faces]

    e1 = triangles[::, 0] - triangles[::, 1]
    e2 = triangles[::, 2] - triangles[::, 1]
    n = np.cross(e2, e1) 

    np.add.at(normals, faces[:,0], n)
    np.add.at(normals, faces[:,1], n)
    np.add.at(normals, faces[:,2], n)

    return normalize(normals)


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms


def fix_collisions(vc, vb, nb, eps=0.002):
    """
    Fix the collisions between the clothing and the body by projecting
    the clothing's vertices outside the body's surface
    """

    # For each vertex of the cloth, find the closest vertices in the body's surface
    closest_vertices = find_nearest_neighbour(vc, vb)
    vb = vb[closest_vertices] 
    nb = nb[closest_vertices] 

    # Test penetrations
    penetrations = np.sum(nb*(vc - vb), axis=1) - eps
    penetrations = np.minimum(penetrations, 0)

    # Fix the clothing
    corrective_offset = -np.multiply(nb, penetrations[:,np.newaxis])
    vc_fixed = vc + corrective_offset

    return vc_fixed

