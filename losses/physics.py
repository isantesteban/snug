import tensorflow as tf

from utils import *
from layers import *


def deformation_gradient(triangles, Dm_inv):
    Ds = get_shape_matrix(triangles)
    return Ds @ Dm_inv 


def green_strain_tensor(F):
    I = tf.eye(2, dtype=F.dtype)
    Ft = tf.transpose(F, perm=[0, 1, 3, 2])
    return 0.5*(Ft @ F - I)


def stretching_energy(v, cloth, return_average=True): 
    '''
    Computes strech energy of the cloth for the vertex positions v
    Material model: Saint-Venant–Kirchhoff (StVK)
    Reference: ArcSim (physics.cpp)
    '''

    batch_size = tf.cast(tf.shape(v)[0], v.dtype)
    triangles = gather_triangles(v, cloth.f)

    Dm_inv = tf.repeat([cloth.Dm_inv], tf.shape(v)[0], axis=0)

    F = deformation_gradient(triangles, Dm_inv)
    G = green_strain_tensor(F)

    # Energy
    mat = cloth.material
    I = tf.eye(2, batch_shape=tf.shape(G)[:2], dtype=G.dtype)
    S = mat.lame_mu * G + 0.5 * mat.lame_lambda * tf.linalg.trace(G)[:, :, tf.newaxis, tf.newaxis] * I
    energy_density = tf.linalg.trace(tf.transpose(S, [0, 1, 3, 2]) @ G)
    energy = cloth.f_area[tf.newaxis] * mat.thickness * energy_density

    if return_average:
        return tf.reduce_sum(energy) / batch_size
    
    return tf.reduce_sum(energy, axis=-1)


def bending_energy(v, cloth, return_average=True): 
    '''
    Computes the bending energy of the cloth for the vertex positions v
    Reference: ArcSim (physics.cpp)
    '''

    batch_size = tf.cast(tf.shape(v)[0], v.dtype)

    # Compute face normals
    fn = FaceNormals(dtype=v.dtype)(v, cloth.f)
    n0 = tf.gather(fn, cloth.f_connectivity[:, 0], axis=1)
    n1 = tf.gather(fn, cloth.f_connectivity[:, 1], axis=1)

    # Compute edge lenght
    v0 = tf.gather(v, cloth.f_connectivity_edges[:, 0], axis=1)
    v1 = tf.gather(v, cloth.f_connectivity_edges[:, 1], axis=1)
    e = v1 - v0
    e_norm, l = tf.linalg.normalize(e, axis=-1)

    # Compute area
    f_area = tf.repeat([cloth.f_area], tf.shape(v)[0], axis=0)
    a0 = tf.gather(f_area, cloth.f_connectivity[:, 0], axis=1)
    a1 = tf.gather(f_area, cloth.f_connectivity[:, 1], axis=1)
    a = a0 + a1

    # Compute dihedral angle between faces
    cos = tf.reduce_sum(tf.multiply(n0, n1), axis=-1)
    sin = tf.reduce_sum(tf.multiply(e_norm, tf.linalg.cross(n0, n1)), axis=-1)
    theta = tf.math.atan2(sin, cos)
    # theta = tf.math.acos(cos)
    
    # Compute bending coefficient according to material parameters,
    # triangle areas (a) and edge length (l)
    mat = cloth.material
    scale = l[..., 0]**2 / (4*a)

    # Bending energy
    energy = mat.bending_coeff * scale * (theta ** 2) / 2

    if return_average:
        return tf.reduce_sum(energy) / batch_size

    return tf.reduce_sum(energy, axis=-1)


def gravitational_energy(x, mass, g=9.81, return_average=True):
    batch_size = tf.cast(tf.shape(x)[0], x.dtype)
    U = g * mass[tf.newaxis, tf.newaxis] * x[:, :, 1]

    if return_average:
        return tf.reduce_sum(U) / batch_size

    return tf.reduce_sum(U, axis=-1)


def inertial_term(x, x_prev, v_prev, mass, time_step, return_average=True):
    batch_size = tf.cast(tf.shape(x)[0], x.dtype)
    
    x_hat = x_prev + time_step * v_prev
    x_diff = x - x_hat

    num = tf.einsum('bvi,bvi->bv', x_diff, mass[:, tf.newaxis] * x_diff)
    den = 2 * time_step ** 2

    if return_average:
        return tf.reduce_sum(num / den) / batch_size

    return tf.reduce_sum(num / den, axis=-1)


def inertial_term_sequence(x, mass, time_step, return_average=True):
    """
    x: tf.Tensor of shape [batch_size, num_frames, num_vertices, 3]
    """
    batch_size = tf.cast(tf.shape(x)[0], x.dtype)
    num_vertices = tf.shape(x)[-2]

    # Compute velocities
    x_current = x[:, 1:]
    x_prev = x[:, :-1] 
    v = (x_current - x_prev) / time_step
    zeros = tf.zeros([batch_size, 1, num_vertices, 3], x.dtype)
    v_prev = tf.concat([zeros, v[:, :-1]], axis=1)   

    # Flatten
    x_current = tf.reshape(x_current, [-1, num_vertices, 3])   
    x_prev = tf.reshape(x_prev, [-1, num_vertices, 3])   
    v_prev = tf.reshape(v_prev, [-1, num_vertices, 3])   

    return inertial_term(x_current, x_prev, v_prev, mass, time_step, return_average)


def collision_penalty(va, vb, nb, eps=1e-3):
    batch_size = tf.cast(tf.shape(va)[0], va.dtype)

    closest_vertices = NearestNeighbour(dtype=va.dtype)(va, vb)
    vb = tf.gather(vb, closest_vertices, batch_dims=1)
    nb = tf.gather(nb, closest_vertices, batch_dims=1)

    distance = tf.reduce_sum(nb*(va - vb), axis=-1) 
    interpenetration = tf.maximum(eps - distance, 0)

    return tf.reduce_sum(interpenetration**3) / batch_size


if __name__ == "__main__":
    from cloth import Cloth
    from material import Material

    # Fabric material parameters
    thickness = 0.00047 # (m)
    bulk_density = 426  # (kg / m3)
    area_density = thickness * bulk_density

    material = Material(
        density=area_density, # Fabric density (kg / m2)
        thickness=thickness,  # Fabric thickness (m)
        young_modulus=0.7e5, 
        poisson_ratio=0.485,
        stretch_multiplier=1,
        bending_multiplier=50
    )

    print(f"Lame mu {material.lame_mu:.2E}, Lame lambda: {material.lame_lambda:.2E}")

    # Initialize structs
    cloth = Cloth(
        path="assets/meshes/tshirt.obj",
        material=material,
        dtype=tf.float32
    )

    # Example of how to call the functions
    batch_size = 128
    num_frames = 3
    dummy_input = tf.reshape(cloth.v_template, (1, 1, -1, 3))
    dummy_input = tf.tile(dummy_input, (batch_size, num_frames, 1, 1))
    dummy_input_flat = tf.reshape(dummy_input, (batch_size * num_frames, -1, 3))

    energy_stretch = stretching_energy(
        v=dummy_input_flat,
        cloth=cloth,
    )

    energy_bend = bending_energy(
        v=dummy_input_flat,
        cloth=cloth
    )

    energy_gravity = gravitational_energy(
        x=dummy_input_flat,
        mass=cloth.v_mass
    )

    inertia = inertial_term_sequence(
        x=dummy_input,
        mass=cloth.v_mass,
        time_step=1/30
    )

    print(f"Strech energy: {energy_stretch:.2E}")
    print(f"Bending energy: {energy_bend:.2E}")
    print(f"Gravity: {energy_gravity:.2E}")
    print(f"Inertia: {inertia:.2E}")