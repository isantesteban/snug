import tensorflow as tf
import tensorflow.keras as keras


class FaceNormals(keras.layers.Layer):
    def __init__(self, normalize=True, **kwargs):
        super(FaceNormals, self).__init__(**kwargs)
        self.normalize = normalize

    def call(self, vertices, faces):
        v = vertices
        f = faces

        if v.shape.ndims == (f.shape.ndims + 1):
            f = tf.tile([f], [tf.shape(v)[0], 1, 1])   

        # Warning: tf.gather is prone to memory problems
        triangles = tf.gather(v, f, axis=-2, batch_dims=v.shape.ndims - 2) 

        # Compute face normals
        v0, v1, v2 = tf.unstack(triangles, axis=-2)
        e1 = v0 - v1
        e2 = v2 - v1
        face_normals = tf.linalg.cross(e2, e1) 

        if self.normalize:
            face_normals = tf.math.l2_normalize(face_normals, axis=-1)

        return face_normals


class PairwiseDistance(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PairwiseDistance, self).__init__(**kwargs)

    def call(self, A, B):
        rA = tf.reduce_sum(tf.square(A), axis=-1)
        rB = tf.reduce_sum(tf.square(B), axis=-1)
        transpose_axes = [0, 2, 1] 
        distances = - 2*tf.matmul(A, tf.transpose(B, transpose_axes)) + rA[:, :, tf.newaxis] + rB[:, tf.newaxis, :]
        return distances


class NearestNeighbour(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NearestNeighbour, self).__init__(**kwargs)

    def call(self, A, B):
        distances = PairwiseDistance(dtype=self.dtype)(A, B)
        nearest_neighbour = tf.argmin(distances, axis=-1)
        return tf.cast(nearest_neighbour, dtype=tf.int32)
