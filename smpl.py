import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class SMPL(keras.layers.Layer):
    def __init__(self, model_path, name="smpl", **kwargs):
        super(SMPL, self).__init__(name=name, **kwargs)

        with open(model_path, 'rb') as f:
            dd = pickle.load(f, encoding='latin1')

        self.num_shapes = dd['shapedirs'].shape[-1]
        self.num_vertices = dd["v_template"].shape[-2]
        self.num_faces = dd["f"].shape[-2]
        self.num_joints = dd["J_regressor"].shape[0]

        self.skinning_weights = tf.convert_to_tensor(
            value=dd["weights"],
            dtype=self.dtype,
            name="skinning_weights"
        )

        self.template_vertices = tf.convert_to_tensor(
            value=dd["v_template"],
            dtype=self.dtype,
            name="template_vertices"
        )

        self.faces = tf.convert_to_tensor(
            value=dd["f"],
            dtype=tf.int32,
            name="faces"
        )

        self.shapedirs = tf.convert_to_tensor(
            value=dd["shapedirs"].reshape([-1, self.num_shapes]).T,
            dtype=self.dtype,
            name="shapedirs"
        )

        self.posedirs = tf.convert_to_tensor(
            value=dd["posedirs"].reshape([-1, dd['posedirs'].shape[-1]]).T,
            dtype=self.dtype,
            name="posedirs"
        )

        self.joint_regressor = tf.convert_to_tensor(
            value=dd["J_regressor"].T.todense(),
            dtype=self.dtype,
            name="joint_regressor"
        )

        self.kintree_table = dd['kintree_table'][0].astype(np.int32)


    def call(self, shape=None, pose=None, translation=None):
        # Add shape blenshape
        shape_blendshape = tf.reshape(
            tensor=tf.matmul(shape, self.shapedirs),
            shape=[-1, self.num_vertices, 3],
            name="shape_blendshape"
        )

        vs = self.template_vertices + shape_blendshape  

        if pose is None:
            return vs, tf.zeros((self.num_joints, 4, 4))

        # Compute local joint locations and rotations
        pose = tf.reshape(pose, [-1, self.num_joints, 3])
        joint_rotations_local = AxisAngleToMatrix()(pose)
        joint_locations_local = tf.stack(
            values=[
                tf.matmul(vs[:, :, 0], self.joint_regressor),
                tf.matmul(vs[:, :, 1], self.joint_regressor),
                tf.matmul(vs[:, :, 2], self.joint_regressor)
            ],
            axis=2,
            name="joint_locations_local"
        )

        # Add pose blenshape
        pose_feature = tf.reshape(
            tensor=joint_rotations_local[:, 1:, :, :] - tf.eye(3),
            shape=[-1, 9 * (self.num_joints - 1)]
        )

        pose_blendshape = tf.reshape(
            tensor=tf.matmul(pose_feature, self.posedirs),
            shape=[-1, self.num_vertices, 3],
            name="pose_blendshape"
        )

        vp = vs + pose_blendshape

        # Compute global joint transforms
        joint_transforms, joint_locations = PoseSkeleton()(
            joint_rotations_local,
            joint_locations_local,
            self.kintree_table
        )

        # Apply linear blend skinning
        v = LBS()(vp, joint_transforms, self.skinning_weights)

        # Apply translation
        if translation is not None:
            v += translation[:, tf.newaxis, :]

        # Compute vertex normals
        n = VertexNormals()(v, self.faces)

        tensor_dict = {
            "shape_blendshape": shape_blendshape,
            "pose_blendshape": pose_blendshape,
            "pose_feature": pose_feature,
            "joint_transforms": joint_transforms,
            "joint_locations": joint_locations,
            "joint_locations_local": joint_locations_local,
            "vertices_shaped": vs,
            "vertices_posed": vp,
            "vertices": v,
            "vertex_normals": n
        }

        return v, tensor_dict


class AxisAngleToMatrix(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AxisAngleToMatrix, self).__init__(**kwargs)


    def call(self, axis_angle):
        """Converts rotations in axis-angle representation to rotation matrices

        Args:
            axis_angle: tensor of shape batch_size x 3

        Returns:
            rotation_matrix: tensor of shape batch_size x 3 x 3
        """
        initial_shape = tf.shape(axis_angle)

        axis_angle = tf.reshape(axis_angle, [-1, 3])
        batch_size = tf.shape(axis_angle)[0]

        angle = tf.expand_dims(tf.norm(axis_angle + 1e-8, axis=1), -1)
        axis = tf.expand_dims(tf.math.divide(axis_angle, angle), -1)

        angle = tf.expand_dims(angle, -1)
        cos = tf.cos(angle)
        sin = tf.sin(angle)

        outer = tf.matmul(axis, axis, transpose_b=True, name="outer")

        eyes = tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
        R = cos * eyes + (1 - cos) * outer + sin * Skew()(axis)
        R = tf.reshape(R, tf.concat([initial_shape, [3]], axis=0))

        return R


class Skew(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Skew, self).__init__(**kwargs)


    def call(self, vec):
        """Returns the skew symetric version of each 3x3 matrix in a vector

        Args:
            vec: tensor of shape batch_size x 3

        Returns:
            rotation_matrix: tensor of shape batch_size x 3 x 3
        """
        batch_size = tf.shape(vec)[0]
        
        col_inds = tf.constant([1, 2, 3, 5, 6, 7])
        batch_inds = tf.reshape(tf.range(0, batch_size) * 9, [-1, 1])
        indices = tf.reshape(batch_inds + col_inds, [-1, 1])

        updates = tf.stack(
            values=[-vec[:, 2], vec[:, 1], vec[:, 2],
                    -vec[:, 0], -vec[:, 1], vec[:, 0]],
            axis=1
        )
        updates = tf.reshape(updates, [-1])
                
        out_shape = [batch_size * 9]
        res = tf.scatter_nd(indices, updates, out_shape)
        res = tf.reshape(res, [batch_size, 3, 3])

        return res


class PoseSkeleton(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PoseSkeleton, self).__init__(**kwargs)


    def call(self, joint_rotations, joint_positions, parents):
        """
        Computes absolute joint locations given pose.

        Args:
            joint_rotations: batch_size x K x 3 x 3 rotation vector of K joints
            joint_positions: batch_size x K x 3, joint locations before posing
            parents: vector of size K holding the parent id for each joint

        Returns
            joint_transforms: `Tensor`: batch_size x K x 4 x 4 relative joint transformations for LBS.
            joint_positions_posed: batch_size x K x 3, joint locations after posing
        """
        batch_size = tf.shape(joint_rotations)[0]
        num_joints = len(parents)

        def make_affine(rotation, translation, name=None):
            '''
            Args:
                rotation: batch_size x 3 x 3
                translation: batch_size x 3 x 1
            '''
            rotation_homo = tf.pad(rotation, [[0, 0], [0, 1], [0, 0]])
            translation_homo = tf.concat([translation, tf.ones([batch_size, 1, 1])], 1)
            affine_transform = tf.concat([rotation_homo, translation_homo], 2)
            return affine_transform

        joint_positions = tf.expand_dims(joint_positions, axis=-1)      
        root_rotation = joint_rotations[:, 0, :, :]
        root_transform = make_affine(root_rotation, joint_positions[:, 0])

        # Traverse joints to compute global transformations
        transforms = [root_transform]
        for joint, parent in enumerate(parents[1:], start=1):
            position = joint_positions[:, joint] - joint_positions[:, parent]
            transform_local = make_affine(joint_rotations[:, joint], position)
            transform_global = tf.matmul(transforms[parent], transform_local)
            transforms.append(transform_global)
        transforms = tf.stack(transforms, axis=1) 

        # Extract joint positions
        joint_positions_posed = transforms[:, :, :3, 3]

        # Compute affine transforms relative to initial state (i.e., t-pose)
        zeros = tf.zeros([batch_size, num_joints, 1, 1])
        joint_rest_positions = tf.concat([joint_positions, zeros], axis=2)
        init_bone = tf.matmul(transforms, joint_rest_positions)
        init_bone = tf.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
        joint_transforms = transforms - init_bone

        return joint_transforms, joint_positions_posed


class LBS(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LBS, self).__init__(**kwargs)


    def call(self, vertices, joint_rotations, skinning_weights):
        skinning_weights = tf.convert_to_tensor(skinning_weights, self.dtype)

        batch_size = tf.shape(vertices)[0]
        num_joints = skinning_weights.shape[-1]
        num_vertices = vertices.shape[-2]
   
        W = skinning_weights
        if len(skinning_weights.shape.as_list()) < len(vertices.shape.as_list()):
            W = tf.tile(tf.convert_to_tensor(skinning_weights), [batch_size, 1])
            W = tf.reshape(W, [batch_size, skinning_weights.shape[-2], num_joints])    
 
        A = tf.reshape(joint_rotations, (-1, num_joints, 16))
        T = tf.matmul(W, A)
        T = tf.reshape(T, (-1, num_vertices, 4, 4))

        ones = tf.ones([batch_size, num_vertices, 1])
        vertices_homo = tf.concat([vertices, ones], axis=2)
        skinned_homo = tf.matmul(T, tf.expand_dims(vertices_homo, -1))
        skinned_vertices = skinned_homo[:, :, :3, 0]

        return skinned_vertices


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


class VertexNormals(keras.layers.Layer):
    def __init__(self, normalize=True, **kwargs):
        super(VertexNormals, self).__init__(**kwargs)
        self.normalize = normalize


    def call(self, vertices, faces):
        batch_size = tf.shape(vertices)[0]

        if not tf.is_tensor(faces):
            faces = tf.convert_to_tensor(faces, dtype=tf.int32)

        faces_flat = tf.reshape(faces, [-1])
        faces_tiled = tf.tile(faces_flat, [batch_size])
        faces = tf.reshape(faces_tiled, [batch_size] + faces.shape.as_list())
    
        shape_faces = faces.shape.as_list()
        mesh_face_normals = FaceNormals(normalize=False, dtype=self.dtype)(vertices, faces)


        outer_indices = tf.range(batch_size, dtype=tf.int32)
        outer_indices = tf.expand_dims(outer_indices, axis=-1)
        outer_indices = tf.expand_dims(outer_indices, axis=-1)
        outer_indices = tf.tile(outer_indices, [1] * len(shape_faces[:-2]) +
                                [tf.shape(input=faces)[-2]] + [1])

        vertex_normals = tf.zeros_like(vertices)
        for i in range(shape_faces[-1]):
            scatter_indices = tf.concat(
                [outer_indices, faces[..., i:i + 1]], axis=-1)

            vertex_normals = tf.compat.v1.tensor_scatter_add(
                vertex_normals, scatter_indices, mesh_face_normals)

        if self.normalize:
            vertex_normals = tf.math.l2_normalize(vertex_normals, axis=-1)

        return vertex_normals