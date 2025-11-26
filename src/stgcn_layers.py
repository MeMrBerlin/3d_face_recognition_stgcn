import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class GraphConv(layers.Layer):
    """
    Spatial graph convolution with fixed normalized adjacency A (J,J).
    Input:  x (B, T, J, C_in)
    Output: y (B, T, J, C_out)
    y_t = (A @ x_t) W   (applied per time t and batch)
    """
    def __init__(self, out_channels, A_norm, use_bias=True, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.A = tf.constant(A_norm, dtype=tf.float32)  # (J,J)
        self.use_bias = use_bias

    def build(self, input_shape):
        # input_shape: (B, T, J, C_in)
        C_in = int(input_shape[-1])
        self.W = self.add_weight(
            shape=(C_in, self.out_channels),
            initializer="glorot_uniform",
            trainable=True,
            name="W"
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.out_channels,),
                initializer="zeros",
                trainable=True,
                name="b"
            )
        else:
            self.b = None

    def call(self, x):
        B, T, J, C = tf.unstack(tf.shape(x))
        # fuse batch and time -> (B*T, J, C)
        x_bt = tf.reshape(x, (B*T, J, C))
        # spatial mix: (B*T, J, C)
        x_sp = tf.matmul(self.A, x_bt)        # left-multiply on node dimension
        # channel mix: (B*T, J, C_out)
        x_sp = tf.tensordot(x_sp, self.W, axes=[[2],[0]])
        if self.b is not None:
            x_sp = x_sp + self.b
        # back to (B, T, J, C_out)
        return tf.reshape(x_sp, (B, T, J, self.out_channels))

#  class STGCNBlock(layers.Layer):
#     """
#     One ST-GCN block: GraphConv -> BN+ReLU -> TemporalConv1D -> BN+Dropout -> Residual -> ReLU
#     """
#     def __init__(self, out_channels, A_norm, stride_t=1, dropout=0.25, temporal_kernel=9, name=None):
#         super().__init__(name=name)
#         self.gcn = GraphConv(out_channels, A_norm, name=f"{name}_gcn")
#         self.bn1 = layers.BatchNormalization()
#         self.relu = layers.ReLU()
#         self.tconv = layers.Conv1D(
#             out_channels,
#             kernel_size=temporal_kernel,
#             strides=stride_t,
#             padding="same",
#             name=f"{name}_tconv"
#         )
#         self.bn2 = layers.BatchNormalization()
#         self.drop = layers.Dropout(dropout)
#         self.down = None
#         self.out_channels = out_channels
#         self.stride_t = stride_t

#     def build(self, input_shape):
#         in_channels = int(input_shape[-1])
#         if in_channels != self.out_channels or self.stride_t != 1:
#             # residual uses 1x1 temporal conv over time (pointwise over channels)
#             self.down = tf.keras.Sequential([
#                 layers.Conv1D(self.out_channels, kernel_size=1, strides=self.stride_t, padding="same"),
#                 layers.BatchNormalization()
#             ])
#         else:
#             self.down = None

#     def call(self, x, training=False):
#         # x: (B, T, J, C)
#         res = x
#         x = self.gcn(x)                           # (B, T, J, C_out)
#         x = self.bn1(x, training=training)
#         x = self.relu(x)
#         # temporal conv expects (B*J, T, C)
#         B, T, J, C = tf.unstack(tf.shape(x))
#         x_btj = tf.reshape(x, (B*J, T, C))
#         x_btj = self.tconv(x_btj)                 # (B*J, T', C_out)
#         x_btj = self.bn2(x_btj, training=training)
#         x_btj = self.drop(x_btj, training=training)
#         # back to (B, T', J, C_out)
#         Tprime = tf.shape(x_btj)[1]
#         Cout = tf.shape(x_btj)[2]
#         x = tf.reshape(x_btj, (B, Tprime, J, Cout))
#         # residual path
#         if self.down is not None:
#             res_btj = tf.reshape(res, (B*J, T, int(res.shape[-1])))
#             res_btj = self.down(res_btj, training=training)
#             res = tf.reshape(res_btj, (B, Tprime, J, int(res_btj.shape[-1])))
#         x = layers.add([x, res])
#         return self.relu(x)
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "out_channels": self.out_channels,
#             "A_norm": self.A_norm.tolist() if hasattr(self.A_norm, "tolist") else self.A_norm,
#             "stride_t": self.stride_t,
#             "dropout": self.drop,
#             "temporal_kernel": self.temporal_kernel,
#         })
#         return config


class STGCNBlock(layers.Layer):
    """
    Spatio-Temporal Graph Convolutional Network (ST-GCN) Block.
    Performs spatial graph convolution + temporal convolution.
    """

    def __init__(self, out_channels, A_norm, stride_t=1, dropout=0.5, temporal_kernel=9, **kwargs):
        super().__init__(**kwargs)

        # Store parameters for serialization
        self.out_channels = out_channels
        self.A_norm = A_norm
        self.stride_t = stride_t
        self.dropout = dropout
        self.temporal_kernel = temporal_kernel

        # Define layers
        self.spatial_conv = layers.Conv2D(
            filters=out_channels,
            kernel_size=(1, 1),
            use_bias=False
        )
        self.temporal_conv = layers.Conv2D(
            filters=out_channels,
            kernel_size=(temporal_kernel, 1),
            strides=(stride_t, 1),
            padding='same'
        )
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        # x shape: (batch, time, joints, channels)
        # Spatial graph conv: multiply adjacency A_norm
        x_spatial = tf.einsum('ntvc,vw->ntwc', x, self.A_norm)
        x_spatial = self.spatial_conv(x_spatial)

        # Temporal convolution
        x_temporal = self.temporal_conv(x_spatial)

        # Normalization + activation + dropout
        x_out = self.bn(x_temporal, training=training)
        x_out = self.relu(x_out)
        x_out = self.drop(x_out, training=training)
        return x_out

    def get_config(self):
        # Convert numpy arrays to lists for JSON serialization
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "A_norm": self.A_norm.tolist() if hasattr(self.A_norm, "tolist") else self.A_norm,
            "stride_t": self.stride_t,
            "dropout": self.dropout,
            "temporal_kernel": self.temporal_kernel
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        A_norm = np.array(config.pop("A_norm"))
        out_channels = config.pop("out_channels")
        return cls(out_channels=out_channels, A_norm=A_norm, **config)




def build_stgcn_model(num_joints, in_channels, num_classes, A_norm,
                      temporal_kernel=9, dropout=0.25):
    """
    Returns a Keras Model:
      Input:  (B, T, J, C_in)
      Output: (B, num_classes)
    """
    inp = layers.Input(shape=(None, num_joints, in_channels))  # T is None (flexible)
    x = layers.BatchNormalization()(inp)
    x = STGCNBlock(64,  A_norm, stride_t=1, dropout=dropout, temporal_kernel=temporal_kernel, name="blk1")(x)
    x = STGCNBlock(128, A_norm, stride_t=1, dropout=dropout, temporal_kernel=temporal_kernel, name="blk2")(x)
    x = STGCNBlock(256, A_norm, stride_t=1, dropout=dropout, temporal_kernel=temporal_kernel, name="blk3")(x)
    # global pooling over T and J
    x = layers.GlobalAveragePooling2D(data_format="channels_last")(x)  # pool over (T,J)
    out = layers.Dense(num_classes, activation=None)(x)
    return tf.keras.Model(inp, out, name="stgcn_tf")

