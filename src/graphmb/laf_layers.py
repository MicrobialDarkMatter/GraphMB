import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import BatchNormalization, Lambda
from tensorflow.keras.layers import Dropout, Layer

def laf_init_uniform(shape, dtype=None):
    a = tf.random.uniform(shape ,0,1, dtype=dtype)
    return a

def laf_init_normal(shape, dtype=None):
    b = tf.random.normal(shape, 1.0, 0.1, dtype=dtype)
    return b

class BiasLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias

class LAF(Layer):
    def __init__(self, units=4, eps=1e-7):
        super(LAF, self).__init__()
        self.units = units
        self.eps = eps
        
    def build(self, input_shape):

        self.b = self.add_weight(shape=(1, self.units, 1),
                                 initializer=laf_init_uniform,
                                 trainable=True, name='b')
        self.d = self.add_weight(shape=(1, self.units, 1),
                                 initializer=laf_init_uniform,
                                 trainable=True, name='d')
        self.f = self.add_weight(shape=(1, self.units, 1),
                                 initializer=laf_init_uniform,
                                 trainable=True, name='f')
        self.h = self.add_weight(shape=(1, self.units, 1),
                                 initializer=laf_init_uniform,
                                 trainable=True, name='h')

        self.a = self.add_weight(shape=(1, self.units, 1),
                                 initializer=laf_init_uniform,
                                 trainable=True, name='a')
        self.c = self.add_weight(shape=(1, self.units, 1),
                                 initializer=laf_init_uniform,
                                 trainable=True, name='c')
        self.e = self.add_weight(shape=(1, self.units, 1),
                                 initializer=laf_init_uniform,
                                 trainable=True, name='e')
        self.g = self.add_weight(shape=(1, self.units, 1),
                                 initializer=laf_init_uniform,
                                 trainable=True, name='g')
       
        self.alpha = self.add_weight(shape=(1, self.units, 1),
                                 initializer=laf_init_normal,
                                 trainable=True, name='alpha')
        self.beta  = self.add_weight(shape=(1, self.units, 1),
                                 initializer=laf_init_normal,
                                 trainable=True, name='beta')
        self.gamma = self.add_weight(shape=(1, self.units, 1),
                                 initializer=laf_init_normal,
                                 trainable=True, name='gamma')
        self.delta = self.add_weight(shape=(1, self.units, 1),
                                 initializer=laf_init_normal,
                                 trainable=True, name='delta')
    def call(self, inputs):
        eps = 1e-6
        adj, inputs = inputs

        x = inputs
        x  = tf.reshape(x, [-1, self.units, inputs.shape[-1]//self.units])
        sig_x  = tf.clip_by_value(tf.nn.sigmoid(x), eps, 1-eps)
        n_sig_x = 1.0 - sig_x

        x_b = tf.math.pow(sig_x, tf.nn.relu(self.b))
        x_d = tf.math.pow(n_sig_x, tf.nn.relu(self.d))
        x_f = tf.math.pow(sig_x, tf.nn.relu(self.f))
        x_h = tf.math.pow(n_sig_x, tf.nn.relu(self.h))

        x_b = tf.reshape(x_b, [-1, inputs.shape[-1]])
        x_d = tf.reshape(x_d, [-1, inputs.shape[-1]])
        x_f = tf.reshape(x_f, [-1, inputs.shape[-1]])
        x_h = tf.reshape(x_h, [-1, inputs.shape[-1]])

        x_b = tf.sparse.sparse_dense_matmul(adj, x_b) + eps
        x_d = tf.sparse.sparse_dense_matmul(adj, x_d) + eps
        x_f = tf.sparse.sparse_dense_matmul(adj, x_f) + eps
        x_h = tf.sparse.sparse_dense_matmul(adj, x_h) + eps

        x_b = tf.reshape(x_b, [-1, self.units, inputs.shape[-1]//self.units])
        x_d = tf.reshape(x_d, [-1, self.units, inputs.shape[-1]//self.units])
        x_f = tf.reshape(x_f, [-1, self.units, inputs.shape[-1]//self.units])
        x_h = tf.reshape(x_h, [-1, self.units, inputs.shape[-1]//self.units])

        x_ab = tf.math.pow(x_b,tf.nn.relu(self.a) ) * self.alpha
        x_cd = tf.math.pow(x_d,tf.nn.relu(self.c) ) * self.beta
        x_ef = tf.math.pow(x_f,tf.nn.relu(self.e) ) * self.gamma
        x_gh = tf.math.pow(x_h,tf.nn.relu(self.g) ) * self.delta

        den = (x_ef + x_gh)
        num = (x_ab + x_cd) * den

        out = num / (den * den + 1e-3)
        out = tf.reshape(out, (-1, inputs.shape[-1]))
        return out
    
    
# https://github.com/danielegrattarola/keras-gat/blob/master/keras_gat/graph_attention_layer.py
# https://www.programmersought.com/article/38174813555/

class GraphAttention(Layer):
    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 laf_units=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = tf.keras.activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.attn_kernel_initializer = tf.keras.initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = tf.keras.regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.attn_kernel_constraint = tf.keras.constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        self.laf_units = laf_units
        if self.laf_units is not None:
            self.laf_agg = LAF(laf_units)

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[1][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs, training=True):
        A = inputs[0]  # Adjacency matrix (N x N) sparse
        X = inputs[1]  # Node features (N x F)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = tf.matmul(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self   = tf.matmul(features, attention_kernel[0])  # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = tf.matmul(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            con_sa_1 = A * attn_for_self 
            con_sa_2 = A * tf.transpose(attn_for_neighs, [1,0])
            
            weights = tf.sparse.add(con_sa_1, con_sa_2)  # concatenation
            weights_act = tf.SparseTensor(indices=weights.indices,
                                          values=tf.nn.leaky_relu(weights.values, alpha=0.2),
                                          dense_shape=weights.dense_shape)
            
            attention = tf.sparse.softmax(weights_act)
            if training and self.dropout_rate > 0.0:
                attention = tf.SparseTensor(indices=attention.indices,
                                            values=tf.nn.dropout(attention.values, self.dropout_rate),
                                            dense_shape=attention.dense_shape)
            if training and self.dropout_rate > 0.0:
                features = tf.nn.dropout(features, self.dropout_rate)
            
            if self.laf_units is not None:
                node_features = self.laf_agg((attention, features))
            else:
                node_features = tf.sparse.sparse_dense_matmul(attention, features)
            
            if self.use_bias:
                node_features = node_features + self.biases[head]
                
            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        output = tf.concat(outputs, axis=1)  # (N x KF')
        if self.attn_heads_reduction != 'concat':
            output = tf.reshape(output, [-1, self.attn_heads, self.F_])
            output = tf.reduce_mean(output, axis=-2)

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape
