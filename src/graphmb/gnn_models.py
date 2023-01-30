import tensorflow as tf

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Softmax, Embedding, LayerNormalization
#from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Lambda, LeakyReLU, Flatten
from tensorflow.keras.layers import Dropout, Add, Concatenate
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
from tensorflow.keras.regularizers import l2    

from graphmb.layers import BiasLayer, LAF, GraphAttention

class VGAE(Model):
    def __init__(self, emb_dim, embeddings=None, 
                 hidden_dim1=None, hidden_dim2=None,
                 dropout=None, l2_reg=None,
                 freeze_embeddings=False, lr=1e-02):
        
        super(VGAE, self).__init__()
        
        N = emb_dim[0] # Number of nodes in the graph
        F = emb_dim[1] # Original size of node features
        self.freeze_embeddings = freeze_embeddings

        x_in = Input(shape=(1,), dtype=tf.int64)
        a_in = Input((N,), dtype=tf.float32)
        if embeddings is not None:
            x = Embedding(N, F, weights=[embeddings], 
                          trainable=not freeze_embeddings)(x_in)
        else:
            x = Embedding(N, F, trainable=not freeze_embeddings)(x_in)
        x = Flatten()(x)
        x_orig = x
        
#         for _ in range(2):
#             x = Dense(256)(x)
#             x = LeakyReLU()(x)
#             x = BatchNormalization(epsilon=1e-3)(x)
        

        x = GCNConv( hidden_dim1,
                     activation=None,
                     kernel_regularizer=l2(l2_reg))([x, a_in])
        x = LeakyReLU()(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(dropout)(x)

        z_mean = GCNConv(
            hidden_dim2,
            activation=None,
            kernel_regularizer=l2(l2_reg),
        )([x, a_in])
        
        z_log_std = GCNConv(
            hidden_dim2,
            activation=None,
            kernel_regularizer=l2(l2_reg),
        )([x, a_in])
        
        self.encoder = Model([x_in, a_in], [z_mean, z_log_std, x_orig])
        self.encoder.build([ (None,1), (None,N) ])
        
        z_in = Input(shape=(hidden_dim2,))
        x = z_in
        for _ in range(2):
            x = Dense(256)(x)
            x = LeakyReLU()(x)
            x = BatchNormalization(epsilon=1e-6)(x)
        #x = Dense(emb_dim[1])(x)
        
        
        self.decoder = Model(z_in, x)
        self.decoder.build((None, hidden_dim2))
        
        self.optimizer = Adam(learning_rate=lr)#, clipnorm=1.0)
    
    def call(self, x, indices=None, training=True):
        x,a = x
        z_mean, z_log_std, x_orig = self.encoder((x,a), training=training)
        #z_log_std = tf.nn.softplus(z_log_std)
        z_log_std = tf.clip_by_value(z_log_std, -2, 2)
        z_sample = tf.random.normal(tf.shape(z_mean)) * tf.exp(z_log_std)
        if training:
            z = z_mean + z_sample
            z = tf.gather(z, indices)
            out = tf.matmul(z, tf.transpose(z))
            out = tf.nn.sigmoid(out)
        else:
            z = z_mean
            out = None
        
        return out, z, z_mean, z_log_std, x_orig
    
    def train_step(self, x, a, y, pos_weight, norm, indices):
        loss = self._train_step(x, a, y, pos_weight, norm, indices)
        return loss.numpy()
    
    @tf.function
    def _train_step(self, x, a, y, pos_weight, norm, indices, loss="graph"): #loss="features"
        with tf.GradientTape() as tape:
            predictions, z, model_z_mean, model_z_log_std, x_orig = self((x, a), indices, training=True)            
            pairs = []
            for i in indices:
                for j in indices:
                    pairs.append((i,j))
            pairs = tf.convert_to_tensor(pairs, dtype=tf.int32)
            y = tf.gather_nd(a, pairs)
            y = tf.reshape(y, [-1])
            predictions = tf.reshape(predictions, [-1])
            rec_loss = norm*tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=predictions, labels=y, pos_weight=pos_weight))

            # latent loss
            kl_loss = (0.5 / tf.cast(tf.shape(x)[-1], tf.float32)) * tf.reduce_mean(tf.reduce_sum(
            1. + 2. * model_z_log_std - tf.square(model_z_mean) - tf.square(tf.exp(model_z_log_std)), 1
            ))
            
            loss = rec_loss - kl_loss + tf.reduce_sum(self.encoder.losses)
            ## Add reconstruction loss
            if self.freeze_embeddings:
                x_hat = self.decoder(z, training=True)
                loss = loss + 0.5*tf.reduce_mean((x_hat - tf.gather(x_orig, indices))**2)
                
        tw = self.encoder.trainable_weights
        if self.freeze_embeddings:
            tw += self.decoder.trainable_weights
        gradients = tape.gradient(loss,tw)
        self.optimizer.apply_gradients(zip(gradients, tw))
        return loss

class GCN(Model):
    def __init__(
        self,
        features_shape,
        input_dim,
        labels,
        adj,
        embsize=None,
        n_labels=None,
        hidden_units=None,
        layers=None,
        conv_last=None,
        use_bn=True,
        use_vae=False,
        predict=False,
        dropout=0.5
    ):
        super(GCN, self).__init__()
        #assert layers > 0
        self.features_shape = features_shape
        self.labels = labels
        self.adj = adj
        self.adj_size = adj.dense_shape.numpy()

        adj_in = Input(shape=self.adj_size[1:], batch_size=self.adj_size[0], dtype=tf.float32, sparse=True)
        node_in = Input(shape=input_dim, batch_size=self.adj_size[0], dtype=tf.float32)

        x = node_in
        # first gcn layer
        for l in range(layers):
            x = Dense(hidden_units, use_bias=(l == 0))(x)
            x = Lambda(lambda t: tf.sparse.sparse_dense_matmul(t[0], t[1]))([adj_in, x])
            x = BiasLayer()(x)
            if use_bn:
                x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            x = Activation("relu")(x)
            x = Dropout(dropout)(x)

        if use_vae:
            mu = Dense(embsize, use_bias=True)(x)
            log_std = Dense(embsize, use_bias=True)(x)
            x = Concatenate(axis=1)([mu, log_std])
        else:
            x = Dense(embsize, use_bias=True)(x)
            if predict:
                x = Dense(n_labels, use_bias=False)(x)
                x = Softmax()(x)
                #x = Lambda(lambda t: tf.sparse.sparse_dense_matmul(t[0], t[1]))([adj_in, x])
                #x = BiasLayer()(x)

        self.model = Model([node_in, adj_in], x)
        self.model.build([(features_shape[0], input_dim), tuple(self.adj_size)])

    def call(self, features, idx, training=True):
        output = self.model((features, self.adj), training=training)
        if idx is None:
            return output
        else:
            return tf.gather(output, idx)

    def summary(self):
        self.model.summary()


class GCNLAF(Model):
    def __init__(self, features_shape, input_dim, labels, adj,
                 n_labels=None, hidden_units=None, embsize=None,
                 layers=None, conv_last=None, dropout=0.5):
        super(GCNLAF, self).__init__()
        #assert layers > 0
        self.features_shape = features_shape
        self.labels = labels
        self.adj = adj
        self.adj_size = adj.dense_shape.numpy()

        adj_in = Input(shape=self.adj_size[1:], batch_size=self.adj_size[0], dtype=tf.float32, sparse=True)
        node_in = Input(shape=input_dim, batch_size=self.adj_size[0], dtype=tf.float32)

        x = node_in
        # first gcn layer
        for l in range(layers):
            x = Dense(hidden_units, use_bias=(l == 0))(x)
            x = LAF(4)([adj_in, x])
            x = BiasLayer()(x)
            x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            x = Activation("relu")(x)
            x = Dropout(dropout)(x)

        # third gcn layer
        if conv_last:
            x = Dense(embsize, use_bias=False)(x)
            x = LAF(4)([adj_in, x])
            x = BiasLayer()(x)
        else:
            x = Dense(embsize, use_bias=True)(x)

        self.model = Model([node_in, adj_in], x)
        self.model.build([(self.features_shape[0], input_dim), tuple(self.adj_size)])

    def call(self, features, idx, training=True):
        output = self.model((features, self.adj), training=training)
        if idx is None:
            return output
        else:
            return tf.gather(output, idx)

    def summary(self):
        self.model.summary()


class GAT(Model):
    def __init__(
        self, features_shape, input_dim, labels, adj, embsize=None,
        hidden_units=None, layers=None, conv_last=None, dropout=0.5
    ):
        super(GAT, self).__init__()
        self.features_shape = features_shape
        self.labels = labels
        self.adj = adj
        self.adj_size = adj.dense_shape.numpy()

        adj_in = Input(shape=self.adj_size[1:], batch_size=self.adj_size[0], dtype=tf.float32, sparse=True)
        node_in = Input(shape=input_dim, batch_size=self.adj_size[0], dtype=tf.float32)

        x = node_in
        for l in range(layers):
            x = GraphAttention(hidden_units, dropout_rate=0.1, attn_heads=1, activation="linear")([adj_in, x])
            x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            x = Activation("relu")(x)
            x = Dropout(dropout)(x)

        # second sage layer
        if conv_last:
            x = GraphAttention(
                embsize, dropout_rate=0.1, attn_heads=1, activation="linear", attn_heads_reduction="average"
            )([adj_in, x])
        else:
            x = Dense(embsize, use_bias=True)(x)

        self.model = Model([node_in, adj_in], x)
        self.model.build([(features_shape[0], input_dim), tuple(self.adj_size)])

    def call(self, features, idx, training=True):
        output = self.model((features, self.adj), training=training)
        if idx is None:
            return output
        else:
            return tf.gather(output, idx)

    def summary(self):
        self.model.summary()


class GATLAF(Model):
    def __init__(self, features, labels, adj, n_labels=None, embsize=None,
                hidden_units=None, layers=None, conv_last=None):
        super(GATLAF, self).__init__()
        self.features = features
        self.labels = labels
        self.adj = adj
        self.adj_size = adj.dense_shape.numpy()

        adj_in = Input(shape=self.adj_size[1:], batch_size=self.adj_size[0], dtype=tf.float32, sparse=True)
        node_in = Input(shape=features.shape[1:], batch_size=self.adj_size[0], dtype=tf.float32)

        x = node_in
        for l in range(layers):
            x = GraphAttention(hidden_units, dropout_rate=0.1, attn_heads=1, activation="linear", laf_units=4)(
                [adj_in, x]
            )
            x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            x = Activation("relu")(x)
            x = Dropout(0.5)(x)

        # second sage layer
        if conv_last:
            x = GraphAttention(
                embsize,
                dropout_rate=0.1,
                attn_heads=1,
                activation="linear",
                attn_heads_reduction="average",
                laf_units=4,
            )([adj_in, x])
        else:
            x = Dense(embsize, use_bias=True)(x)

        self.model = Model([node_in, adj_in], x)
        self.model.build([tuple(self.features.shape), tuple(self.adj_size)])

    def call(self, idx, training=True):
        output = self.model((self.features, self.adj), training=training)
        if idx is None:
            return output
        else:
            return tf.gather(output, idx)

    def summary(self):
        self.model.summary()


class SAGE(Model):
    def __init__(
        self, features_shape, input_dim, labels, adj, embsize=None,
        n_labels=None, hidden_units=None, layers=None, conv_last=None,
        dropout=0.5
    ):
        super(SAGE, self).__init__()
        self.features_shape = features_shape
        self.labels = labels
        self.adj = adj
        self.adj_size = adj.dense_shape.numpy()

        adj_in = Input(shape=self.adj_size[1:], batch_size=self.adj_size[0], dtype=tf.float32, sparse=True)
        node_in = Input(shape=input_dim, batch_size=self.adj_size[0], dtype=tf.float32)

        x = node_in
        # first sage layer
        for l in range(layers):
            x_node = Dense(hidden_units, use_bias=(l == 0))(x)
            x_neigh = Dense(hidden_units, use_bias=(l == 0))(x)
            x = Lambda(lambda t: tf.sparse.sparse_dense_matmul(t[0], t[1]))([adj_in, x_neigh])
            x = BiasLayer()(x)
            x = Add()([x_node, x])
            x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            x = Activation("relu")(x)
            x = Dropout(dropout)(x)

        # second sage layer
        if conv_last:
            x_node = Dense(embsize, use_bias=False)(x)
            x_neigh = Dense(embsize, use_bias=False)(x)
            x = Lambda(lambda t: tf.sparse.sparse_dense_matmul(t[0], t[1]))([adj_in, x_neigh])
            x = BiasLayer()(x)
            x = Add()([x_node, x])
        else:
            x = Dense(embsize, use_bias=True)(x)

        self.model = Model([node_in, adj_in], x)
        self.model.build([(self.features_shape[0], input_dim), tuple(self.adj_size)])

    def call(self, features, idx, training=True):
        output = self.model((features, self.adj), training=training)
        if idx is None:
            return output
        else:
            return tf.gather(output, idx)

    def summary(self):
        self.model.summary()


class SAGELAF(Model):
    def __init__(self, features_shape, labels, input_dim, adj,
                n_labels=None, embsize=None, hidden_units=None,
                layers=None, conv_last=None, dropout=0.5):
        super(SAGELAF, self).__init__()
        self.features_shape = features_shape
        self.labels = labels
        self.adj = adj
        self.adj_size = adj.dense_shape.numpy()

        adj_in = Input(shape=self.adj_size[1:], batch_size=self.adj_size[0], dtype=tf.float32, sparse=True)
        node_in = Input(shape=input_dim, batch_size=self.adj_size[0], dtype=tf.float32)

        x = node_in
        # first sage layer
        for l in range(layers):
            x_node = Dense(hidden_units, use_bias=(l == 0))(x)
            x_neigh = Dense(hidden_units, use_bias=(l == 0))(x)
            x = LAF(4)([adj_in, x_neigh])
            x = BiasLayer()(x)
            x = Add()([x_node, x])
            x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            x = Activation("relu")(x)
            x = Dropout(dropout)(x)

        # second sage layer
        if conv_last:
            x_node = Dense(embsize, use_bias=False)(x)
            x_neigh = Dense(embsize, use_bias=False)(x)
            x = LAF(4)([adj_in, x_neigh])
            x = BiasLayer()(x)
            x = Add()([x_node, x])
        else:
            x = Dense(embsize, use_bias=True)(x)

        self.model = Model([node_in, adj_in], x)
        self.model.build([(self.features_shape[0], input_dim),
                           tuple(self.adj_size)])

    def call(self, features, idx, training=True):
        output = self.model((features, self.adj), training=training)
        if idx is None:
            return output
        else:
            return tf.gather(output, idx)

    def summary(self):
        self.model.summary()

name_to_model = {"SAGE": SAGE, "SAGELAF": SAGELAF, "GCN": GCN, "GCNLAF": GCNLAF, "GAT": GAT, "GATLAF": GATLAF}
