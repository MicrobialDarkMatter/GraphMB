import tensorflow as tf

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Lambda
from tensorflow.keras.layers import Dropout, Layer, Add, Concatenate

from graphmb.layers import BiasLayer, LAF, GraphAttention


class TH:
    def __init__(
        self,
        input_features,
        model,
        lr=0.01,
        lambda_vae=0.1,
        all_different_idx=None,
        all_same_idx=None,
        use_ae=False,
        latentdim=32,
        gnn_weight=1.0,
        kmer_dim=136,
        kmer_alpha=0.5,
        num_negatives=50
    ):
        self.opt = Adam(learning_rate=lr, epsilon=1e-8)
        # self.opt = SGD(learning_rate=lr)
        self.lambda_vae = lambda_vae
        self.model = model
        self.features = input_features
        # self.dense_adj = tf.sparse.to_dense(self.model.adj)
        self.adj_shape = self.model.adj.dense_shape
        self.kmer_dim = kmer_dim
        self.kmer_alpha = kmer_alpha
        self.num_negatives = num_negatives
        S = tf.cast(tf.reduce_sum(self.model.adj.values), tf.float32)
        s0 = tf.cast(self.adj_shape[0], tf.float32)

        self.all_different_idx = all_different_idx
        self.all_same_idx = all_same_idx
        self.pos_weight = (s0 * s0 - S) / S
        self.norm = s0 * s0 / ((s0 * s0 - S) * 2.0)
        self.use_ae = use_ae
        if self.use_ae:
            self.features = input_features
            self.encoder = tf.keras.Sequential(
                [
                    Dense(256, activation="relu", name="encoder1"),
                    BatchNormalization(),
                    Dense(latentdim, activation=None, name="encoder2"),
                ]
            )
            self.decoder = tf.keras.Sequential(
                [
                    Dense(256, activation="relu", name="decoder1"),
                    BatchNormalization(),
                    Dense(input_features.shape[1], activation=None, name="decoder2"),
                ]
            )
            self.encoder.build(self.features.shape)
            self.decoder.build((self.features.shape[0], latentdim))
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.gnn_weight = gnn_weight

    @tf.function
    def train_unsupervised(self, idx):
        with tf.GradientTape() as tape:
            #
            # breakpoint()
            if self.use_ae:
                features = self.encoder(self.features)
            else:
                features = self.features
            node_hat = self.model(features, idx)
            if self.use_ae:
                recon_features = self.decoder(node_hat)
                # assert recon_features.shape == self.features.shape
                #breakpoint()
                kmer_recon = recon_features[:, : self.kmer_dim]
                abund_recon = recon_features[:, self.kmer_dim:]
                kmer_loss = self.mse_loss(self.features[:, :self.kmer_dim], kmer_recon)
                abund_loss = self.mse_loss(self.features[:, self.kmer_dim:], abund_recon)
                recon_loss = (self.kmer_alpha / float(self.kmer_dim)) * kmer_loss + (1.0 - self.kmer_alpha) * abund_loss
            else:
                recon_loss = 0

            diff_loss = 0
            same_loss = 0

            row_embs = tf.gather(indices=self.model.adj.indices[:, 0], params=node_hat)
            col_embs = tf.gather(indices=self.model.adj.indices[:, 1], params=node_hat)
            positive_pairwise = tf.reduce_sum(tf.math.multiply(row_embs, col_embs), axis=1)

            neg_idx = tf.random.uniform(
                shape=(self.num_negatives * len(self.model.adj.indices),),
                minval=0,
                maxval=self.adj_shape[0] * self.adj_shape[1] - 1,
                dtype=tf.int64,
            )
            neg_idx_row = tf.math.minimum(
                tf.cast(self.adj_shape[0] - 1, tf.float32),
                tf.cast(neg_idx, tf.float32) / tf.cast(self.adj_shape[1], tf.float32),
            )
            neg_idx_row = tf.cast(neg_idx_row, tf.int64)[:, None]
            neg_idx_col = tf.cast(tf.math.minimum(self.adj_shape[1] - 1, (neg_idx % self.adj_shape[1])), tf.int64)[
                :, None
            ]
            neg_idx = tf.concat((neg_idx_row, neg_idx_col), axis=-1)
            try:
                #negative_pairs = tf.gather_nd(pairwise_similarity, neg_idx)
                neg_row_embs = tf.gather(indices=neg_idx_row, params=node_hat)
                neg_col_embs = tf.gather(indices=neg_idx_col, params=node_hat)
                negative_pairs = tf.reduce_sum(tf.math.multiply(neg_row_embs, neg_col_embs), axis=1)
            except:
                breakpoint()

            pos_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.ones_like(positive_pairwise), positive_pairwise, from_logits=True
                )
            )
            neg_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.zeros_like(negative_pairs), negative_pairs, from_logits=True)
            )
            gnn_loss = 0.5 * (pos_loss + neg_loss) #* self.gnn_weight 
            loss =  gnn_loss + recon_loss

            if self.all_different_idx is not None:
                ns1 = tf.gather(node_hat, self.all_different_idx[:, 0])
                ns2 = tf.gather(node_hat, self.all_different_idx[:, 1])
                all_diff_pairs = tf.math.exp(-0.5 * tf.reduce_sum((ns1 - ns2) ** 2, axis=-1))
                diff_loss = tf.reduce_mean(all_diff_pairs) * 100
                loss = loss + diff_loss
            if self.all_same_idx is not None:
                ns1 = tf.gather(node_hat, self.all_same_idx[:, 0])
                ns2 = tf.gather(node_hat, self.all_same_idx[:, 1])
                all_same_pairs = tf.math.exp(-0.5 * tf.reduce_sum((ns1 - ns2) ** 2, axis=-1))
                same_loss = -tf.reduce_mean(all_same_pairs)
                loss = loss + same_loss
        tw = self.model.trainable_weights
        grads = tape.gradient(loss, tw)
        self.opt.apply_gradients(zip(grads, tw))
        return gnn_loss, recon_loss, diff_loss

    @staticmethod
    def sample_idx(idx, n):
        n = tf.cast(n, tf.int64)
        N = tf.cast(tf.shape(idx)[0], tf.int64)
        random_idx = tf.random.uniform(shape=(n,), minval=0, maxval=N, dtype=tf.int64)
        s_idx = tf.gather_nd(idx, random_idx)
        return s_idx


class GCN(Model):
    def __init__(
        self,
        features_shape,
        input_dim,
        labels,
        adj,
        n_labels=None,
        hidden_units=None,
        layers=None,
        conv_last=None,
        use_bn=True,
        use_vae=False,
    ):
        super(GCN, self).__init__()
        assert layers > 0
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
            x = Dropout(0.5)(x)

        if use_vae:
            mu = Dense(n_labels, use_bias=True)(x)
            log_std = Dense(n_labels, use_bias=True)(x)
            x = Concatenate(axis=1)([mu, log_std])
        else:

            if conv_last:
                x = Dense(n_labels, use_bias=False)(x)
                x = Lambda(lambda t: tf.sparse.sparse_dense_matmul(t[0], t[1]))([adj_in, x])
                x = BiasLayer()(x)
            else:
                x = Dense(n_labels, use_bias=True)(x)

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
    def __init__(self, features, labels, adj, n_labels=None, hidden_units=None, layers=None, conv_last=None):
        super(GCNLAF, self).__init__()
        assert layers > 0
        self.features = features
        self.labels = labels
        self.adj = adj
        self.adj_size = adj.dense_shape.numpy()

        adj_in = Input(shape=self.adj_size[1:], batch_size=self.adj_size[0], dtype=tf.float32, sparse=True)
        node_in = Input(shape=features.shape[1:], batch_size=self.adj_size[0], dtype=tf.float32)

        x = node_in
        # first gcn layer
        for l in range(layers):
            x = Dense(hidden_units, use_bias=(l == 0))(x)
            x = LAF(4)([adj_in, x])
            x = BiasLayer()(x)
            x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            x = Activation("relu")(x)
            x = Dropout(0.5)(x)

        # third gcn layer
        if conv_last:
            x = Dense(n_labels, use_bias=False)(x)
            x = LAF(4)([adj_in, x])
            x = BiasLayer()(x)
        else:
            x = Dense(n_labels, use_bias=True)(x)

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


class GAT(Model):
    def __init__(
        self, features, input_dim, labels, adj, n_labels=None, hidden_units=None, layers=None, conv_last=None
    ):
        super(GAT, self).__init__()
        self.features = features
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
            x = Dropout(0.5)(x)

        # second sage layer
        if conv_last:
            x = GraphAttention(
                n_labels, dropout_rate=0.1, attn_heads=1, activation="linear", attn_heads_reduction="average"
            )([adj_in, x])
        else:
            x = Dense(n_labels, use_bias=True)(x)

        self.model = Model([node_in, adj_in], x)
        self.model.build([(self.features.shape[0], input_dim), tuple(self.adj_size)])

    def call(self, idx, training=True):
        output = self.model((self.features, self.adj), training=training)
        if idx is None:
            return output
        else:
            return tf.gather(output, idx)

    def summary(self):
        self.model.summary()


class GATLAF(Model):
    def __init__(self, features, labels, adj, n_labels=None, hidden_units=None, layers=None, conv_last=None):
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
                n_labels,
                dropout_rate=0.1,
                attn_heads=1,
                activation="linear",
                attn_heads_reduction="average",
                laf_units=4,
            )([adj_in, x])
        else:
            x = Dense(n_labels, use_bias=True)(x)

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
        self, features, input_dim, labels, adj, n_labels=None, hidden_units=None, layers=None, conv_last=None
    ):
        super(SAGE, self).__init__()
        self.features = features
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
            x = Dropout(0.5)(x)

        # second sage layer
        if conv_last:
            x_node = Dense(n_labels, use_bias=False)(x)
            x_neigh = Dense(n_labels, use_bias=False)(x)
            x = Lambda(lambda t: tf.sparse.sparse_dense_matmul(t[0], t[1]))([adj_in, x_neigh])
            x = BiasLayer()(x)
            x = Add()([x_node, x])
        else:
            x = Dense(n_labels, use_bias=True)(x)

        self.model = Model([node_in, adj_in], x)
        self.model.build([(self.features.shape[0], input_dim), tuple(self.adj_size)])

    def call(self, idx, training=True):
        output = self.model((self.features, self.adj), training=training)
        if idx is None:
            return output
        else:
            return tf.gather(output, idx)

    def summary(self):
        self.model.summary()


class SAGELAF(Model):
    def __init__(self, features, labels, adj, n_labels=None, hidden_units=None, layers=None, conv_last=None):
        super(SAGELAF, self).__init__()
        self.features = features
        self.labels = labels
        self.adj = adj
        self.adj_size = adj.dense_shape.numpy()

        adj_in = Input(shape=self.adj_size[1:], batch_size=self.adj_size[0], dtype=tf.float32, sparse=True)
        node_in = Input(shape=features.shape[1:], batch_size=self.adj_size[0], dtype=tf.float32)

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
            x = Dropout(0.5)(x)

        # second sage layer
        if conv_last:
            x_node = Dense(n_labels, use_bias=False)(x)
            x_neigh = Dense(n_labels, use_bias=False)(x)
            x = LAF(4)([adj_in, x_neigh])
            x = BiasLayer()(x)
            x = Add()([x_node, x])
        else:
            x = Dense(n_labels, use_bias=True)(x)

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
