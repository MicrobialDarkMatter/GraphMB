import tensorflow as tf

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import BatchNormalization, Lambda
from tensorflow.keras.layers import Dropout, Layer, Add, Concatenate

from graphmb.layers import BiasLayer, LAF, GraphAttention


class TH:
    def __init__(
        self,
        model,
        lr=0.01,
        lambda_vae=0.1,
        all_different_idx=None,
        all_same_idx=None,
    ):
        self.opt = Adam(learning_rate=lr, epsilon=1e-8)
        # self.opt = SGD(learning_rate=lr)
        self.lambda_vae = lambda_vae
        self.model = model
        self.dense_adj = tf.sparse.to_dense(self.model.adj)
        self.adj_shape = self.model.adj.dense_shape
        S = tf.cast(tf.reduce_sum(self.model.adj.values), tf.float32)
        s0 = tf.cast(self.adj_shape[0], tf.float32)

        self.all_different_idx = all_different_idx
        self.all_same_idx = all_same_idx
        self.pos_weight = (s0 * s0 - S) / S
        self.norm = s0 * s0 / ((s0 * s0 - S) * 2.0)

        self.decoder = Dense(self.model.features.shape[1], activation="relu")

    @tf.function
    def train(self, idx):
        with tf.GradientTape() as tape:
            node_features = self.model(idx)
            # Not ideal to this
            node_pairwise = tf.matmul(node_features, node_features, transpose_b=True)

            # Gather only the non zero indices
            non_zero_node_pairwise = tf.gather_nd(node_pairwise, self.model.adj.indices)

            logits = tf.gather(self.model.labels, idx)

            labels = tf.where(self.model.adj.indices > 0.5, tf.ones_like(logits), tf.zeros_like(logits))

            loss = tf.keras.losses.binary_crossentropy(labels, logits, from_logits=True)
            # loss = tf.nn.weighted_cross_entropy_with_logits(tf.ones_like(logits), logits, self.model.adj.values)
            loss = tf.reduce_mean(loss)
        tw = self.model.trainable_weights
        grads = tape.gradient(loss, tw)
        self.opt.apply_gradients(zip(grads, tw))
        return loss

    @tf.function
    def train_unsupervised(self, idx):
        with tf.GradientTape() as tape:
            # breakpoint()
            node_hat = self.model(idx)
            recon_features = self.decoder(node_hat)
            assert recon_features.shape == self.model.features.shape
            recon_loss = tf.keras.losses.MeanSquaredError()(self.model.features, recon_features)

            pairwise_similarity = tf.matmul(node_hat, node_hat, transpose_b=True)
            # y = tf.reshape(self.dense_adj, [-1])
            # y_hat = tf.reshape(pairwise_similarity, [-1])
            ##loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_hat, labels=y, pos_weight=self.pos_weight)
            # loss = self.norm * tf.reduce_mean(loss)

            diff_loss = 0
            same_loss = 0

            positive_pairwise = tf.gather_nd(pairwise_similarity, self.model.adj.indices)

            neg_idx = tf.random.uniform(
                shape=(50 * len(self.model.adj.indices),),
                minval=0,
                maxval=self.adj_shape[0] * self.adj_shape[1] - 1,
                dtype=tf.int64,
            )
            neg_idx_row = tf.cast(neg_idx, tf.float32) / tf.cast(self.adj_shape[1], tf.float32)
            neg_idx_row = tf.cast(neg_idx_row, tf.int64)[:, None]
            neg_idx_col = tf.cast((neg_idx % self.adj_shape[1]), tf.int64)[:, None]
            neg_idx = tf.concat((neg_idx_row, neg_idx_col), axis=-1)

            negative_pairs = tf.gather_nd(pairwise_similarity, neg_idx)

            pos_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.ones_like(positive_pairwise), positive_pairwise, from_logits=True
                )
            )
            neg_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.zeros_like(negative_pairs), negative_pairs, from_logits=True)
            )
            loss = 0.5 * (pos_loss + neg_loss)
            loss += recon_loss

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
        return loss, recon_loss, diff_loss

    @tf.function
    def train_unsupervised_vae(self, idx):
        with tf.GradientTape() as tape:
            node_hat = self.model(idx)
            n_nodes = tf.cast(tf.shape(idx)[0], tf.float32)

            mu, log_std = tf.split(node_hat, 2, axis=1)
            std = tf.math.exp(log_std)
            eps = tf.random.normal(tf.shape(std))
            z = eps * std + mu
            kl = 0.5 / n_nodes * tf.reduce_mean(tf.reduce_sum(1.0 + 2.0 * log_std - mu ** 2 - std ** 2, axis=1))

            pairwise_similarity = tf.matmul(z, z, transpose_b=True)
            y = tf.reshape(self.dense_adj, [-1])
            y_hat = tf.reshape(pairwise_similarity, [-1])

            loss = self.norm * tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(logits=y_hat, labels=y, pos_weight=self.pos_weight)
            )

            loss = loss - self.lambda_vae * kl
        tw = self.model.trainable_weights
        grads = tape.gradient(loss, tw)
        self.opt.apply_gradients(zip(grads, tw))
        return loss

    @tf.function
    def train_unsupervised_v2(self, idx):
        with tf.GradientTape() as tape:
            node_hat = self.model(idx)
            pairwise_similarity = tf.matmul(node_hat, node_hat, transpose_b=True)
            positive_pairwise = tf.gather_nd(pairwise_similarity, self.model.adj.indices)
            pos_weights = self.model.adj.values
            pos_logits = tf.reshape(positive_pairwise, [-1])
            pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(pos_logits), pos_logits)
            pos_loss = tf.reduce_sum(pos_weights * pos_loss) / tf.reduce_sum(pos_weights)
            # pos_loss = tf.reduce_mean(pos_loss)

            diag = tf.reduce_sum(node_hat * node_hat, axis=-1)
            diag_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(diag), diag)
            diag_loss = tf.reduce_mean(diag_loss)

            neg_idx = self.all_different_idx
            negative_pairwise = tf.gather_nd(pairwise_similarity, neg_idx)
            neg_logits = tf.reshape(negative_pairwise, [-1])
            neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(neg_logits), neg_logits)
            neg_loss = tf.reduce_mean(neg_loss)

            # neg_idx = tf.random.uniform(shape=(50*len(self.model.adj.indices),), minval=0, \
            #                            maxval=self.adj_shape[0]*self.adj_shape[1]-1, dtype=tf.int64)
            # neg_idx_row = tf.cast(neg_idx, tf.float32) / tf.cast(self.adj_shape[1], tf.float32)
            # neg_idx_row = tf.cast(neg_idx_row, tf.int64)[:, None]
            # neg_idx_col = tf.cast((neg_idx % self.adj_shape[1]), tf.int64)[:,None]
            # neg_idx = tf.concat((neg_idx_row, neg_idx_col), axis=-1)
            # negative_pairwise = tf.gather_nd(pairwise_similarity, neg_idx)
            # neg_logits = tf.reshape(negative_pairwise, [-1])
            # neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(neg_logits), neg_logits)
            # neg_loss = tf.reduce_mean(neg_loss)

            # neg_idx = self.all_different_idx
            # n1 = tf.gather(node_hat, neg_idx[:,0])
            # n2 = tf.gather(node_hat, neg_idx[:,1])
            # neg_dists = tf.math.exp(-0.5*tf.reduce_sum((n1-n2)**2, axis=-1))
            # neg_dists = tf.reduce_mean(neg_dists)

            # loss = pos_loss - diag_loss + neg_loss
            loss = pos_loss + diag_loss + neg_loss
        tw = self.model.trainable_weights
        grads = tape.gradient(loss, tw)
        self.opt.apply_gradients(zip(grads, tw))
        return loss

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
        features,
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
        self.model.build([tuple(self.features.shape), tuple(self.adj_size)])

    def call(self, idx, training=True):
        output = self.model((self.features, self.adj), training=training)
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
    def __init__(self, features, labels, adj, n_labels=None, hidden_units=None, layers=None, conv_last=None):
        super(GAT, self).__init__()
        self.features = features
        self.labels = labels
        self.adj = adj
        self.adj_size = adj.dense_shape.numpy()

        adj_in = Input(shape=self.adj_size[1:], batch_size=self.adj_size[0], dtype=tf.float32, sparse=True)
        node_in = Input(shape=features.shape[1:], batch_size=self.adj_size[0], dtype=tf.float32)

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
        self.model.build([tuple(self.features.shape), tuple(self.adj_size)])

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
    def __init__(self, features, labels, adj, n_labels=None, hidden_units=None, layers=None, conv_last=None):
        super(SAGE, self).__init__()
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
        self.model.build([tuple(self.features.shape), tuple(self.adj_size)])

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
