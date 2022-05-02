import tensorflow as tf

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Lambda, LeakyReLU
from tensorflow.keras.layers import Dropout, Layer, Add, Concatenate
import numpy as np
from graphmb.layers import BiasLayer, LAF, GraphAttention



class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, latentdim, activation):
        super(Encoder, self).__init__()
        self.model =tf.keras.Sequential(
                [
                    Dense(intermediate_dim, activation=activation, name="encoder1"),
                    #LeakyReLU(1e-2),
                    #Dropout(0.2),
                    BatchNormalization(),
                    Dense(intermediate_dim, activation=activation, name="encoder2"),
                    #LeakyReLU(1e-2),
                    #Dropout(0.2),
                    BatchNormalization(),
                    #Dense(latentdim, activation=activation, name="encoder3"),
                ]
            )
        
    def call(self, input_features):
       return self.model(input_features)


class Decoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim, original_dim, activation):
    super(Decoder, self).__init__()
    self.model = tf.keras.Sequential(
                [
                    Dense(intermediate_dim, activation=activation,  name="decoder1"),
                    #LeakyReLU(1e-2),
                    #Dropout(0.2),
                    BatchNormalization(),
                    Dense(intermediate_dim, activation=activation,  name="decoder2"),
                    #LeakyReLU(1e-2),
                    #Dropout(0.2),
                    BatchNormalization(),
                    Dense(original_dim, activation=None, name="output_layer"),
                ]
            )
  
  def call(self, code):
    return self.model(code)


class Autoencoder(tf.keras.Model):
  def __init__(self, hidden_dim, latent_dim, features_dim, activation):
    super(Autoencoder, self).__init__()
    #breakpoint()
    self.encoder = Encoder(intermediate_dim=hidden_dim, latentdim=latent_dim, activation=activation)
    self.decoder = Decoder(intermediate_dim=hidden_dim, original_dim=features_dim, activation=activation)
  
  def call(self, input_features):
    code = self.encoder(input_features)
    reconstructed = self.decoder(code)
    return code, reconstructed

class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, hidden_dim, latent_dim, features_dim, activation):
        super(VariationalAutoencoder, self).__init__()
        #breakpoint()
        self.encoder = Encoder(intermediate_dim=hidden_dim, latentdim=hidden_dim, activation=activation)
        self.mu_layer = Dense(latent_dim, activation=None, name="mu_layer")
        self.logsigma_layer = Dense(latent_dim, activation=None, name="logsigma_layer")
        self.decoder = Decoder(intermediate_dim=hidden_dim, original_dim=features_dim, activation=activation)


    def call(self, input_features):
        code = self.encoder(input_features)
        mu = self.mu_layer(code)
        logsigma = tf.math.softplus(self.logsigma_layer(code))
        resampled = self.reparameterize(mu, logsigma)
        reconstructed = self.decoder(code)
        return mu, logsigma, reconstructed

    def reparameterize(self, mu, logsigma):
        epsilon = tf.random.normal(mu.shape)
        latent = mu + epsilon * tf.math.exp(logsigma/2)
        return latent

class TH:
    def __init__(
        self,
        input_features,
        gnn_model,
        lr=0.01,
        all_different_idx=None,
        all_same_idx=None,
        ae_model=None,
        latentdim=32,
        gnn_weight=1.0,
        ae_weight=1.0,
        scg_weight=100.0,
        kmer_dim=103,
        kmer_alpha=0.5,
        num_negatives=50,
        decoder_input="gnn",
        #no_gnn=False
    ):
        self.opt = Adam(learning_rate=lr, epsilon=1e-8)
        # self.opt = SGD(learning_rate=lr)
        self.gnn_model = gnn_model
        self.features = input_features
        # self.dense_adj = tf.sparse.to_dense(self.model.adj)
        if gnn_model is not None:
            self.adj_shape = self.gnn_model.adj.dense_shape
            S = tf.cast(tf.reduce_sum(self.gnn_model.adj.values), tf.float32)
            s0 = tf.cast(self.adj_shape[0], tf.float32)
            self.pos_weight = (s0 * s0 - S) / S
            self.norm = s0 * s0 / ((s0 * s0 - S) * 2.0)
        self.kmer_dim = kmer_dim
        self.ab_dim = input_features.shape[1] - kmer_dim
        self.kmer_alpha = kmer_alpha
        self.num_negatives = num_negatives
        self.nlatent = latentdim
        self.decoder_input = decoder_input
        self.all_different_idx = all_different_idx
        self.all_same_idx = all_same_idx
        
        self.autoencoder = ae_model
        self.use_ae = ae_model is not None
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.gnn_weight = gnn_weight
        self.ae_weight = ae_weight
        self.scg_weight = scg_weight
        self.no_gnn = gnn_model is None


    def ae_loss(self, new_features, original_features):
        #breakpoint()
        #reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(new_features, original_features)))
        kmer_recon = new_features[:, self.ab_dim:]
        abund_recon = new_features[:, :self.ab_dim]
        abund_loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.squared_difference(abund_recon, original_features[:, :self.ab_dim]), axis=1))
        kmer_loss = self.mse_loss(original_features[:, self.ab_dim:], kmer_recon)
        abund_dim = new_features.shape[1]-float(self.kmer_dim)
        abund_w = (1-self.kmer_alpha)
        if abund_dim > 1:
            abund_w /= tf.math.log(abund_dim)
        kmer_w = self.kmer_alpha/float(self.kmer_dim)
        reconstruction_error = abund_loss*abund_w + kmer_loss*kmer_w
        #print(abund_loss, abund_w, kmer_loss, kmer_w)
        return reconstruction_error

    def vae_loss(self, new_features, mu, logsigma, original_features,  kld_beta=200):
        #reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(new_features, original_features)))
        #breakpoint()
        reconstruction_error = self.ae_loss(new_features, original_features)
        #kld_weight = 1 / (self.nlatent * kld_beta)
        #kld = -0.5 * tf.math.reduce_mean(tf.math.reduce_sum(1 + logsigma - tf.math.pow(mu, 2) - tf.math.exp(logsigma), axis=1))
        #reconstruction_error += kld*kld_weight
        #print(kld, kld_weight, kld_beta, self.nlatent)
        return reconstruction_error

    @tf.function
    def train_autoencoder(self):
        with tf.GradientTape() as tape:
            embs, recons = self.autoencoder(self.features)
            recon_loss = self.ae_loss(recons, self.features)
            gradients = tape.gradient(recon_loss, self.autoencoder.trainable_variables)
            gradient_variables = zip(gradients, self.autoencoder.trainable_variables)
            self.opt.apply_gradients(gradient_variables)
        return 0, recon_loss, 0

    @tf.function
    def train_vae(self):
        #breakpoint()
        with tf.GradientTape() as tape:
            mu, logsigma, recons = self.autoencoder(self.features)
            recon_loss = self.vae_loss(recons, mu, logsigma, self.features)
            gradients = tape.gradient(recon_loss, self.autoencoder.trainable_variables)
            gradient_variables = zip(gradients, self.autoencoder.trainable_variables)
            self.opt.apply_gradients(gradient_variables)
        return 0, recon_loss, 0


    @tf.function
    def train_unsupervised(self, idx):
        with tf.GradientTape() as tape:
            #breakpoint()
            # run encoder first
            if self.use_ae:
                ae_embs = self.autoencoder.encoder.model(self.features)
            else:
                ae_embs = self.features

            # run gnn model
            node_hat = self.gnn_model(ae_embs, idx)

            # run decoder and compute AE loss
            recon_loss = tf.constant(0, dtype=tf.float32)
            if self.use_ae:
                if self.decoder_input == "gnn":
                    recon_features = self.autoencoder.decoder.model(node_hat)
                elif self.decoder_input == "ae":
                    recon_features = self.autoencoder.decoder.model(ae_embs)
                #node_hat = features
                # assert recon_features.shape == self.features.shape
                #breakpoint()
                #kmer_recon = recon_features[:, : self.kmer_dim]
                #abund_recon = recon_features[:, self.kmer_dim:]
                #kmer_loss = self.mse_loss(self.features[:, :self.kmer_dim], kmer_recon)
                #kmer_loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.squared_difference(kmer_recon, self.features[:, :self.kmer_dim]), axis=1))
                #abund_loss = self.mse_loss(self.features[:, self.kmer_dim:], abund_recon)
                #abund_loss = tf.math.reduce_mean(np.sum(np.square(abund_recon - self.features[:, self.kmer_dim:]), axis=1))
                #abund_loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.squared_difference(abund_recon, self.features[:, self.kmer_dim:]), axis=1))
                #recon_loss = (self.kmer_alpha / float(self.kmer_dim)) * kmer_loss + (1.0 - self.kmer_alpha) * abund_loss
                #recon_loss = self.mse_loss(self.features, recon_features)
                #recon_loss *= self.ae_weight
                #recon_loss = kmer_loss
                #recon_loss = self.ae_loss(recon_features, self.features)
            loss = recon_loss
            gnn_loss = tf.constant(0, dtype=tf.float32)
            if not self.no_gnn:
                # create random negatives for gnn_loss
                row_embs = tf.gather(indices=self.gnn_model.adj.indices[:, 0], params=node_hat)
                col_embs = tf.gather(indices=self.gnn_model.adj.indices[:, 1], params=node_hat)
                positive_pairwise = tf.reduce_sum(tf.math.multiply(row_embs, col_embs), axis=1)

                neg_idx = tf.random.uniform(
                    shape=(self.num_negatives * len(self.gnn_model.adj.indices),),
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
                gnn_loss = 0.5 * (pos_loss + neg_loss) * self.gnn_weight
                #loss += gnn_loss
            
            # SCG loss
            scg_loss = tf.constant(0, dtype=tf.float32)
            if self.all_different_idx is not None and self.scg_weight > 0:
                ns1 = tf.gather(node_hat, self.all_different_idx[:, 0])
                ns2 = tf.gather(node_hat, self.all_different_idx[:, 1])
                all_diff_pairs = tf.math.exp(-0.5 * tf.reduce_sum((ns1 - ns2) ** 2, axis=-1))
                scg_loss = tf.reduce_mean(all_diff_pairs) * self.scg_weight
                loss += scg_loss

        """if self.no_gnn or self.use_ae:
            tw_encoder = self.encoder.trainable_weights
            tw_decoder = self.decoder.trainable_weights
            if self.no_gnn:
                tw = tw_encoder + tw_decoder
        if not self.no_gnn:
            tw = self.model.trainable_weights
            if self.use_ae:
                tw += tw_encoder + tw_decoder"""
        tw = self.gnn_model.trainable_weights
        grads = tape.gradient(loss, tw)
        self.opt.apply_gradients(zip(grads, tw))
        return gnn_loss, recon_loss, scg_loss
    
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
