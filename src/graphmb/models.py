import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Softmax, Embedding, LayerNormalization
#from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Lambda, LeakyReLU, Flatten
from tensorflow.keras.layers import Dropout, Add, Concatenate
import numpy as np
from tensorflow.keras.regularizers import l2

from spektral.layers import GCNConv

from graphmb.layers import BiasLayer, LAF, GraphAttention


class VAEEncoder(Model):
    def __init__(self, abundance_dim, kmers_dim, hiddendim, zdim=64, dropout=0, layers=2):
        super(VAEEncoder, self).__init__()
        self.abundance_dim = abundance_dim
        self.kmers_dim = kmers_dim
        in_ = Input(shape=(abundance_dim+kmers_dim,))
        x = in_
        for _ in range(layers):
            x = Dense(hiddendim, activation='linear')(x)
            x = LeakyReLU(0.01)(x)
            if dropout > 0:
                x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
        mu = Dense(zdim)(x)
        logvar = Dense(zdim, kernel_initializer='zeros')(x)
        self.model = Model(in_, [mu, logvar])
 
    def call(self, x, training=False):
        mu, sigma = self.model(x, training=training)
        return mu, sigma
    
class VAEDecoder(Model):
    def __init__(self, abundance_dim, kmers_dim, hiddendim, zdim=64, dropout=0, layers=2):
        super(VAEDecoder, self).__init__()
        self.abundance_dim = abundance_dim
        self.kmers_dim = kmers_dim
        in_ = Input(shape=(zdim,))
        x = in_
        for _ in range(layers):
            x = Dense(hiddendim, activation='linear')(x)
            x = LeakyReLU(0.01)(x)
            if dropout > 0:
                x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
            
        x1 = Dense(abundance_dim)(x)
        x2 = Dense(kmers_dim)(x)
        if self.abundance_dim > 1:
            x1 = Softmax()(x1)
        x = Concatenate()((x1,x2))
        
        self.model = Model(in_, x)
 
    def call(self, z, training=False):
        x_hat = self.model(z, training=training)
        return x_hat


class TrainHelperVAE:
    def __init__(self, encoder, decoder, learning_rate=1e-3,  kld_weight=1/200.):
        self.encoder = encoder
        self.decoder = decoder
        self.kld_weight = kld_weight
        self.abundance_dim = self.encoder.abundance_dim
        self.kmers_dim = self.encoder.kmers_dim
        if self.abundance_dim > 1:
            self.abundance_weight = 0.85 / tf.math.log(float(self.abundance_dim))
            self.kmer_weight = 0.15
        else:
            self.abundance_weight = 0.5
            self.kmer_weight = 0.5
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.logvar = None
        self.mu = None
        self.z = None
        
    def train_step(self, x, writer=None, epoch=0, vae=True):
        #breakpoint()
        losses = self._train_step(x, vae=vae, writer=writer, epoch=epoch)
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('mean logvar', self.logvar, step=epoch)
                tf.summary.scalar('mean mu', self.mu, step=epoch)
        losses = [loss.numpy() for loss in losses]
        return losses
    
    @tf.function
    def loss(self, x, mu, logvar, vae, training=True, writer=None, epoch=0):
        if vae:
            epsilon = tf.random.normal(tf.shape(mu))
            z = mu + epsilon * tf.math.exp(0.5 * logvar)
            kld  = 0.5*tf.math.reduce_mean(tf.math.reduce_mean(1.0 + logvar - tf.math.pow(mu, 2) - tf.math.exp(logvar), axis=1))
            kld  = kld * self.kld_weight
        else:
            z = mu
            kld = 0
        x_hat = self.decoder(z, training=training)
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('min ab', tf.reduce_min(x_hat[:, :self.abundance_dim]), step=epoch)
        if self.abundance_dim > 1:
            mse1 = - tf.reduce_mean(tf.reduce_sum((tf.math.log(x_hat[:, :self.abundance_dim] + 1e-9) * x[:, :self.abundance_dim]), axis=1))
        else:
            mse1 = self.abundance_weight*tf.reduce_mean( (x[:, :self.abundance_dim] - x_hat[:, :self.abundance_dim])**2)
        mse2 = self.kmer_weight*tf.reduce_mean( tf.reduce_mean((x[:, self.abundance_dim:] - x_hat[:, self.abundance_dim:])**2, axis=1))

        return mse1, mse2, kld

    @tf.function
    def _train_step(self, x, vae=True, writer=None, epoch=0):
        with tf.GradientTape() as tape:
            mu, logvar = self.encoder(x, training=True)
            logvar = tf.clip_by_value(logvar, -2, 2)
            self.logvar = tf.cast(tf.math.reduce_mean(logvar), float)
            self.mu = tf.cast(tf.math.reduce_mean(mu), float)
            mse1, mse2, kld = self.loss(x, mu, logvar, vae, training=True, writer=writer, epoch=epoch)
            loss = mse1 + mse2 - kld

        tw = self.encoder.trainable_weights + self.decoder.trainable_weights   
        grads = tape.gradient(loss, tw)
        grad_norm = tf.linalg.global_norm(grads)
        clip_grads, _ = tf.clip_by_global_norm(grads, 5,  use_norm=grad_norm)
        new_grad_norm = tf.linalg.global_norm(clip_grads)
        self.opt.apply_gradients(zip(clip_grads, tw))

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('grad norm', grad_norm, step=epoch)
                tf.summary.scalar('clipped grad norm', new_grad_norm, step=epoch)

        return loss, mse2, mse1, kld



class TH:
    def __init__(
        self,
        input_features,
        gnn_model,
        lr=0.01,
        all_different_idx=None,
        all_same_idx=None,
        ae_encoder=None,
        ae_decoder=None,
        latentdim=32,
        gnn_weight=1.0,
        ae_weight=1.0,
        kld_weight = 1/200,
        scg_weight=100.0,
        kmer_weight=0.15,
        abundance_weight=0.85,
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
        
        self.encoder = ae_encoder
        self.decoder = ae_decoder
        self.use_ae = ae_encoder is not None
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.gnn_weight = gnn_weight
        self.ae_weight = ae_weight
        self.kld_weight = kld_weight
        self.scg_weight = scg_weight
        self.kmer_weight = kmer_weight
        self.abundance_weight = abundance_weight
        self.no_gnn = gnn_model is None
        self.train_ae = False
        self.abundance_dim = self.decoder.abundance_dim
        self.kmers_dim = self.decoder.kmers_dim

    @tf.function
    def train_unsupervised(self, idx):
        with tf.GradientTape() as tape:
            #breakpoint()
            # run encoder first
            if self.use_ae:
                ae_embs = self.encoder(self.features)[0]

            else:
                ae_embs = self.features

            # run gnn model
            node_hat = self.gnn_model(ae_embs, idx)
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
                loss = gnn_loss
            
            # SCG loss
            scg_loss = tf.constant(0, dtype=tf.float32)
            if self.all_different_idx is not None and self.scg_weight > 0:
                ns1 = tf.gather(node_hat, self.all_different_idx[:, 0])
                ns2 = tf.gather(node_hat, self.all_different_idx[:, 1])
                all_diff_pairs = tf.math.exp(-0.5 * tf.reduce_sum((ns1 - ns2) ** 2, axis=-1))
                scg_loss = tf.reduce_mean(all_diff_pairs) * self.scg_weight
                loss += scg_loss

        tw = self.gnn_model.trainable_weights
        grads = tape.gradient(loss, tw)
        self.opt.apply_gradients(zip(grads, tw))
        return loss, gnn_loss, scg_loss
    
    @tf.function
    def ae_loss(self, x, mu, logvar, vae, training=True, writer=None, epoch=0):
        if vae:
            epsilon = tf.random.normal(tf.shape(mu))
            z = mu + epsilon * tf.math.exp(0.5 * logvar)
            kld  = 0.5*tf.math.reduce_mean(tf.math.reduce_mean(1.0 + logvar - tf.math.pow(mu, 2) - tf.math.exp(logvar), axis=1))
            kld  = kld * self.kld_weight
        else:
            z = mu
            kld = 0
        x_hat = self.decoder(z, training=training)
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('min ab', tf.reduce_min(x_hat[:, :self.abundance_dim]), step=epoch)
        if self.abundance_dim > 1:
            mse1 = - tf.reduce_mean(tf.reduce_sum((tf.math.log(x_hat[:, :self.abundance_dim] + 1e-9) * x[:, :self.abundance_dim]), axis=1))
        else:
            mse1 = self.abundance_weight*tf.reduce_mean( (x[:, :self.abundance_dim] - x_hat[:, :self.abundance_dim])**2)
        mse2 = self.kmer_weight*tf.reduce_mean( tf.reduce_mean((x[:, self.abundance_dim:] - x_hat[:, self.abundance_dim:])**2, axis=1))

        return mse1, mse2, kld

    @tf.function
    def train_unsupervised_decode(self, idx):
        with tf.GradientTape() as tape:
            #breakpoint()
            # run gnn model
            # TODO GNN model that encodes mu and logvar
            gnn_embs = self.gnn_model(self.features, idx)
            mu, logvar = self.encoder(gnn_embs, training=True)
            z_sample = tf.random.normal(tf.shape(mu)) * tf.exp(logvar)

            gnn_loss = tf.constant(0, dtype=tf.float32)
            if not self.no_gnn:
                # create random negatives for gnn_loss
                row_embs = tf.gather(indices=self.gnn_model.adj.indices[:, 0], params=z_sample)
                col_embs = tf.gather(indices=self.gnn_model.adj.indices[:, 1], params=z_sample)
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
                    neg_row_embs = tf.gather(indices=neg_idx_row, params=z_sample)
                    neg_col_embs = tf.gather(indices=neg_idx_col, params=z_sample)
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
                loss = gnn_loss
            
            # SCG loss
            scg_loss = tf.constant(0, dtype=tf.float32)
            if self.all_different_idx is not None and self.scg_weight > 0:
                ns1 = tf.gather(z_sample, self.all_different_idx[:, 0])
                ns2 = tf.gather(z_sample, self.all_different_idx[:, 1])
                all_diff_pairs = tf.math.exp(-0.5 * tf.reduce_sum((ns1 - ns2) ** 2, axis=-1))
                scg_loss = tf.reduce_mean(all_diff_pairs) * self.scg_weight
                loss += scg_loss

            # decode
            kmer_loss, ab_loss, kld_loss = self.ae_loss(tf.gather(self.features, idx), mu, logvar, vae=True)
            loss += kmer_loss + ab_loss - kld_loss

        tw = self.gnn_model.trainable_weights
        tw += self.encoder.trainable_weights
        tw += self.decoder.trainable_weights
        grads = tape.gradient(loss, tw)
        self.opt.apply_gradients(zip(grads, tw))
        return loss, gnn_loss, scg_loss, kmer_loss, ab_loss, kld_loss

    @staticmethod
    def sample_idx(idx, n):
        n = tf.cast(n, tf.int64)
        N = tf.cast(tf.shape(idx)[0], tf.int64)
        random_idx = tf.random.uniform(shape=(n,), minval=0, maxval=N, dtype=tf.int64)
        s_idx = tf.gather_nd(idx, random_idx)
        return s_idx


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
