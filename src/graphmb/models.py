from os import uname
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

from spektral.layers import GCNConv
from tqdm import tqdm

#import tensorflow_probability as tfp

class VAEEncoder(Model):
    def __init__(self, abundance_dim, kmers_dim, hiddendim, zdim=64, dropout=0, layers=2):
        super(VAEEncoder, self).__init__()
        self.abundance_dim = abundance_dim
        self.kmers_dim = kmers_dim
        self.zdim = zdim
        in_ = Input(shape=(abundance_dim+kmers_dim,))
        x = in_
        for i in range(layers):
            x = Dense(hiddendim, activation='linear', name=f"encoder_{i}")(x)
            x = LeakyReLU(0.01)(x)
            if dropout > 0:
                x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
        mu = Dense(zdim, name="mu")(x)
        logvar = Dense(zdim, kernel_initializer='zeros', name="logvar")(x)
        self.model = Model(in_, [mu, logvar])
 
    def call(self, x, training=False):
        mu, sigma = self.model(x, training=training)
        return mu, sigma

    
class VAEDecoder(Model):
    def __init__(self, abundance_dim, kmers_dim, hiddendim, zdim=64, dropout=0, layers=2):
        super(VAEDecoder, self).__init__()
        self.abundance_dim = abundance_dim
        self.kmers_dim = kmers_dim
        self.zdim = zdim
        in_ = Input(shape=(zdim,))
        x = in_
        for i in range(layers):
            x = Dense(hiddendim, activation='linear', name=f"decoder_{i}")(x)
            x = LeakyReLU(0.01)(x)
            if dropout > 0:
                x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
            
        x1 = Dense(abundance_dim, name="decoder_ab")(x)
        x2 = Dense(kmers_dim, name="decoder_kmer")(x)
        if self.abundance_dim > 1:
            x1 = Softmax()(x1)
        x = Concatenate()((x1,x2))
        
        self.model = Model(in_, x)
 
    def call(self, z, training=False):
        x_hat = self.model(z, training=training)
        return x_hat

class Encoder(Model):
    def __init__(self, features_dim, hiddendim, zdim=64, dropout=0, layers=2):
        super(VAEEncoder, self).__init__()
        self.zdim = zdim
        in_ = Input(shape=(features_dim,))
        x = in_
        for i in range(layers):
            x = Dense(hiddendim, activation='linear', name=f"encoder_{i}")(x)
            x = LeakyReLU(0.01)(x)
            if dropout > 0:
                x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
        mu = Dense(zdim, name="mu")(x)
        logvar = Dense(zdim, kernel_initializer='zeros', name="logvar")(x)
        self.model = Model(in_, [mu, logvar])
 
    def call(self, x, training=False):
        mu, sigma = self.model(x, training=training)
        return mu, sigma

    
class Decoder(Model):
    def __init__(self, features_dim, hiddendim, zdim=64, dropout=0, layers=2):
        super(VAEDecoder, self).__init__()
        self.features_dim = features_dim
        self.zdim = zdim
        in_ = Input(shape=(zdim,))
        x = in_
        for i in range(layers):
            x = Dense(hiddendim, activation='linear', name=f"decoder_{i}")(x)
            x = LeakyReLU(0.01)(x)
            if dropout > 0:
                x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
            
        x = Dense(features_dim, name="decoder_ab")(x)
        #x2 = Dense(kmers_dim, name="decoder_kmer")(x)
        #if self.abundance_dim > 1:
        #    x1 = Softmax()(x1)
        #x = Concatenate()((x1,x2))
        
        self.model = Model(in_, x)
 
    def call(self, z, training=False):
        x_hat = self.model(z, training=training)
        return x_hat


class LabelClassifier(Model):
    def __init__(self, n_classes, hiddendim=128, zdim=64, dropout=0, layers=1):
        super(LabelClassifier, self).__init__()
        in_ = Input(shape=(zdim,))
        x = in_
        for nl in range(layers):
            if nl == layers-1:
                x = Dense(n_classes, activation='softmax')(x)
            else:
                x = Dense(hiddendim, activation='linear')(x)
                #x = LeakyReLU(0.01)(x)
                if dropout > 0:
                    x = Dropout(dropout)(x)

        self.model = Model(in_, x)
        self.loss_fn = SparseCategoricalCrossentropy()
 
    def call(self, z, mask, training=False):
        predictions = self.model(tf.gather(params=z, indices=mask), training=training)
        return predictions

    def loss(self, gold_labels, predicted_labels):
        return self.loss_fn(y_true=gold_labels,y_pred=predicted_labels)

class TrainHelperVAE:
    def __init__(self, encoder, decoder, learning_rate=1e-3,  kld_weight=1/200.,
                train_weights=False, classification=False, n_classes=0,
                gold_labels=None, mask_labels=0.0, ae_alpha=1):
        self.encoder = encoder
        self.decoder = decoder
        self.train_weights = train_weights
        self.abundance_dim = self.encoder.abundance_dim
        self.kmers_dim = self.encoder.kmers_dim
        self.ae_alpha = ae_alpha
        if train_weights:
            #self.kld_weight = tf.Variable(kld_weight)
            self.kld_weight = kld_weight
            #self.abundance_weight = tf.Variable(0.5)
            self.kmer_weight = tf.Variable(0.1)
        else:
            self.kld_weight = kld_weight
            if self.abundance_dim > 1:
                #self.abundance_weight = 0.85 / tf.math.log(float(self.abundance_dim))
                self.kmer_weight = 0.15
            else:
                #self.abundance_weight = 0.5
                self.kmer_weight = 0.5
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.logvar = None
        self.mu = None
        self.z = None
        self.classify = classification
        if self.classify:
            self.classifier = LabelClassifier(n_classes, zdim=encoder.zdim)
            self.gold_labels = gold_labels
            self.mask_labels = mask_labels
            
    def train_step(self, x, writer=None, epoch=0, vae=True, gold_labels=None):
        losses = self._train_step(x, vae=vae, writer=writer, epoch=epoch, gold_labels=gold_labels)
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('mean logvar', self.logvar, step=epoch)
                tf.summary.scalar('mean mu', self.mu, step=epoch)
        losses = [loss.numpy() for loss in losses]
        return losses
    
    @tf.function
    def loss(self, x, mu, logvar, vae, training=True, writer=None, epoch=0, gold_labels=None):
        if vae:
            epsilon = tf.random.normal(tf.shape(mu))
            z = mu + epsilon * tf.math.exp(0.5 * logvar)
            kld  = 0.5*tf.math.reduce_mean(tf.math.reduce_mean(1.0 + logvar - tf.math.pow(mu, 2) - tf.math.exp(logvar), axis=1))
            kld  = kld * self.kld_weight
        else:
            z = mu
            kld = tf.convert_to_tensor(0.0)
        x_hat = self.decoder(z, training=training)
        if self.classify:
            if self.mask_labels > 0:
                use_labels = np.random.choice(z.shape[0], int(z.shape[0]*(1-self.mask_labels)))
            else:
                use_labels = np.arange(z.shape[0])
            predictions = self.classifier(z, mask=use_labels)
            prediction_loss = self.classifier.loss(gold_labels[use_labels], predictions)
        else:
            prediction_loss = tf.convert_to_tensor(0.0)
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('min ab', tf.reduce_min(x_hat[:, :self.abundance_dim]), step=epoch)
        if self.abundance_dim > 1:
            mse1 = -tf.reduce_mean(tf.reduce_sum((tf.math.log(x_hat[:, :self.abundance_dim] + 1e-9) * x[:, :self.abundance_dim]), axis=1))
        else:
            mse1 = tf.reduce_mean( (x[:, :self.abundance_dim] - x_hat[:, :self.abundance_dim])**2)
        if self.train_weights:
            kmer_weight = tf.math.sigmoid(self.kmer_weight)
        else:
            kmer_weight = self.kmer_weight
        mse1 *= (1-kmer_weight)
        mse2 = kmer_weight*tf.reduce_mean( tf.reduce_mean((x[:, self.abundance_dim:] - x_hat[:, self.abundance_dim:])**2, axis=1))

        return mse1, mse2, kld, prediction_loss

    @tf.function
    def _train_step(self, x, vae=True, writer=None, epoch=0, gold_labels=None):
        with tf.GradientTape() as tape:
            mu, logvar = self.encoder(x, training=True)
            logvar = tf.clip_by_value(logvar, -2, 2)
            self.logvar = tf.cast(tf.math.reduce_mean(logvar), float)
            self.mu = tf.cast(tf.math.reduce_mean(mu), float)
            mse1, mse2, kld, predl = self.loss(x, mu, logvar, vae, training=True,
                                               writer=writer, epoch=epoch,
                                               gold_labels=gold_labels)
            loss = mse1 + mse2 - kld + predl
            loss *= self.ae_alpha
        tw = self.encoder.trainable_weights + self.decoder.trainable_weights 
        if self.train_weights:
            tw += [self.kmer_weight] # self.kld_weight
        if self.classify:
            tw += self.classifier.trainable_weights
        grads = tape.gradient(loss, tw)
        grad_norm = tf.linalg.global_norm(grads)
        clip_grads, _ = tf.clip_by_global_norm(grads, 5,  use_norm=grad_norm)
        new_grad_norm = tf.linalg.global_norm(clip_grads)
        self.opt.apply_gradients(zip(clip_grads, tw))

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('grad norm', grad_norm, step=epoch)
                tf.summary.scalar('clipped grad norm', new_grad_norm, step=epoch)

        return loss, mse2, mse1, kld, predl


class NoiseModel(Model):
    def __init__(self, abundance_dim, emb_dim, hiddendim=128, dropout=0, layers=1):
        super(NoiseModel, self).__init__()
        self.abundance_dim = abundance_dim
        self.kmers_dim = emb_dim
        in_ = Input(shape=((abundance_dim+emb_dim)*2,))
        x = in_
        for i in range(layers):
            x = Dense(hiddendim, activation='linear', name=f"noise_{i}")(x)
            x = LeakyReLU(0.01)(x)
            if dropout > 0:
                x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
        noise = Dense(1, name="mu")(x)
        self.model = Model(in_, noise)
 
    def call(self, x, training=False):
        noise = self.model(x, training=training)
        return noise

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
        decoder_input="gnn",
        classifier=None,
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
        kmers_dim=103,
        abundance_dim=4,
        labels=None,
        use_gnn=True,
        use_noise=False,
        loglevel="warning",
        pretrainvae=0
    ):
        self.opt = Adam(learning_rate=lr, epsilon=1e-8)
        # self.opt = SGD(learning_rate=lr)
        self.gnn_model = gnn_model
        self.features = input_features
        # self.dense_adj = tf.sparse.to_dense(self.model.adj)
        if gnn_model is not None:
            self.adj_shape = self.gnn_model.adj.dense_shape
        self.kmer_dim = kmer_dim
        self.ab_dim = input_features.shape[1] - kmer_dim
        self.kmer_alpha = kmer_alpha
        self.num_negatives = num_negatives
        self.nlatent = latentdim
        self.scg_pairs = all_different_idx
        self.all_same_idx = all_same_idx
        self.classifier = classifier
        self.encoder = ae_encoder
        self.decoder = ae_decoder
        self.decoder_input = decoder_input
        self.use_ae = ae_encoder is not None
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.gnn_weight = gnn_weight
        self.ae_weight = ae_weight
        self.kld_weight = kld_weight
        self.scg_weight = scg_weight
        self.kmer_weight = kmer_weight
        self.noise_reg = 1
        self.abundance_weight = abundance_weight
        self.no_gnn = gnn_model is None
        self.train_ae = False
        self.abundance_dim = abundance_dim
        self.kmers_dim = kmers_dim
        self.use_gnn = use_gnn
        self.use_noise = use_noise
        self.loglevel = loglevel
        self.pretrainvae = pretrainvae
        self.epoch = 0
        if self.use_noise:
            #noise_units = 1
            #self.positive_noises = tf.Variable(tf.random_normal_initializer()(shape=(self.gnn_model.adj.values.shape + noise_units), dtype=tf.float32), trainable=True)
            #self.scg_noises = tf.Variable(tf.random_normal_initializer()(shape=(self.scg_pairs.shape[0], noise_units), dtype=tf.float32), trainable=True)
            #self.noise_weights = tf.Variable(tf.random_normal_initializer()(shape=(noise_units,), dtype=tf.float32), trainable=True)
            # alt: Dense layer with concat of embs (or feats) as input and single value output
            self.noise_model = NoiseModel(abundance_dim=abundance_dim, emb_dim=0)
            self.noise_dict = {}

    def sample_negatives_old(self, edge_idx):
        # get numbers between 0 and n*(n-1)
        neg_idx = tf.random.uniform(
            shape=[len(edge_idx)*self.num_negatives,],
            minval=0,
            maxval=self.adj_shape[0] * self.adj_shape[1] - 1,
            dtype=tf.int64,
        )
        neg_idx_row = tf.math.minimum(
            tf.cast(self.adj_shape[0] - 1, tf.float32),
            tf.cast(neg_idx, tf.float32) / tf.cast(self.adj_shape[1], tf.float32),
        )
        neg_idx_row = tf.cast(neg_idx_row, tf.int64)
        neg_idx_col = tf.cast(tf.math.minimum(self.adj_shape[1] - 1,
                              (neg_idx % self.adj_shape[1])), tf.int64)
        return neg_idx_row, neg_idx_col

    @tf.function
    def sample_negatives(self, edge_idx, node_idx):
        # for 5 negatives and batchsize 256, this should give 1% or less false negatives
        neg_idx = tf.random.uniform(shape=(2, len(edge_idx)*self.num_negatives),
                                    maxval=node_idx.shape[0], dtype=tf.int32)
        # get neg idx relative to node_idx (of this batch)
        neg_idx_row = tf.gather(params=node_idx, indices=neg_idx[0,:])
        neg_idx_col = tf.gather(params=node_idx, indices=neg_idx[1,:])
        return neg_idx_row, neg_idx_col
        #return neg_idx[0,:], neg_idx[1,:]

    @tf.function
    def nodedist(self, u, v):
        #breakpoint()
        #return tf.reduce_sum((tf.expand_dims(x, 0) - tf.expand_dims(y, 1)**2), axis=-1)
        u = tf.nn.l2_normalize(u, axis=1)
        v = tf.nn.l2_normalize(v, axis=1)
        pairwise = tf.reduce_sum(tf.math.multiply(u, v), axis=1)
        #pairwise = -tf.norm(tf.math.subtract(u, v) + + 1.0e-12, ord='euclidean', axis=1,)
        return pairwise
    
    @tf.function
    def edge_loss(self, pos_dists, neg_dists):
        #breakpoint()
        y_true = tf.concat((tf.ones_like(pos_dists), tf.zeros_like(neg_dists)), axis=0)
        y_pred = tf.concat((pos_dists, neg_dists), axis=0)
        gnn_loss = tf.keras.metrics.binary_crossentropy(y_true, y_pred, from_logits=True)
        #gnn_loss = tf.keras.metrics.binary_crossentropy(tf.ones_like(pos_dists), pos_dists, from_logits=True)
        #gnn_loss = tf.keras.metrics.binary_crossentropy(tf.zeros_like(neg_dists), neg_dists, from_logits=True)
        #gnn_loss /= 2 
        #gnn_loss =  tf.reduce_mean(tf.where(
        #                            tf.equal(y_true, 1),
        #                                1-y_pred,
        #                            tf.maximum(tf.zeros_like(y_pred), y_pred)))
        #print(tf.reduce_mean(pos_dists).numpy(), tf.reduce_mean(neg_dists).numpy())
        #gnn_loss = tf.reduce_mean(pos_dists) + tf.reduce_mean(tf.maximum(tf.zeros_like(neg_dists), 1-neg_dists))
        return gnn_loss
    
    def train_vae(self, nodes_idx, vae=True):
        if not vae: # not variational, no kld loss
            # make logvar non trainable
            layer_names = [layer.name for layer in self.encoder.layers[0].layers]
            logvar_idx = layer_names.index("logvar")
            self.encoder.layers[0].layers[logvar_idx].trainable = False
            #ae_embs = tf.concat((self.features[:,:self.abundance_dim], self.encoder(self.features)[0]), axis=1)

        ae_mu, ae_logvar = self.encoder(tf.gather(self.features, nodes_idx), training=(self.epoch < self.pretrainvae or self.pretrainvae == 0))
        #ae_mu, ae_logvar = self.encoder(tf.gather(self.features, nodes_idx), training=True)
        if self.decoder_input == "gnn" and self.use_gnn and (self.epoch > self.pretrainvae or self.pretrainvae == 0):
            ae_mu = tf.scatter_nd(indices=nodes_idx[:,None],
                                     updates=ae_mu,
                                     shape=(self.features.shape[0], ae_mu.shape[1]))
            ae_mu = self.gnn_model(ae_mu, nodes_idx, training=True)
        ae_recon = self.decoder(ae_mu, training=(self.epoch < self.pretrainvae or self.pretrainvae == 0))
        #ae_recon = self.decoder(ae_mu, training=True)
        ae_logvar = tf.clip_by_value(ae_logvar, -2, 2)
        try:
            losses = self.ae_loss(tf.gather(self.features, nodes_idx), ae_recon, ae_mu, ae_logvar, vae=vae)
        except:
            breakpoint()
        ae_loss = tf.reduce_sum(losses) * self.ae_weight
        ae_losses = {"kmer_loss": losses[0],
                        "ab_loss": losses[1],
                        "kld": losses[2],
                        "vae_loss": ae_loss,
                        "mean_logvar": tf.reduce_mean(ae_logvar)}
        if ae_losses["kld"] < -1:
            breakpoint()
        #logvar_idx = layer_names.index("logvar")
        #self.encoder.layers[0].layers[logvar_idx].trainable = False
        #ae_embs = tf.concat((self.features[:,:self.abundance_dim], self.encoder(self.features)[0]), axis=1)
        #ae_embs = self.encoder(self.features)[0]
        return ae_mu, ae_losses

    def train_edges(self, node_hat, nodes_idx, edges_idx, train_pairs):
        """_summary_

        :param node_hat: node embeddings
        :type node_hat: _type_
        :param nodes_idx: node idx to be considered in this batch
        :type nodes_idx: _type_
        :param edges_idx: edge idx to be considered in this batch
        :type edges_idx: _type_
        :param train_pairs: graph edges, pairs of node idxs of this batch
        :type train_pairs: 
        :return: _description_
        :rtype: _type_
        """
        gnn_losses = {"gnn_loss": tf.constant(0, dtype=tf.float32),
                      "pos": tf.constant(0, dtype=tf.float32),
                      "neg": tf.constant(0, dtype=tf.float32)}
        if self.gnn_weight > 0:
            #breakpoint()
            src_embs = tf.gather(indices=train_pairs[0], params=node_hat)
            dst_embs = tf.gather(indices=train_pairs[1], params=node_hat)
            positive_pairwise_dist = self.nodedist(src_embs, dst_embs)
            # create random negatives for gnn_loss
            batch_neg_idx_src, batch_neg_idx_dst = self.sample_negatives(edge_idx=edges_idx,
                                                                node_idx=nodes_idx)
            if self.loglevel == "debug":
                pset = set(zip(train_pairs[0].numpy(), train_pairs[1].numpy()))
                nset = set(zip(batch_neg_idx_src.numpy(), batch_neg_idx_dst.numpy()))
                print("false random negatives", round(len(pset & nset)/len(edges_idx), 4))
            
            try:
                #negative_pairs = tf.gather_nd(pairwise_similarity, neg_idx)
                neg_row_embs = tf.gather(indices=batch_neg_idx_src, params=node_hat)
                neg_col_embs = tf.gather(indices=batch_neg_idx_dst, params=node_hat)
                negative_pairwise_dist = self.nodedist(neg_row_embs, neg_col_embs)
            except:
                breakpoint()
            if self.use_noise:
                # reduce noises to one dim
                src_ab = tf.gather(indices=train_pairs[0], params=self.features )[:, :self.ab_dim]
                dst_ab = tf.gather(indices=train_pairs[1], params=self.features )[:, :self.ab_dim]
                noise_input = tf.concat((src_ab, dst_ab), axis=1)
                pos_noises = self.noise_model(noise_input)[:,0]
                positive_pairwise_dist = positive_pairwise_dist + pos_noises

                src_ab = tf.gather(indices=batch_neg_idx_src, params=self.features )[:, :self.ab_dim]
                dst_ab = tf.gather(indices=batch_neg_idx_dst, params=self.features )[:, :self.ab_dim]
                noise_input = tf.concat((src_ab, dst_ab), axis=1)
                neg_noises = self.noise_model(noise_input)[:,0]
                negative_pairwise_dist = negative_pairwise_dist + neg_noises
                self.noise_dict.update({(train_pairs[0][i].numpy(), train_pairs[1][i].numpy()): pos_noises[i].numpy() for i in range(len(edges_idx))})
                self.noise_dict.update({(batch_neg_idx_src[i].numpy(), batch_neg_idx_dst[i].numpy()): neg_noises[i].numpy() for i in range(len(batch_neg_idx_src))})



            pos_dist = tf.reduce_mean(positive_pairwise_dist)
            if self.num_negatives > 0:
                neg_dist = tf.reduce_mean(negative_pairwise_dist)
            else:
                neg_dist = tf.constant(0)
            #gnn_loss = (pos_loss + neg_loss) * self.gnn_weight
            #neg_loss = 0
            gnn_losses["pos"] = pos_dist
            gnn_losses["neg"] = neg_dist
            gnn_loss = self.edge_loss(positive_pairwise_dist, negative_pairwise_dist)
            gnn_loss = gnn_loss * self.gnn_weight
            gnn_losses["gnn_loss"] = gnn_loss
        #loss = gnn_loss
        return gnn_losses

    def train_scg(self, node_hat, scgs_idx):
        scg_loss = tf.constant(0, dtype=tf.float32)
        if self.scg_pairs is not None and self.scg_weight > 0:
            scg_row_embs = tf.gather(node_hat, self.scg_pairs[scgs_idx, 0])
            scg_col_embs = tf.gather(node_hat, self.scg_pairs[scgs_idx, 1])
            scg_pairwise = self.nodedist(scg_row_embs, scg_col_embs)
            #scg_pairwise = tf.nn.sigmoid(scg_pairwise)
            #if self.use_noise:
            #    scg_noises = tf.matmul(tf.gather(indices=scgs_idx, params=self.scg_noises), noises_weights)
            #    scg_pairwise = scg_pairwise + scg_noises[:,0]
                #scg_pairwise = tf.clip_by_value(scg_pairwise, 0, 1)
            #scg_loss = tf.keras.losses.binary_crossentropy(
            #        tf.zeros_like(scg_pairwise), scg_pairwise, from_logits=True
            #)
            scg_loss = self.edge_loss(tf.ones_like([0.0]), scg_pairwise)
            scg_loss *= self.scg_weight
        return scg_loss

    @tf.function
    def train_scg_only(self):
        scgs_idx = range(0, len(self.scg_pairs))
        layer_names = [layer.name for layer in self.encoder.layers[0].layers]
        logvar_idx = layer_names.index("logvar")
        self.encoder.layers[0].layers[logvar_idx].trainable = False
        with tf.GradientTape() as tape:

            node_hat, _ = self.encoder(self.features, training=True)
            scg_loss = self.train_scg(node_hat, scgs_idx)
            # aggregate model weights
            if self.use_gnn and (self.epoch > self.pretrainvae or self.pretrainvae ==  0) :
                tw = self.gnn_model.trainable_weights
            else:
                tw = []
            if self.use_ae:
                if self.pretrainvae == 0 or self.epoch < self.pretrainvae:
                    tw += self.encoder.trainable_weights
                #self.encoder.layers[0].layers[logvar_idx].trainable = True
            grads = tape.gradient(scg_loss, tw)

            self.opt.apply_gradients(zip(grads, tw))
        self.encoder.layers[0].layers[logvar_idx].trainable = True
        return scg_loss


    @tf.function
    def train_unsupervised(self, nodes_idx=None, edges_idx=None,
                            scgs_idx=None,training=True, vae=True,
                            mask_labels=None, gold_labels=None, last_batch=False):
        #### get node indices to be used for this batch
        if edges_idx is None:
            edges_idx = range(0,self.gnn_model.adj.indices.shape[0])
        if nodes_idx is None:
            # get nodes_idx from edges_idx
            # this are the node pairs with their original indices, e.g. (1,2), (2,3), (3,1)
            train_src_original = tf.gather(indices=edges_idx, params=self.gnn_model.adj.indices[:,0])
            train_dst_original = tf.gather(indices=edges_idx, params=self.gnn_model.adj.indices[:,1])
            unique_nodes = tf.unique(tf.concat((train_src_original, train_dst_original), axis=0))
            nodes_idx = unique_nodes.y # e.g. (A,B,C)
            # get new indices for edges in relation to current node list(A,B), (B,C), (C,A)
            #train_src_new = unique_nodes.idx[:train_src_original.shape[0]] 
            #train_dst_new = unique_nodes.idx[train_src_original.shape[0]:]
            #train_idx_new = (train_src_new, train_dst_new)
            if self.loglevel == "debug":
                print(f"using {nodes_idx.shape} nodes for this batch")
        else:
            train_idx_new = (self.gnn_model.adj.indices[:,0], self.gnn_model.adj.indices[:,1])
            train_src_original = self.gnn_model.adj.indices[:,0]
            train_dst_original = self.gnn_model.adj.indices[:,1]
        if scgs_idx is None:
            scgs_idx = range(0, len(self.scg_pairs))
        #####
        
        with tf.GradientTape() as tape:
            #####   run encoder first on nodes of this batch
            if self.use_ae:
                # only nodes in nodes_idx are processed, the output may have a different dimension
                ae_embs, ae_losses = self.train_vae(nodes_idx)
                # ae_embs is only nodes in node_idx
                #reverse gather, expand so that ae_embs has the same dim as self.features
                ae_embs = tf.scatter_nd(indices=nodes_idx[:,None],
                                     updates=ae_embs,
                                     shape=(self.features.shape[0], ae_embs.shape[1]))
            else:
                ae_embs = self.features # ae_embs is all nodes
                ae_losses = {}
            ######
            ###### run gnn model
            if self.use_gnn and self.decoder_input == "vae" and (self.epoch > self.pretrainvae or self.pretrainvae ==  0):
                gnn_embs = self.gnn_model(ae_embs, nodes_idx, training=True)
                node_hat = tf.scatter_nd(indices=nodes_idx[:,None],
                                     updates=gnn_embs,
                                     shape=(self.features.shape[0], gnn_embs.shape[1]))
            else:
                node_hat = ae_embs
            gnn_losses = self.train_edges(node_hat, nodes_idx, edges_idx,
                                          (train_src_original, train_dst_original))
            # 
            # SCG loss
            #if last_batch:
            #    #breakpoint()
            #    scg_loss = self.train_scg(node_hat, scgs_idx)
            #    gnn_losses["scg_loss"] = scg_loss
            #else:
            scg_loss = tf.convert_to_tensor(0.0)
            
            #
            
            # noise
            #if self.use_noise:
            #    noise_loss = self.noise_reg*(tf.reduce_sum(self.positive_noises ** 2) + tf.reduce_sum(self.scg_noises ** 2))
            #else:
            #    noise_loss = tf.constant(0, dtype=tf.float32)

            # classification loss
            if self.classifier is not None:
                if mask_labels is not None and mask_labels > 0:
                    use_labels = np.random.choice(node_hat.shape[0], int(node_hat.shape[0]*(1-self.mask_labels)))
                else:
                    use_labels = np.arange(node_hat.shape[0])
                predictions = self.classifier(node_hat, mask=use_labels)
                pred_loss = self.classifier.loss(gold_labels[use_labels], predictions)
            else:
                pred_loss = tf.convert_to_tensor(0.0)
            gnn_losses["pred_loss"] = pred_loss
            
            # combine losses and update model
            loss = gnn_losses["gnn_loss"] + scg_loss
            if training:
                # aggregate model weights
                if self.use_gnn and (self.epoch > self.pretrainvae or self.pretrainvae ==  0) :
                    tw = self.gnn_model.trainable_weights
                else:
                    tw = []
                if self.use_ae:
                    loss += ae_losses["vae_loss"]
                    if self.pretrainvae == 0 or self.epoch < self.pretrainvae:
                        tw += self.encoder.trainable_weights + self.decoder.trainable_weights # skip logvar 
                    #self.encoder.layers[0].layers[logvar_idx].trainable = True
                if self.use_noise:
                    #breakpoint()
                    #tw += [self.positive_noises, self.scg_noises]
                    tw += self.noise_model.trainable_weights
                    #loss += noise_loss
                if self.classifier is not None:
                    tw += self.classifier.trainable_weights
                    loss += pred_loss
                #################
                #grads = tape.gradient([ae_losses.get("vae_loss", tf.convert_to_tensor(0.0)), gnn_losses["gnn_loss"], scg_loss, pred_loss], tw)
                grads = tape.gradient([ae_losses.get("vae_loss", tf.convert_to_tensor(0.0)), gnn_losses["gnn_loss"], pred_loss], tw)
                grad_norm = tf.linalg.global_norm(grads)
                #clip_grads, _ = tf.clip_by_global_norm(grads, 2,  use_norm=grad_norm)
                #new_grad_norm = tf.linalg.global_norm(clip_grads)
                self.opt.apply_gradients(zip(grads, tw))
                ae_losses["grad_norm"] = grad_norm.numpy()
                #ae_losses["grad_norm_clip"] = new_grad_norm.numpy()
        return loss, gnn_losses, ae_losses
        
        
    #@tf.function
    def ae_loss(self, x, x_hat, mu, logvar, vae):
        if vae:
            epsilon = tf.random.normal(tf.shape(mu))
            z = mu + epsilon * tf.math.exp(0.5 * logvar)
            kld  = 0.5*tf.math.reduce_mean(tf.math.reduce_mean(1.0 + logvar - tf.math.pow(mu, 2) - tf.math.exp(logvar), axis=1))
            kld  = -kld * self.kld_weight
        else:
            z = mu
            kld = tf.convert_to_tensor(0.0)
        if self.abundance_dim > 1:
            mse1 = - tf.reduce_mean(tf.reduce_sum((tf.math.log(x_hat[:, :self.abundance_dim] + 1e-9) * x[:, :self.abundance_dim]), axis=1))
        else:
            mse1 = tf.reduce_mean( (x[:, :self.abundance_dim] - x_hat[:, :self.abundance_dim])**2)
        mse1 *= (1-self.kmer_weight)
        mse2 = self.kmer_weight*tf.reduce_mean( tf.reduce_mean((x[:, self.abundance_dim:] - x_hat[:, self.abundance_dim:])**2, axis=1))

        return mse2, mse1, kld

    @tf.function
    def train_unsupervised_decode(self, idx):
        with tf.GradientTape() as tape:
            # run gnn model
            z_sample, mu, logvar, x_orginal, x_hat = self.gnn_model(self.features, self.adj,
                                                           indices=idx, training=True)
 

            loss = 0
            gnn_loss = tf.constant(0, dtype=tf.float32)
            # SCG loss
            scg_loss = tf.constant(0, dtype=tf.float32)
            if self.scg_pairs is not None and self.scg_weight > 0:
                ns1 = tf.gather(z_sample, self.scg_pairs[:, 0])
                ns2 = tf.gather(z_sample, self.scg_pairs[:, 1])
                all_diff_pairs = tf.math.exp(-0.5 * tf.reduce_sum((ns1 - ns2) ** 2, axis=-1))
                scg_loss = tf.reduce_mean(all_diff_pairs) * self.scg_weight
                loss += scg_loss

            # decode
            kmer_loss, ab_loss, kld_loss = self.ae_loss(tf.gather(self.features, idx), x_hat, mu, logvar, vae=True)
            loss += kmer_loss + ab_loss - kld_loss

        tw = self.gnn_model.trainable_weights
        #tw += self.encoder.trainable_weights
        #tw += self.decoder.trainable_weights
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

