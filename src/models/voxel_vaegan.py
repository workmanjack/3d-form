# project imports
from models import MODEL_DIR


# python & package imports
from collections import defaultdict
import tensorflow as tf
import logging.config
import numpy as np
import time
import os


# set seeds for reproducibility
np.random.seed(12)
tf.set_random_seed(12)


class VoxelVaegan():
    
    def __init__(self, input_dim, latent_dim, learning_rate, keep_prob, verbose=False, 
                 kl_div_loss_weight=5, recon_loss_weight=5e-4, debug=False, ckpt_dir='voxel_vaegan'):
        """
        Args:
            input_dim: int, dimension of voxels to process
            latent_dim: int, size of latent vector
            learning_rate: float, learning rate of optimizer
            keep_prob: float, prob of keeping weights for dropout layer
            verbose: bool, flag on amount of printing to do
            kl_div_loss_weight: float, weight for KL Divergence loss when computing total loss
            recon_loss_weight: float, weight for reconstruction loss when computing total loss
            debug: bool, flag on whether to log debug info or not
            ckpt_dir: str, name of output dir for model ckpts

        """ 
        logging.info('Initializing VoxelVaegan')
        # network and training params
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.verbose = verbose
        self.debug = debug
        self.kl_div_loss_weight = kl_div_loss_weight
        self.recon_loss_weight = recon_loss_weight
        
        self._input_x = tf.placeholder(tf.float32, shape=(None, self.input_dim, self.input_dim, self.input_dim, 1))
        self._keep_prob = tf.placeholder(dtype=tf.float32)
        self._trainable = tf.placeholder(dtype=tf.bool)

        # add ops to this list as a tuple with (<op name>, <op>) to see them executed, returned, and printed
        # to console during execution
        self._debug_ops = list()
        
        # Construct the TensorFlow Graph
        self.encoder, self.enc_mu, self.enc_sig = self._make_encoder(self._input_x, self._keep_prob, self._trainable)
        self.decoder = self._make_decoder(self.encoder, self._trainable)
        
        self.D_real, self.D_real_logits = self._discriminator(self._input_x, self._trainable)
        self.D_fake, self.D_fake_logits = self._discriminator(self.decoder, self._trainable)

        # VAE loss
        self.loss, self.optimizer, self.mean_recon, self.mean_kl = self._make_loss(self._input_x, self.decoder,
                                                                                   self.enc_mu, self.enc_sig)
        # GAN loss
        self.D_loss, self.G_loss, self.D_optim, self.G_optim = self._make_GAN_loss(self.D_real_logits, self.D_fake_logits)
    
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()
        
        # Setup Model Saving
        self.ckpt_dir = os.path.join(MODEL_DIR, ckpt_dir)
        self.saver = tf.train.Saver()
        self.recons_pre = list()
        self.recons_post = list()
        self.metrics = defaultdict(dict)

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        
    def _log_shape(self, tensor, name=None):
        if self.verbose:
            if not name:
                name = tensor.name
            logging.debug('{}: {}'.format(name, tensor.shape))
        return
    
    def _make_encoder(self, input_x, keep_prob, trainable):
        
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        
            # tf conv3d: https://www.tensorflow.org/api_docs/python/tf/layers/conv3d
            # tf glorot init: https://www.tensorflow.org/api_docs/python/tf/glorot_uniform_initializer
            conv1 = tf.layers.batch_normalization(tf.layers.conv3d(input_x,
                                     filters=8,
                                     kernel_size=[3, 3, 3],
                                     strides=(1, 1, 1),
                                     padding='valid',
                                     activation=tf.nn.elu,
                                     kernel_initializer=tf.initializers.glorot_uniform()),
                                     name='enc_conv1')
            self._log_shape(conv1)

            # the Example VAE specifies the activation functions as part of the layer
            # we specify the activation function as a seperate tensor
            # it is unknown if this is the preferred method in Tensorflow, but we know
            # it works from work in the 3D-VAE-GAN notebook
            # we also take advantage of batch_normalization
            # more info here:
            # https://medium.com/@ilango100/batch-normalization-speed-up-neural-network-training-245e39a62f85
            # with the hope that it gives speed without sacrificing quality
            # tf batch norm: https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
            # tf elu (exponential linear unit): https://www.tensorflow.org/api_docs/python/tf/nn/elu

            conv2 = tf.layers.batch_normalization(tf.layers.conv3d(conv1,
                                     filters=16,
                                     kernel_size=[3, 3, 3],
                                     strides=(2, 2, 2),
                                     padding='same',
                                     activation=tf.nn.elu,
                                     kernel_initializer=tf.initializers.glorot_uniform()))
            self._log_shape(conv2)

            conv3 = tf.layers.batch_normalization(tf.layers.conv3d(conv2,
                                     filters=32,
                                     kernel_size=[3, 3, 3],
                                     strides=(1, 1, 1),
                                     padding='valid',
                                     activation=tf.nn.elu,
                                     kernel_initializer=tf.initializers.glorot_uniform()))
            self._log_shape(conv3)

            conv4 = tf.layers.batch_normalization(tf.layers.conv3d(conv3,
                                     filters=64,
                                     kernel_size=[3, 3, 3],
                                     strides=(2, 2, 2),
                                     padding='same',
                                     activation=tf.nn.elu,
                                     kernel_initializer=tf.initializers.glorot_uniform()))
            self._log_shape(conv4)

            # Apply one fully-connected layer after Conv3d layers
            # tf dense layer: https://www.tensorflow.org/api_docs/python/tf/layers/dense
            dense1 = tf.layers.batch_normalization(tf.layers.dense(conv4,
                                 units=343,
                                 activation=tf.nn.elu,
                                 kernel_initializer=tf.initializers.glorot_uniform()))
            self._log_shape(dense1)

            # Apply dropout
            flatten = tf.layers.flatten(tf.nn.dropout(dense1, keep_prob))

            # Calculate outputs
            enc_mu = tf.layers.batch_normalization(tf.layers.dense(flatten,
                                 units=self.latent_dim,
                                 activation=None))
            self._log_shape(enc_mu)
            enc_sig = tf.layers.batch_normalization(tf.layers.dense(flatten,
                                 units=self.latent_dim,
                                 activation=None))
            self._log_shape(enc_sig)

            # epsilon is a random draw from the latent space
            epsilon = tf.random_normal(tf.stack([tf.shape(dense1)[0], self.latent_dim]))
            self._log_shape(epsilon, 'epsilon')
            enc_z = enc_mu + tf.multiply(epsilon, tf.exp(enc_sig))
            self._log_shape(enc_z, 'z')

        return enc_z, enc_mu, enc_sig

    def _make_decoder(self, input_z, trainable):
        
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):

            # There is some magic in the Example VAE that adds conditional input based on the
            # class of the image. We do not have that luxury as we are attempting to do this
            # with input that lacks classes.
            # TODO: if poor results, try classes
            self._log_shape(input_z, 'input_z')

            # Why conv3d_transpose instead of conv3d?
            #
            # from https://www.tensorflow.org/api_docs/python/tf/nn/conv3d_transpose,
            #     "This operation is sometimes called "deconvolution" after Deconvolutional Networks,
            #      but is actually the transpose (gradient) of conv3d rather than an actual deconvolution."
            #
            # conv3d_transpose: https://www.tensorflow.org/api_docs/python/tf/layers/conv3d_transpose
            dense1 = tf.layers.dense(input_z,
                                     units=343,
                                     kernel_initializer=tf.initializers.glorot_uniform(),
                                     name='dec_dense1')
            self._log_shape(dense1)
            lrelu1 = tf.nn.elu(tf.layers.batch_normalization(dense1, training=trainable))
            self._log_shape(lrelu1)

            #z = tf.reshape(z, (-1, 1, 1, 1, n_latent))
            reshape_z = tf.reshape(lrelu1, shape=(-1, 7, 7, 7, 1), name='reshape_z')
            self._log_shape(reshape_z)

            conv1 = tf.layers.batch_normalization(tf.layers.conv3d_transpose(reshape_z,
                                               filters=64,
                                               kernel_size=[3, 3, 3],
                                               strides=(1, 1, 1),
                                               padding='same',
                                               activation=tf.nn.elu,
                                               # Example VAE does not mention bias
                                               use_bias=False,
                                               kernel_initializer=tf.initializers.glorot_uniform(),
                                               name='dec_conv1'))
            self._log_shape(conv1)

            conv2 = tf.layers.batch_normalization(tf.layers.conv3d_transpose(conv1,
                                               filters=32,
                                               kernel_size=[3, 3, 3],
                                               # Example VAE used .5 stride values, but Tensorflow complains
                                               # of being forced to use a float value here
                                               #strides=(1.0 / 2, 1.0 / 2, 1.0 / 2),
                                               strides=(2, 2, 2),
                                               padding='valid',
                                               activation=tf.nn.elu,
                                               use_bias=False,
                                               kernel_initializer=tf.initializers.glorot_uniform(),
                                               name='dec_conv2'))
            self._log_shape(conv2)

            conv3 = tf.layers.batch_normalization(tf.layers.conv3d_transpose(conv2,
                                               filters=16,
                                               kernel_size=[3, 3, 3],
                                               strides=(1, 1, 1),
                                               # changed to valid to hit correct dimension
                                               padding='same',
                                               activation=tf.nn.elu,
                                               use_bias=False,
                                               kernel_initializer=tf.initializers.glorot_uniform(),
                                               name='dec_conv3'))
            self._log_shape(conv3)

            conv4 = tf.layers.batch_normalization(tf.layers.conv3d_transpose(conv3,
                                               filters=8,
                                               kernel_size=[4, 4, 4],
                                               #strides=(1.0 / 2, 1.0 / 2, 1.0 / 2),
                                               strides=(2, 2, 2),
                                               padding='valid',
                                               activation=tf.nn.elu,
                                               use_bias=False,
                                               kernel_initializer=tf.initializers.glorot_uniform(),
                                               name='dec_conv4'))
            self._log_shape(conv4)

            conv5 = tf.layers.conv3d_transpose(conv4,
                                               filters=1,
                                               kernel_size=[3, 3, 3],
                                               strides=(1, 1, 1),
                                               padding='same',
                                               use_bias=False,
                                               kernel_initializer=tf.initializers.glorot_uniform(),
                                               name='dec_conv5')
            self._log_shape(conv5)

            #decoded_output = tf.nn.tanh(conv5)
            decoded_output = tf.nn.sigmoid(conv5)
            #decoded_output = tf.clip_by_value(decoded_output, 1e-7, 1.0 - 1e-7)
            #self._add_debug_op('max decoded_output', tf.math.reduce_max(decoded_output), False)
            #self._add_debug_op('min decoded_output', tf.math.reduce_min(decoded_output), False)
            #self._add_debug_op('mean decoded_output', tf.math.reduce_mean(decoded_output), False)
            self._log_shape(decoded_output)

        return decoded_output
    
    def _discriminator(self, input_x, trainable):

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

            # need to clip the values?
            self._log_shape(input_x, 'input_x')

            # 1st hidden layer
            conv1 = tf.layers.conv3d(input_x, 128, [3, 3, 3], strides=(2, 2, 2), padding='same', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            lrelu1 = tf.nn.elu(conv1)
            self._log_shape(lrelu1)
            # 2nd hidden layer
            conv2 = tf.layers.conv3d(lrelu1, 256, [3, 3, 3], strides=(2, 2, 2), padding='same', use_bias=False,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            lrelu2 = tf.nn.elu(tf.layers.batch_normalization(conv2, training=trainable))
            self._log_shape(lrelu2)
            # 3rd hidden layer
            conv3 = tf.layers.conv3d(lrelu2, 512, [3, 3, 3], strides=(2, 2, 2), padding='same', use_bias=False,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            lrelu3 = tf.nn.elu(tf.layers.batch_normalization(conv3, training=trainable))
            self._log_shape(lrelu3)
            # output layer
            #conv4 = tf.layers.conv3d(lrelu3, 1, [4, 4, 4], strides=(1, 1, 1), padding='valid', use_bias=False,
            #                          kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv4 = tf.layers.conv3d(lrelu3, 1, [3, 3, 3], strides=(1, 1, 1), padding='valid', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            o = tf.nn.sigmoid(conv4)
            self._log_shape(conv4)
            self._log_shape(o)

        return o, conv4
    
    def _make_loss(self, enc_input, dec_output, z_mu, z_sig):
        """
        Info on loss in VAE:
          * https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
          
        Args:
            enc_input: tensor, input tensor into VAE
            dec_output: tensor, decoded output tensor from VAE

        Return:
            float, 
        """
        # Weighted binary cross-entropy for use in voxel loss. Allows weighting of false positives relative to false negatives.
        # Nominally set to strongly penalize false negatives
        # we must clip because values of 0 or 1 will cause errors
        clipped_input = tf.clip_by_value(enc_input, 1e-7, 1.0 - 1e-7)
        clipped_output = tf.clip_by_value(dec_output, 1e-7, 1.0 - 1e-7)
        #self._add_debug_op('max clipped_input', tf.math.reduce_max(clipped_input), False)
        #self._add_debug_op('min clipped_input', tf.math.reduce_min(clipped_input), False)
        #self._add_debug_op('mean clipped_input', tf.math.reduce_mean(clipped_input), False)
        self._add_debug_op('max clipped_output', tf.math.reduce_max(clipped_output), False)
        self._add_debug_op('min clipped_output', tf.math.reduce_min(clipped_output), False)
        self._add_debug_op('mean clipped_output', tf.math.reduce_mean(clipped_output), False)
        bce = -(98.0 * clipped_input * tf.log(clipped_output) + 2.0 * (1.0 - clipped_input) * tf.log(1.0 - clipped_output)) / 100.0
        #self._add_debug_op('bce', bce, False)
        #bce = tf.keras.backend.binary_crossentropy(enc_output, dec_output)
        
        # Voxel-Wise Reconstruction Loss 
        # Note that the output values are clipped to prevent the BCE from evaluating log(0).
        recon_loss = tf.reduce_mean(bce, 1)
   
        #recon_loss = tf.reduce_sum(tf.squared_difference(
        #    tf.reshape(dec_output, (-1, self.input_dim ** 3)),
        #    tf.reshape(self._input_x, (-1, self.input_dim ** 3))), 1)
        
        kl_divergence = -0.5 * tf.reduce_sum(1.0 + 2.0 * z_sig - z_mu ** 2 - tf.exp(2.0 * z_sig), 1)

        mean_kl = tf.reduce_sum(kl_divergence)
        #self._add_debug_op('mean_kl', mean_kl, False)
        mean_recon = tf.reduce_sum(recon_loss)
        #self._add_debug_op('mean_recon', mean_recon, False)

        # tf reduce_mean: https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
        loss = tf.reduce_mean(self.kl_div_loss_weight * kl_divergence + self.recon_loss_weight * recon_loss)
        #self._add_debug_op('loss', loss, False)
        
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True).minimize(loss)

        return loss, optimizer, mean_recon, mean_kl

    def _make_GAN_loss(self, D_real_logits, D_fake_logits):
        """
        From https://github.com/zhangqianhui/vae-gan-tensorflow/blob/master/vaegan.py
        
        sigmoid_cross_entropy
        """

        D_loss_real = tf.reduce_mean(D_real_logits)
        D_loss_fake = tf.reduce_mean(D_fake_logits)
        D_loss = D_loss_real - D_loss_fake
        G_loss = -tf.reduce_mean(D_fake_logits)
        
        D_optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(-D_loss)
        G_optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(G_loss)
        
        return D_loss, G_loss, D_optim, G_optim

    def _add_debug_op(self, name, op, newline=True):
        self._debug_ops.append((name, op, newline))
        return

    def _log_debug_ops(self, results):
        if self.debug:
            for i, debug_op in enumerate(self._debug_ops):
                msg = 'DEBUG_OP "{}": '.format(debug_op[0])
                if len(debug_op) > 2 and debug_op[2]:
                    msg += '\n'
                msg += '{}'.format(results[i])
                logging.debug(msg)
        return

    def train(self, generator, epochs=10, input_repeats=1, display_step=1, save_step=1, viz_data=None):
        
        start = time.time()
        for epoch_num, epoch in enumerate(range(epochs)):

            for batch_num, batch in enumerate(generator()):
                
                if self.verbose:
                    logging.debug('Epoch: {}, Batch: {}, Elapsed time: {:.2f} mins'.format(epoch_num, batch_num, (time.time() - start) / 60))
                # repeat for extra practice on each shape
                for _ in range(input_repeats):
                    
                    ops = tuple([self.optimizer, self.loss, self.mean_kl, self.mean_recon
                                 , self.D_loss, self.G_loss, self.D_optim, self.G_optim] + 
                                [op for name, op, _ in self._debug_ops])

                    results = self.sess.run(
                        ops,
                        feed_dict={self._input_x: batch, self._keep_prob:self.keep_prob, self._trainable: True}
                    )
                    _, loss, kl_divergence, recon_loss, D_loss, G_loss, _, _ = results[:8]
                    self._log_debug_ops(results[8:])
                
                if self.verbose:
                    #print('\tKL Divergence = {:.5f}, Reconstruction Loss = {:.5f}'.format(kl_divergence, recon_loss))
                    print('\tKL Divergence = {}, Reconstruction Loss = {}, -D_loss = {}, G_loss = {}'.format(
                        kl_divergence, recon_loss, -D_loss, G_loss))

            elapsed_time = (time.time() - start) / 60
            # save the epoch's data for review later
            self.metrics['epoch' + str(epoch)] = {
                'loss': loss,
                'kl_divergence': kl_divergence,
                'reconstruction_loss': recon_loss,
                'elapsed_time': elapsed_time 
            }

            if (epoch + 1) % display_step == 0:
                print("Epoch: {}, ".format(epoch + 1) + 
                      "Loss = {:.5f}, ".format(loss) + 
                      "KL Divergence = {:.5f}, ".format(kl_divergence) +
                      "Reconstruction Loss = {:.5f}, ".format(recon_loss) +
                      "-D_Loss = {:.5f}, ".format(-D_loss) +
                      "G_Loss = {:.5f}, ".format(G_loss) +
                      "Elapsed time: {:.2f} mins".format((time.time() - start) / 60))
                print('Generation Example:')
                
                # prepare for generation
                if viz_data is not None:
                    logging.info('Generation Example:')
                    self._log_shape(viz_data, 'Example shape (before reshape)')
                    recon_input = np.reshape(viz_data, (1, self.input_dim, self.input_dim, self.input_dim, 1))
                    self._log_shape(recon_input, 'Example shape')

                    # generate!
                    recon = self.reconstruct(recon_input)
                    self._log_shape(recon, 'Recon')

                    # prepare for plotting
                    recon_input = np.reshape(recon_input, (self.input_dim, self.input_dim, self.input_dim))
                    self._log_shape(recon_input, 'Example shape (for plotting)')
                    recon = np.reshape(recon, (self.input_dim, self.input_dim, self.input_dim))
                    self._log_shape(recon, 'Recon (for plotting)')
                    # network outputs decimals; here we force them to True/False for plotting
                    self.recons_pre.append(recon)
                    recon = recon > 0.5
                    self.recons_post.append(recon)
                    # replace all nans with zeros
                    #recon = np.nan_to_num(recon)

                    # save the generated object in case we wish to review later
                    path = os.path.join(self.ckpt_dir, 'recon_epoch-{}.npy'.format(epoch))

                    # visualize
                    self.visualize_reconstruction(recon_input, recon)

            if (epoch + 1) % save_step == 0:
                # Save the variables to disk.
                save_path = self.saver.save(self.sess, os.path.join(self.ckpt_dir, "model_epoch-{}.ckpt".format(epoch)))
                logging.info("Model saved in path: {}".format(save_path))
                                       
        return

    def restore(self, model_ckpt):
        self.saver.restore(self.sess, model_ckpt)
        return
    
    def close(self):
        self.sess.close()
        return
            
    def reconstruct(self, input_x):
        """
        Use VAE to reconstruct given data
        """
        ops = tuple([self.decoder] + [op for name, op, _ in self._debug_ops])
                    
        results = self.sess.run(ops, 
            feed_dict={self._input_x: input_x, self._keep_prob: 1.0, self._trainable: False})
        
        decoded = results[0]
        self._log_debug_ops(results[1:])
                    
        return decoded
    
    def visualize_reconstruction(self, original_x, reconstructed_x, name=None):
        """
        This function was used to visualize the output of each epoch during training.
        This functionality is being moved to the train_vae.py script.
        """
        title = '' if not name else ': {}'.format(name)
        plot_voxels(original_x, title='Original' + title)
        plot_voxels(reconstructed_x, title='Autoencoded' + title)
        return

    def __repr__(self):
        return '<VariationalAutoencoder(input_dim={}, latent_dim={}, learning_rate={}, keep_prob={})>'.format(
            self.input_dim, self.latent_dim, self.learning_rate, self.keep_prob)
