# project imports
from utils import elapsed_time, memory


# python & package imports
from collections import defaultdict
import tensorflow as tf
import logging.config
import numpy as np
import json
import math
import time
import os


# set seeds for reproducibility
np.random.seed(12)
tf.set_random_seed(12)


class VoxelVaegan():
    
    SCOPE_ENCODER = 'encoder'
    SCOPE_DECODER = 'decoder'
    SCOPE_DISCRIMINATOR = 'discriminator'
    
    def __init__(self, input_dim, latent_dim, keep_prob, verbose=False, no_gan=False,
                 kl_div_loss_weight=5, recon_loss_weight=5e-4, ll_weight=.001, dec_weight=10,
                 debug=False, ckpt_dir='voxel_vaegan', tb_dir='tb'):
        """
        Args:
            input_dim: int, dimension of voxels to process
            latent_dim: int, size of latent vector
            enc_lr: float, learning rate of encoder optimizer
            dec_lr: float, learning rate of decoder optimizer
            dis_lr: float, learning rate of discriminator optimizer
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
        self.keep_prob = keep_prob
        self.verbose = verbose
        self.no_gan = no_gan
        self.debug = debug
        self.kl_div_loss_weight = kl_div_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.ll_weight = ll_weight
        self.dec_weight = dec_weight
        self.step = 0
        
        self._input_x = tf.placeholder(tf.float32, shape=(None, self.input_dim, self.input_dim, self.input_dim, 1))
        self._keep_prob = tf.placeholder(dtype=tf.float32)
        self._trainable = tf.placeholder(dtype=tf.bool)
        self._lr_enc = tf.placeholder(tf.float32, shape=[])
        self._lr_dec = tf.placeholder(tf.float32, shape=[])
        self._lr_dis = tf.placeholder(tf.float32, shape=[])

        # record the learning rates
        tf.summary.scalar('enc_current_lr', self._lr_enc)
        tf.summary.scalar('dec_current_lr', self._lr_dec)
        tf.summary.scalar('dis_current_lr', self._lr_dis)
        
        # add ops to this list as a tuple with (<op name>, <op>) to see them executed, returned, and printed
        # to console during execution
        self._debug_ops = list()
        
        # Construct the TensorFlow Graph
        self.encoder, self.enc_mu, self.enc_sig = self._make_encoder(self._input_x, self._keep_prob, self._trainable)
        self.decoder = self._make_decoder(self.encoder, self._trainable)

        if self.no_gan:
            logging.info('Running VAE-GAN in VAE-Only Mode')
            self.enc_loss, self.enc_optim, self.mean_recon, self.mean_kl = self._make_encoder_loss(self._input_x, self.decoder,
                                                                           self.enc_mu, self.enc_sig)

        else:
            self.dec_random = self._make_decoder(self._random_latent(), self._trainable)
            self.dis_input_output, self.dis_input_ll = self._discriminator(self._input_x, self._trainable)
            self.dis_dec_output, self.dis_dec_ll = self._discriminator(self.dec_random, self._trainable)
            
            self.dis_loss, self.dis_optim = self._make_discriminator_loss(self.dis_input_output, self.dis_dec_output)
            
            self.ll_loss = self._make_ll_loss(self.dis_input_ll, self.dis_dec_ll)
            
            self.dec_loss, self.dec_optim = self._make_decoder_loss(self.dec_random, self.ll_loss)
            self.enc_loss, self.enc_optim, self.mean_recon, self.mean_kl = self._make_encoder_loss(self._input_x, self.decoder,
                                                                           self.enc_mu, self.enc_sig, ll_loss=self.ll_loss)

    
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()
        
        # Setup Model Saving
        self.ckpt_dir = ckpt_dir
        self.tb_dir = os.path.join(self.ckpt_dir, tb_dir)
        os.makedirs(tb_dir, exist_ok=True)
        self.saver = tf.train.Saver()
        self.recons_pre = list()
        self.recons_post = list()
        self.metrics = defaultdict(dict)

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    @classmethod
    def initFromCfg(cls, cfg):
        cfg_model = cfg.get('model')
        vaegan = cls(input_dim=cfg_model.get('voxels_dim'),
                     latent_dim=cfg_model.get('latent_dim'),
                     keep_prob=cfg_model.get('keep_prob'),
                     kl_div_loss_weight=cfg_model.get('kl_div_loss_weight'),
                     recon_loss_weight=cfg_model.get('recon_loss_weight'),
                     ll_weight=cfg_model.get('ll_weight'),
                     dec_weight=cfg_model.get('dec_weight'),
                     verbose=cfg_model.get('verbose'),
                     debug=cfg_model.get('debug'),
                     ckpt_dir=cfg_model.get('ckpt_dir'),
                     tb_dir=cfg_model.get('tb_dir'),
                     no_gan=cfg_model.get('no_gan'))
        return vaegan
        
    def _log_shape(self, tensor, name=None):
        if self.debug:
            if not name:
                name = tensor.name
            logging.debug('{}: {}'.format(name, tensor.shape))
        return
    
    def _make_encoder(self, input_x, keep_prob, trainable):
        
        with tf.variable_scope(self.SCOPE_ENCODER, reuse=tf.AUTO_REUSE):
        
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
            self._batch_size = tf.shape(dense1)[0]
            epsilon = self._random_latent()
            self._log_shape(epsilon, 'epsilon')
            enc_z = enc_mu + tf.multiply(epsilon, tf.exp(enc_sig))
            self._log_shape(enc_z, 'z')

        return enc_z, enc_mu, enc_sig

    def _make_decoder(self, input_z, trainable):
        
        with tf.variable_scope(self.SCOPE_DECODER, reuse=tf.AUTO_REUSE):

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

            decoded_output = tf.nn.sigmoid(conv5)
            #decoded_output = tf.clip_by_value(decoded_output, 1e-7, 1.0 - 1e-7)
            #self._add_debug_op('max decoded_output', tf.math.reduce_max(decoded_output), False)
            #self._add_debug_op('min decoded_output', tf.math.reduce_min(decoded_output), False)
            #self._add_debug_op('mean decoded_output', tf.math.reduce_mean(decoded_output), False)
            self._log_shape(decoded_output)

        return decoded_output
    
    def _discriminator(self, input_x, trainable):
        """
        Thank you: https://github.com/Spartey/3D-VAE-GAN-Deep-Learning-Project/blob/master/3D-VAE-WGAN/model.py
        """

        with tf.variable_scope(self.SCOPE_DISCRIMINATOR, reuse=tf.AUTO_REUSE):

            # need to clip the values?
            self._log_shape(input_x, 'input_x')

            # 1st hidden layer
            conv1 = tf.layers.conv3d(input_x, 128, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            lrelu1 = tf.nn.elu(conv1)
            self._log_shape(lrelu1)

            # 2nd hidden layer
            conv2 = tf.layers.conv3d(lrelu1, 256, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            lrelu2 = tf.nn.elu(tf.layers.batch_normalization(conv2, training=trainable))
            self._log_shape(lrelu2)

            # 3rd hidden layer
            conv3 = tf.layers.conv3d(lrelu2, 512, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            lrelu3 = tf.nn.elu(tf.layers.batch_normalization(conv3, training=trainable))
            self._log_shape(lrelu3)
            
            # output layer
            #conv4 = tf.layers.conv3d(lrelu3, 1, [4, 4, 4], strides=(1, 1, 1), padding='valid', use_bias=False,
            #                          kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv4 = tf.layers.conv3d(lrelu3, 1, [4, 4, 4], strides=(1, 1, 1), padding='valid', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            self._log_shape(conv4)
            
            flatten = tf.layers.flatten(conv4)
            self._log_shape(flatten)
        
            lth_layer = tf.layers.dense(flatten,
                                        units=1024,
                                        kernel_initializer=tf.initializers.glorot_uniform(),
                                        name='lth_layer')
            lth_relu = tf.nn.elu(tf.layers.batch_normalization(lth_layer, training=trainable))
            self._log_shape(lth_relu)
            
            D = tf.layers.dense(lth_relu,
                                units=1,
                                kernel_initializer=tf.initializers.glorot_uniform(),
                                name='D')
            self._log_shape(D)

        return D, lth_relu
    
    def _get_vars_by_scope(self, scope):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    
    def _make_decoder_loss(self, dec_random, ll_loss):
        
        # old
        #dec_loss = -tf.reduce_mean(dis_fake_logits)
        
        # Loss
        dec_random_loss = tf.reduce_mean(-1. * (tf.log(dec_random)))
        dec_loss = ll_loss * self.ll_weight + dec_random_loss * self.dec_weight

        #self._add_debug_op('ll_loss', ll_loss, False)
        #self._add_debug_op('dec_random_loss', dec_random_loss, False)
        #self._add_debug_op('ll_loss weighted', ll_loss * self.ll_weight, False)
        #self._add_debug_op('dec_random_loss weighted', dec_random_loss * self.dec_weight, False)        
        
        # Optimizer
        var_list = self._get_vars_by_scope(self.SCOPE_DECODER)
        dec_optim = tf.train.RMSPropOptimizer(self._lr_dec).minimize(dec_loss, var_list=var_list)

        tf.summary.scalar("dec_random_loss", dec_random_loss) 
        tf.summary.scalar("dec_loss", dec_loss) 
        
        return dec_loss, dec_optim

    def _make_ll_loss(self, dis_input_ll, dis_dec_ll):
        # Lth Layer Loss - the 'learned similarity measure'  
        ll_loss = tf.reduce_mean(tf.reduce_sum(tf.square(dis_input_ll - dis_dec_ll)))
        tf.summary.scalar("ll_loss", ll_loss)
        return ll_loss
    
    def _make_discriminator_loss(self, dis_input_output, dis_dec_output):
        """
        """
        #dis_loss = tf.reduce_mean(-1.*(tf.log(dis_input_output) + 
        #                            tf.log(1.0 - dis_dec_output)))
        dis_loss = tf.reduce_mean(-1.*(tf.log(tf.clip_by_value(dis_input_output,1e-5,1.0)) + 
                                    tf.log(tf.clip_by_value(1.0 - dis_dec_output,1e-5,1.0))))
        dis_loss = dis_loss
        
        # Optimizer
        var_list = self._get_vars_by_scope(self.SCOPE_DISCRIMINATOR)
        dis_optim = tf.train.RMSPropOptimizer(self._lr_dis).minimize(dis_loss, var_list=var_list)

        # Summaries
        tf.summary.scalar('dis_loss', dis_loss)

        return dis_loss, dis_optim

        ###
        # The below is based on
        # https://github.com/uclaacmai/Generative-Adversarial-Network-Tutorial/blob/master/Generative%20Adversarial%20Networks%20Tutorial.ipynb
        # it is for gans; we replaced it with the vaegan version above
        ###
        
        # Discriminator Loss with Dataset Input
        #self._add_debug_op('dis_real_logits', dis_real_logits, False)
        #dis_loss_real = tf.reduce_mean(dis_real_logits)
        dis_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dis_real_logits,
                labels=tf.ones_like(dis_real_logits)
            ))
        self._add_debug_op('dis_loss_real', dis_loss_real, False)

        # Discriminator Loss with VAE Generated Input
        #self._add_debug_op('dis_fake_logits', dis_fake_logits, False)
        #dis_loss_fake = tf.reduce_mean(dis_fake_logits)
        dis_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dis_fake_logits,
                labels=tf.zeros_like(dis_fake_logits)
            ))
        self._add_debug_op('dis_loss_fake', dis_loss_fake, False)

        # Combine Losses
        dis_loss = dis_loss_real + dis_loss_fake
        self._add_debug_op('dis_loss', dis_loss, False)

        # Optimizer
        # thank you: https://stackoverflow.com/questions/36533723/tensorflow-get-all-variables-in-scope
        var_list = self._get_vars_by_scope(self.SCOPE_DISCRIMINATOR)
        dis_optim = tf.train.RMSPropOptimizer(self._lr_dis).minimize(dis_loss, var_list=var_list)

        # Summaries
        tf.summary.scalar('dis_loss', dis_loss)
        tf.summary.scalar('dis_loss_real', dis_loss_real)
        tf.summary.scalar('dis_loss_fake', dis_loss_fake)

        return dis_loss, dis_optim
    
    def _make_encoder_loss(self, enc_input, dec_output, z_mu, z_sig, ll_loss=None):
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
        #self._add_debug_op('max clipped_output', tf.math.reduce_max(clipped_output), False)
        #self._add_debug_op('min clipped_output', tf.math.reduce_min(clipped_output), False)
        #self._add_debug_op('mean clipped_output', tf.math.reduce_mean(clipped_output), False)
        bce = -(98.0 * clipped_input * tf.log(clipped_output) + 2.0 * (1.0 - clipped_input) * tf.log(1.0 - clipped_output)) / 100.0
        #self._add_debug_op('bce', bce, False)
        #bce = tf.keras.backend.binary_crossentropy(enc_output, dec_output)
        
        # Voxel-Wise Reconstruction Loss 
        # Note that the output values are clipped to prevent the BCE from evaluating log(0).
        recon_loss = tf.reduce_mean(bce, 1)
        mean_recon = tf.reduce_mean(recon_loss)
   
        #recon_loss = tf.reduce_sum(tf.squared_difference(
        #    tf.reshape(dec_output, (-1, self.input_dim ** 3)),
        #    tf.reshape(self._input_x, (-1, self.input_dim ** 3))), 1)
        kl_divergence = -0.5 * tf.reduce_sum(1.0 + 2.0 * z_sig - z_mu ** 2 - tf.exp(2.0 * z_sig), 1)
        #kl_divergence = tf.reduce_sum(-0.5 * tf.reduce_sum(1 + z_sig) - tf.square(z_mu) - tf.exp(z_sig), 1)
        mean_kl = tf.reduce_mean(kl_divergence)

        #self._add_debug_op('mean_kl', mean_kl, False)

        # tf reduce_mean: https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
        if self.no_gan:
            # if no gan, we include the reconstruction loss as a factor here
            loss = tf.reduce_mean(self.kl_div_loss_weight * kl_divergence + self.recon_loss_weight * recon_loss)
        else:
            # otherwise, we use the discriminator's input
            #L_e = tf.clip_by_value(KL_loss*KL_param + LL_loss, -100, 100)
            loss = self.kl_div_loss_weight * mean_kl + ll_loss * self.ll_weight
            #self._add_debug_op('mean_kl', mean_kl, False)
            #self._add_debug_op('mean_kl weighted', self.kl_div_loss_weight * mean_kl, False)

        #self._add_debug_op('loss', loss, False)
        
        var_list = self._get_vars_by_scope(self.SCOPE_ENCODER)
        if self.no_gan:
            # if no gan, we'll want this optimizer to update the decoder network's weights, too
            var_list += self._get_vars_by_scope(self.SCOPE_DECODER)
        optimizer = tf.train.AdamOptimizer(learning_rate=self._lr_enc).minimize(loss, var_list=var_list)

        tf.summary.scalar("mean_kl", mean_kl) 
        tf.summary.scalar("mean_recon", mean_recon) 
        tf.summary.scalar("enc_loss", loss) 
        
        return loss, optimizer, mean_recon, mean_kl

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

    def _log_model_step_results(self, enc_loss, kl, recon, ll_loss, dis_loss, dec_loss, elapsed_time):
        """
        Helper function for logging results from _model_step func
        
        Note that there's probably a better way to do this (perhaps with a class
        to represent all expected losses), but for now we use this function just to
        avoid writing the same log logic for train/dev/test output
        """
        logging.info("Enc Loss = {:.2f}, ".format(enc_loss) + 
                     "KL Divergence = {:.2f}, ".format(kl) +
                     "Reconstruction Loss = {:.2f}, ".format(recon) +
                     "ll_loss = {:.2f}, ".format(ll_loss) +
                     "dis_Loss = {:.2f}, ".format(dis_loss) +
                     "dec_Loss = {:.2f}, ".format(dec_loss) +
                     "Elapsed time: {:.2f} mins".format(elapsed_time))
        return

    def _save_model_step_results(self, epoch, enc_loss, kl, recon, ll_loss, dis_loss, dec_loss, elapsed_time):
        # save the epoch's data for review later
        self.metrics['epoch' + str(epoch)] = {
            'enc_loss': float(enc_loss),
            'kl_divergence': float(kl),
            'reconstruction_loss': float(recon),
            'll_loss': float(ll_loss),
            'dis_loss': float(dis_loss),
            'dec_loss': float(dec_loss),
            'elapsed_time': float(elapsed_time) 
        }
        return
    
    def _train_recon_example(self, epoch, viz_data):
        """
        Generates a side-by-side reconstruction example during the training process
        """
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

        return

    def sigmoid(self, x, shift, mult):
        """
        Using this sigmoid to discourage one network overpowering the other
        Credit: https://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN/blob/master/VAE-GAN-multi-gpu-celebA.ipynb
        """
        return 1 / (1 + math.exp(-(x + shift) * mult))
    
    def _save_model_ckpt(self, epoch):
        # Save the variables to disk.
        save_path = self.saver.save(self.sess, os.path.join(self.ckpt_dir, "model_epoch-{}.ckpt".format(epoch)))
        logging.info("Model saved in path: {}".format(save_path))
        metrics_json = os.path.join(self.ckpt_dir, "metrics.json")
        with open(metrics_json, 'w') as fp:
            json.dump(self.metrics, fp)
        logging.info("Metrics saved in path: {}".format(metrics_json))
        return
        
    def _model_step(self, feed_dict, step, summary_writer, summary_op, optim_ops=None, exec_ops=None, debug_ops=None):
        """
        Performs a single step of the model training process
        
        Writes summary_op result with summary_writer, prints debug_ops results, and returns
        exec_ops results
        
        Args:
            feed_dict: dict, model input tensorflow style
            step: int, id of current step
            summary_writer: tf summary writer for logging summary ops
            summary_op: tf.tensor, summary op
            optim_ops: list of tf.tensors, the tf optimizers
            exec_ops: list of tf.tensors, all tf tensors to execute and return
            debug_ops: list of tf.tensors, debug ops
            
        Returns:
            list, results of exec_ops
        """
        # build ops list; allow for Nones
        ops = [summary_op]
        ops.extend([] if not optim_ops else optim_ops)
        ops.extend([] if not exec_ops else exec_ops)
        ops.extend([] if not debug_ops else debug_ops)
        # execute
        results = self.sess.run(ops, feed_dict=feed_dict)
        # retrieve results for each set of ops
        summary = results[0]
        base = 1
        if optim_ops:
            optim_results = results[base:base + len(optim_ops)]
            base += len(optim_results)
        if exec_ops:
            exec_results = results[base:base + len(exec_ops)]
            base += len(exec_results)
        if debug_ops:
            debug_results = results[base:]
            # write debug ops
            self._log_debug_ops(debug_results)
        
        # write to summary
        summary_writer.add_summary(summary, step)

        # only return the execs (the loss values)
        return exec_results
    
    def train(self, train_generator, dev_generator=None, test_generator=None, epochs=10, input_repeats=1, display_step=1,
              save_step=1, viz_data=None, dev_step=10, enc_lr=0.0001, dec_lr=0.0001, dis_lr=0.0001):
        
        start = time.time()
        
        train_writer = tf.summary.FileWriter(os.path.join(self.tb_dir, 'train'), self.sess.graph)
        dev_writer = tf.summary.FileWriter(os.path.join(self.tb_dir, 'dev'), self.sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(self.tb_dir, 'test'), self.sess.graph)
        merge = tf.summary.merge_all()

        # https://riptutorial.com/tensorflow/example/13426/use-graph-finalize---to-catch-nodes-being-added-to-the-graph

        self.sess.graph.finalize()  # Graph is read-only after this statement.

        optim_ops = [self.enc_optim]
        exec_ops = [self.enc_loss, self.mean_kl, self.mean_recon]
        if not self.no_gan:
            optim_ops += [self.dec_optim, self.dis_optim]
            exec_ops = exec_ops + [self.ll_loss, self.dis_loss, self.dec_loss]
        debug_ops = [op for name, op, _ in self._debug_ops]
        
        ### Begin Training ###

        inputs = 0
        for epoch in range(epochs):

            logging.info("Epoch: {}, Elapsed Time: {:.2f}".format(epoch, elapsed_time(start)))
            batch_progress = 0

            ### Training Loop ###
            for batch_num, batch in enumerate(train_generator()):
                
                if self.verbose:
                    # keep track of how many inputs we have
                    if epoch == 0:
                        inputs += len(batch)
                    else:
                        batch_progress += len(batch)

                    logging.debug('Epoch: {} / {}, Batch: {} ({} / {}), Elapsed time: {:.2f} mins'.format(
                        epoch, epochs, batch_num, batch_progress, inputs, elapsed_time(start)))
                
                # adaptive learning rate
                #d_real = self.dis_input_output
                #d_fake = self.dis_dec_output
                if epoch == 0 and batch_num == 0:
                    d_real = 0
                    d_fake = 0
                enc_current_lr = enc_lr * self.sigmoid(np.mean(d_real), -.5, 15)
                dec_current_lr = dec_lr * self.sigmoid(np.mean(d_real), -.5, 15)
                dis_current_lr = dis_lr * self.sigmoid(np.mean(d_fake), -.5, 15)

                results = self._model_step(feed_dict={self._input_x: batch,
                                                      self._keep_prob:self.keep_prob,
                                                      self._trainable: True,
                                                      self._lr_enc: enc_current_lr,
                                                      self._lr_dec: dec_current_lr,
                                                      self._lr_dis: dis_current_lr,},
                                           step=self.step,
                                           summary_writer=train_writer,
                                           summary_op=merge,
                                           optim_ops=optim_ops,
                                           exec_ops=exec_ops + [self.dis_input_output, self.dis_dec_output],
                                           debug_ops=debug_ops)
                self.step += 1
                if self.no_gan:
                    enc_loss, kl, recon = results
                    # we know these didn't run; set loss values for reporting purposes
                    ll_loss = -999
                    dis_loss = -999
                    dec_loss = -999
                else:
                    enc_loss, kl, recon, ll_loss, dis_loss, dec_loss, d_real, d_fake = results

                if self.verbose:
                    self._log_model_step_results(enc_loss, kl, recon, ll_loss, dis_loss, dec_loss, elapsed_time(start))
                    
                logging.info('Memory Use (GB): {}'.format(memory()))

            self._save_model_step_results(epoch, enc_loss, kl, recon, ll_loss, dis_loss, dec_loss, elapsed_time(start))

            ### Save Model Checkpoint ###
            if (epoch + 1) % save_step == 0:
                self._save_model_ckpt(epoch)

            if (epoch + 1) % display_step == 0:
                self._log_model_step_results(enc_loss, kl, recon, dis_loss, dec_loss, elapsed_time(start))
                if viz_data is not None:
                    self._train_recon_example(epoch, viz_data)

            ### Evaluate Dev Dataset ###
            if dev_generator and (epoch + 1) % dev_step == 0:
                logging.info('Evaluating Dev')
                
                for batch_num, batch in enumerate(dev_generator()):
                    
                    #merge = tf.summary.merge_all()
                    results = self._model_step(
                       feed_dict={self._input_x: batch,
                                  self._keep_prob:self.keep_prob,
                                  self._trainable: True,
                                  self._lr_enc: enc_current_lr,
                                  self._lr_dec: dec_current_lr,
                                  self._lr_dis: dis_current_lr,},
                       step=self.step,
                       summary_writer=dev_writer,
                       summary_op=merge,
                       exec_ops=exec_ops)
                    if self.no_gan:
                        enc_loss, kl, recon = results
                        # we know these didn't run; set loss values for reporting purposes
                        ll_loss = -999
                        dis_loss = -999
                        dec_loss = -999
                    else:
                        enc_loss, kl, recon, ll_loss, dis_loss, dec_loss = results
                    self._log_model_step_results(enc_loss, kl, recon, ll_loss, dis_loss, dec_loss, elapsed_time(start))
                
        ### Evaluate Test Dataset ###
        if test_generator:
            logging.info('Evaluating Test')

            for batch_num, batch in enumerate(test_generator()):
                    
                    #merge = tf.summary.merge_all()
                    results = self._model_step(
                       feed_dict={self._input_x: batch,
                                  self._keep_prob:self.keep_prob,
                                  self._trainable: True,
                                  self._lr_enc: enc_current_lr,
                                  self._lr_dec: dec_current_lr,
                                  self._lr_dis: dis_current_lr,},
                       step=self.step,
                       summary_writer=test_writer,
                       summary_op=merge,
                       exec_ops=exec_ops)
                    if self.no_gan:
                        enc_loss, kl, recon = results
                        # we know these didn't run; set loss values for reporting purposes
                        ll_loss = -999
                        dis_loss = -999
                        dec_loss = -999
                    else:
                        enc_loss, kl, recon, ll_loss, dis_loss, dec_loss = results
                    self._log_model_step_results(enc_loss, kl, recon, ll_loss, dis_loss, dec_loss, elapsed_time(start))
            
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
    
    def _random_latent(self):
        return tf.random_normal(tf.stack([self._batch_size, self.latent_dim]))

    def random_latent_vector(self):
        """
        Create a latent vector for vaegan
        """
        latent_vector = self.sess.run(vaegan._random_latent())
        return latent_vector

    def latent_recon(self, latent_vector):
        """
        Decode an object from the provided latent vector
        """
        
        ops = tuple([self.decoder] + [op for name, op, _ in self._debug_ops])

        results = self.sess.run(self.decoder, 
            feed_dict={self.encoder: latent_vector, self._trainable: False})
        
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
