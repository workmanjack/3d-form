import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os
%matplotlib inline

np.random.seed(0)
tf.set_random_seed(0)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


class StlVaeGan(object):
    
    def __init__(self):
        return

    def _init_discriminator(self):
        # Discriminator Net
        X = tf.placeholder(tf.float32, shape=[None, n_input]) # what's the number of features for stl, # of vertices?

        D_W1 = tf.Variable(xavier_init([n_input, 128])) # used 128 for MNIST, batch size or neurons?
        D_b1 = tf.Variable(tf.zeros(shape=[128]))

        D_W2 = tf.Variable(xavier_init([128, 1]))
        D_b2 = tf.Variable(tf.zeros(shape=[1]))

        theta_D = [D_W1, D_W2, D_b1, D_b2]

        return
    
    def _init_generator(self):
        # Generator Net
        Z = tf.placeholder(tf.float32, shape=[None, 100]) # inialized vector of 100 noises

        G_W1 = tf.Variable(xavier_init([100, 128]))
        G_b1 = tf.Variable(tf.zeros(shape=[128]))

        G_W2 = tf.Variable(xavier_init([128, n_input])) # which dimension shall i use for the 2nd? try different numbers?
        G_b2 = tf.Variable(tf.zeros(shape=[n_input]))

        theta_G = [G_W1, G_W2, G_b1, G_b2]
        
    
    def __repr__(self):
        return 'StlVaeGan()>'
    





def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


### need to normalize features or use different layer than sigmoid for the last step above?

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 10 # 128
Z_dim = 100


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('_output/'):
    os.makedirs('_output/')

i = 0
start = time.time()

for it in range(1000):
    if it % 200 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})  ### is 16 batch size here?
#         print(samples)
        x_reconstruct_vectors = samples.reshape([-1, 3, 3])
        plot_mesh(x_reconstruct_vectors, title='GAN')
        save_vectors_as_stl(x_reconstruct_vectors, 'first_gan_stl_'+str(it)+'.stl')
        
#     X_mb, _ = mnist.train.next_batch(mb_size)
    ### what's the right format to generate batch data
    for batch in thingi.batchmaker(batch_size=10, normalize=True, flat=True, pad_length=n_input, filenames=False):
        X_mb = batch
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

        if it % 200 == 0:
            print('Iter: {}'.format(it))
            print('Elapsed Time: {:.2f} mins'.format((time.time() - start) / 60))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()