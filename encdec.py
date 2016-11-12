from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from config import cfg
import rnncell
import utils


class EncoderDecoderModel(object):

    '''The encoder-decoder model.'''

    def __init__(self, vocab, training):
        self.vocab = vocab
        self.training = training

        # left-aligned data:  <sos> w1 w2 ... w_T <eos> <pad...>
        self.data = tf.placeholder(tf.int32, [cfg.batch_size, None], name='data')
        # sentences with word dropout
        self.data_dropped = tf.placeholder(tf.int32, [cfg.batch_size, None],
                                            name='data_dropped')
        # sentence lengths
        self.lengths = tf.placeholder(tf.int32, [cfg.batch_size], name='lengths')

        embs = self.word_embeddings(self.data)
        embs_dropped = self.word_embeddings(self.data_dropped, reuse=True)
        self.latent = self.encoder(embs[:, 1:, :])

        output = self.decoder(embs_dropped, self.latent)

        # shift left the input to get the targets
        targets = tf.concat(1, [self.data[:, 1:], tf.zeros([cfg.batch_size, 1],
                                                            tf.int32)])
        mle_loss = self.mle_loss(output, targets)
        self.nll = tf.reduce_sum(mle_loss) / cfg.batch_size
        self.cost = self.nll
        if training:
            self.train_op = self.train(self.cost)
        else:
            self.train_op = tf.no_op()

    def rnn_cell(self, num_layers, latent=None):
        '''Return a multi-layer RNN cell.'''
        return tf.nn.rnn_cell.MultiRNNCell([rnncell.GRUCell(cfg.hidden_size,
                                                            latent=latent)
                                                for _ in range(num_layers)])

    def word_embeddings(self, inputs, reuse=None):
        '''Look up word embeddings for the input indices.'''
        with tf.device('/cpu:0'), tf.variable_scope("Embeddings", reuse=reuse):
            embedding = tf.get_variable('word_embedding', [len(self.vocab.vocab),
                                                           cfg.word_emb_size],
                                     initializer=tf.random_uniform_initializer(-1.0, 1.0))
            embeds = tf.nn.embedding_lookup(embedding, inputs,
                                            name='word_embedding_lookup')
        return embeds

    def encoder(self, inputs):
        '''Encode sentence and return a latent representation.'''
        with tf.variable_scope("Encoder"):
            _, state = tf.nn.dynamic_rnn(self.rnn_cell(cfg.num_layers), inputs,
                                         sequence_length=self.lengths-1, swap_memory=True,
                                         dtype=tf.float32)
            latent = utils.highway(state)
            latent = utils.linear(latent, cfg.latent_size, True, scope='encoder_latent')
        return latent

    def decoder(self, inputs, latent):
        '''Use the latent representation and word inputs to predict next words.'''
        with tf.variable_scope("Decoder"):
            output, _ = tf.nn.dynamic_rnn(self.rnn_cell(cfg.num_layers, latent), inputs,
                                          sequence_length=self.lengths-1,
                                          swap_memory=True, dtype=tf.float32)
        return output

    def mle_loss(self, outputs, targets):
        '''Maximum likelihood estimation loss.'''
        mask = tf.cast(tf.greater(targets, 0, name='targets_mask'), tf.float32)
        output = tf.reshape(tf.concat(1, outputs), [-1, cfg.hidden_size])
        with tf.variable_scope("MLE_Softmax"):
            softmax_w = tf.get_variable("W", [len(self.vocab.vocab), cfg.hidden_size],
                                       initializer=tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable("b", [len(self.vocab.vocab)],
                                        initializer=tf.zeros_initializer)
        if self.training and cfg.softmax_samples < len(self.vocab.vocab):
            targets = tf.reshape(targets, [-1, 1])
            mask = tf.reshape(mask, [-1])
            loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, output, targets,
                                              cfg.softmax_samples, len(self.vocab.vocab))
            loss *= mask
        else:
            logits = tf.nn.bias_add(tf.matmul(output, tf.transpose(softmax_w),
                                              name='softmax_transform_mle'), softmax_b)
            loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                                          [tf.reshape(targets, [-1])],
                                                          [tf.reshape(mask, [-1])])
        return tf.reshape(loss, [cfg.batch_size, -1])

    def train(self, cost):
        '''Generic training helper'''
        self.lr = tf.get_variable("lr", shape=[], initializer=tf.zeros_initializer,
                                  trainable=False)
        optimizer = utils.get_optimizer(self.lr, cfg.optimizer)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads = tf.gradients(cost, tvars)
        if cfg.max_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, cfg.max_grad_norm)
        return optimizer.apply_gradients(list(zip(grads, tvars)))

    def assign_lr(self, session, lr):
        '''Update the learning rate.'''
        session.run(tf.assign(self.lr, lr))
