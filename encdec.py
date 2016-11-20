import tensorflow as tf

from config import cfg
import rnncell
import utils


class EncoderDecoderModel(object):

    '''The variational encoder-decoder model.'''

    def __init__(self, vocab, training, generator=False):
        self.vocab = vocab
        self.training = training
        self.global_step = tf.get_variable('global_step', shape=[],
                                           initializer=tf.zeros_initializer,
                                           trainable=False)
        self.summary_op = None
        self.summaries = []

        with tf.name_scope('input'):
            # left-aligned data:  <sos> w1 w2 ... w_T <eos> <pad...>
            self.data = tf.placeholder(tf.int32, [cfg.batch_size, None], name='data')
            # sentences with word dropout
            self.data_dropped = tf.placeholder(tf.int32, [cfg.batch_size, None],
                                               name='data_dropped')
            # sentence lengths
            self.lengths = tf.placeholder(tf.int32, [cfg.batch_size], name='lengths')

        embs = self.word_embeddings(self.data)
        embs_dropped = self.word_embeddings(self.data_dropped, reuse=True)
        with tf.name_scope('reverse-embeddings'):
            embs_reversed = tf.reverse_sequence(embs, self.lengths, 1)

        if generator:
            self.z = tf.placeholder(tf.float32, [cfg.batch_size, cfg.latent_size])
        else:
            self.z_mean, z_logvar = self.encoder(embs_reversed[:, 1:, :])
            with tf.name_scope('reparameterize'):
                eps = tf.random_normal([cfg.batch_size, cfg.latent_size])
                self.z = self.z_mean + tf.mul(tf.sqrt(tf.exp(z_logvar)), eps)
        output = self.decoder(embs_dropped, self.z)

        # shift left the input to get the targets

        with tf.name_scope('left-shift'):
            targets = tf.concat(1, [self.data[:, 1:], tf.zeros([cfg.batch_size, 1],
                                                               tf.int32)])
        with tf.name_scope('mle-cost'):
            self.nll = tf.reduce_sum(self.mle_loss(output, targets)) / cfg.batch_size
            self.perplexity = tf.exp(self.nll/tf.cast(tf.shape(self.data)[1], tf.float32))
            self.summaries.append(tf.scalar_summary('perplexity',
                                                    tf.reduce_mean(self.perplexity)))
        with tf.name_scope('kld-cost'):
            if generator:
                self.kld = 0.0
            else:
                self.kld = tf.reduce_sum(self.kld_loss(self.z_mean, z_logvar)) / \
                           cfg.batch_size
            self.kld_weight = tf.sigmoid((7 / cfg.anneal_bias)
                                         * (self.global_step - cfg.anneal_bias))
            self.summaries.append(tf.scalar_summary('weight_kld', self.kld_weight))
        with tf.name_scope('cost'):
            self.cost = self.nll + (self.kld_weight * self.kld)

        if training and not generator:
            self.train_op = self.train(self.cost)
        else:
            self.train_op = tf.no_op()


    def rnn_cell(self, num_layers):
        '''Return a multi-layer RNN cell.'''
        return tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(cfg.hidden_size)
                                            for _ in range(num_layers)])

    def word_embeddings(self, inputs, reuse=None):
        '''Look up word embeddings for the input indices.'''
        with tf.device('/cpu:0'), tf.variable_scope("Embeddings", reuse=reuse):
            self.embedding = tf.get_variable('word_embedding', [len(self.vocab.vocab),
                                                                cfg.word_emb_size],
                                     initializer=tf.random_uniform_initializer(-1.0, 1.0))
            embeds = tf.nn.embedding_lookup(self.embedding, inputs,
                                            name='word_embedding_lookup')
        return embeds

    def encoder(self, inputs):
        '''Encode sentence and return a latent representation.'''
        with tf.variable_scope("Encoder"):
            _, state = tf.nn.dynamic_rnn(self.rnn_cell(cfg.num_layers), inputs,
                                         sequence_length=self.lengths-1, swap_memory=True,
                                         dtype=tf.float32)
            z = utils.highway(state)
            z_mean = utils.linear(z, cfg.latent_size, True, scope='encoder_z_mean')
            z_logvar = utils.linear(z, cfg.latent_size, True, scope='encoder_z_logvar')
        return z_mean, z_logvar

    def decoder(self, inputs, z):
        '''Use the latent representation and word inputs to predict next words.'''
        with tf.variable_scope("Decoder"):
            z = utils.highway(z)
            z = utils.linear(z, cfg.latent_size, True, scope='decoder_latent')
            initial = []
            for i in range(cfg.num_layers):
                initial.append(tf.nn.tanh(utils.linear(z, cfg.hidden_size, True, 0.0,
                                                       scope='decoder_initial%d' % i)))
            self.decode_initial = tuple(initial)
            self.decode_cell = self.rnn_cell(cfg.num_layers)
            output, _ = tf.nn.dynamic_rnn(self.decode_cell, inputs,
                                          initial_state=self.decode_initial,
                                          sequence_length=self.lengths-1,
                                          swap_memory=True, dtype=tf.float32)
        return output

    def mle_loss(self, outputs, targets):
        '''Maximum likelihood estimation loss.'''
        mask = tf.cast(tf.greater(targets, 0, name='targets_mask'), tf.float32)
        output = tf.reshape(tf.concat(1, outputs), [-1, cfg.hidden_size])
        with tf.variable_scope("MLE_Softmax"):
            self.softmax_w = tf.get_variable("W", [len(self.vocab.vocab),cfg.hidden_size],
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.softmax_b = tf.get_variable("b", [len(self.vocab.vocab)],
                                             initializer=tf.zeros_initializer)
        if self.training and cfg.softmax_samples < len(self.vocab.vocab):
            targets = tf.reshape(targets, [-1, 1])
            mask = tf.reshape(mask, [-1])
            loss = tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, output,
                                              targets, cfg.softmax_samples,
                                              len(self.vocab.vocab))
            loss *= mask
        else:
            logits = tf.nn.bias_add(tf.matmul(output, tf.transpose(self.softmax_w),
                                              name='softmax_transform_mle'),
                                    self.softmax_b)
            loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                                          [tf.reshape(targets, [-1])],
                                                          [tf.reshape(mask, [-1])])
        loss = tf.reshape(loss, [cfg.batch_size, -1])
        self.summaries.append(tf.scalar_summary('cost_mle', tf.reduce_mean(loss)))
        return loss

    def kld_loss(self, z_mean, z_logvar):
        '''KL divergence loss.'''
        z_var = tf.exp(z_logvar)
        z_mean_sq = tf.square(z_mean)
        kld_loss = 0.5 * tf.reduce_sum(z_var + z_mean_sq - 1 - z_logvar, 1)
        self.summaries.append(tf.scalar_summary('cost_kld', tf.reduce_mean(kld_loss)))
        return kld_loss

    def train(self, cost):
        '''Generic training helper'''
        self.lr = tf.get_variable("lr", shape=[], initializer=tf.zeros_initializer,
                                  trainable=False)
        optimizer = utils.get_optimizer(self.lr, cfg.optimizer)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads = tf.gradients(cost, tvars)
        if cfg.max_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, cfg.max_grad_norm)
        return optimizer.apply_gradients(list(zip(grads, tvars)),
                                         global_step=self.global_step)

    def assign_lr(self, session, lr):
        '''Update the learning rate.'''
        session.run(tf.assign(self.lr, lr))

    def summary(self):
        if self.summary_op is None:
            self.summary_op = tf.merge_summary(self.summaries)
        return self.summary_op
