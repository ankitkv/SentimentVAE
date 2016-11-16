import tensorflow as tf
import numpy as np
from tf_beam_decoder import BeamDecoder

sess = tf.InteractiveSession()

class MarkovChainCell(tf.nn.rnn_cell.RNNCell):
    """
    This cell type is only used for testing the beam decoder.

    It represents a Markov chain characterized by a probability table p(x_t|x_{t-1},x_{t-2}).
    """
    def __init__(self, table):
        """
        table[a,b,c] = p(x_t=c|x_{t-1}=b,x_{t-2}=a)
        """
        assert len(table.shape) == 3 and table.shape[0] == table.shape[1] == table.shape[2]
        self.log_table = tf.log(np.asarray(table, dtype=np.float32))
        self._output_size = table.shape[0]

    def __call__(self, inputs, state, scope=None):
        """
        inputs: [batch_size, 1] int tensor
        state: [batch_size, 1] int tensor
        """
        logits = tf.reshape(self.log_table, [-1, self.output_size])
        indices = state[0] * self.output_size + inputs
        return tf.gather(logits, tf.reshape(indices, [-1])), (inputs,)

    @property
    def state_size(self):
        return (1,)

    @property
    def output_size(self):
        return self._output_size

# Test 1

table = np.array([[[0.0, 0.6, 0.4],
                   [0.0, 0.4, 0.6],
                   [0.0, 0.0, 1.0]]] * 3)

cell = MarkovChainCell(table)
initial_state = cell.zero_state(1, tf.int32)
initial_input = initial_state[0]

beam_decoder = BeamDecoder(num_classes=3, stop_token=2, beam_size=7, max_len=5)

_, final_state = tf.nn.seq2seq.rnn_decoder(
                        [beam_decoder.wrap_input(initial_input)] + [None] * 4,
                        beam_decoder.wrap_state(initial_state),
                        beam_decoder.wrap_cell(cell),
                        loop_function = lambda prev_symbol, i: tf.reshape(prev_symbol, [-1, 1])
                    )

best_dense = beam_decoder.unwrap_output_dense(final_state)
best_sparse = beam_decoder.unwrap_output_sparse(final_state)
best_logprobs = beam_decoder.unwrap_output_logprobs(final_state)

assert all(best_sparse.eval().values == [2])
assert np.isclose(np.exp(best_logprobs.eval())[0], 0.4)

# Test 2

table = np.array([[[0.9, 0.1, 0],
                   [0, 0.9, 0.1],
                   [0, 0, 1.0]]] * 3)
cell = MarkovChainCell(table)
initial_state = cell.zero_state(1, tf.int32)
initial_input = initial_state[0]

beam_decoder = BeamDecoder(num_classes=3, stop_token=2, beam_size=10, max_len=3)

_, final_state = tf.nn.seq2seq.rnn_decoder(
                        [beam_decoder.wrap_input(initial_input)] + [None] * 2,
                        beam_decoder.wrap_state(initial_state),
                        beam_decoder.wrap_cell(cell),
                        loop_function = lambda prev_symbol, i: tf.reshape(prev_symbol, [-1, 1])
                    )

candidates, candidate_logprobs = sess.run((final_state[2], final_state[3]))

assert all(candidates[0,:] == [0,0,0])
assert np.isclose(np.exp(candidate_logprobs[0]), 0.9 * 0.9 * 0.9)
# Note that these three candidates all have the same score, and the sort order
# may change in the future
assert all(candidates[1,:] == [0,0,1])
assert np.isclose(np.exp(candidate_logprobs[1]), 0.9 * 0.9 * 0.1)
assert all(candidates[2,:] == [0,1,1])
assert np.isclose(np.exp(candidate_logprobs[2]), 0.9 * 0.1 * 0.9)
assert all(candidates[3,:] == [1,1,1])
assert np.isclose(np.exp(candidate_logprobs[3]), 0.1 * 0.9 * 0.9)
assert all(np.isclose(np.exp(candidate_logprobs[4:]), 0.0))
