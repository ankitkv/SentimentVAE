import tensorflow as tf


class SoftmaxWrapper(tf.nn.rnn_cell.RNNCell):

    """Operator adding a softmax projection to the given cell."""

    def __init__(self, cell, softmax_w, softmax_b, stddev=-1):
        """Create a cell with output projection."""
        self.cell = cell
        self.softmax_w = softmax_w
        self.softmax_b = softmax_b
        self.stddev = stddev

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.softmax_b.get_shape()[0]

    def __call__(self, inputs, state, scope=None):
        """Run the cell and output projection on inputs, starting from state."""
        output, res_state = self.cell(inputs, state)
        #output = tf.Print(output, [tf.sqrt(tf.nn.moments(output, [0, 1])[1])])
        # Default scope: "SoftmaxWrapper"
        with tf.variable_scope(scope or type(self).__name__):
            if self.stddev > 0.0:
                output += tf.random_normal(tf.shape(output), stddev=self.stddev)
            projected = tf.nn.bias_add(tf.matmul(output, tf.transpose(self.softmax_w),
                                                 name='softmax_transform'),
                                       self.softmax_b)
        return projected, res_state
