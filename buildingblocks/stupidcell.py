import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class StupidRNNCell(tf.nn.rnn_cell.BasicRNNCell):

    def __init__(self,
                 num_units = 1,
                 activation= None,
                 inner_activation= tf.sigmoid,
                 reuse=None,
                 name=None,
                 dtype=None):
        super(StupidRNNCell, self).__init__(num_units ,activation, reuse, name, dtype)
        self._inner_activation = inner_activation
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"% inputs_shape)

        input_depth = inputs_shape[1].value
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth -1, 2])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[2],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):

        rnn_coeff = math_ops.matmul(inputs[:,1:], self._kernel)
        rnn_coeff = nn_ops.bias_add(rnn_coeff, self._bias)
        rnn_coeff = self._inner_activation(rnn_coeff)

        gate_inputs = tf.multiply( array_ops.concat([inputs[:,:1], state], 1) , rnn_coeff)
        print(gate_inputs.shape)
        output = tf.reduce_sum( gate_inputs, axis=1, keepdims=True)

        return output, output

class StupidRNNCell2(tf.nn.rnn_cell.BasicRNNCell):

    def __init__(self,
                 num_units = 1,
                 activation= None,
                 inner_activation= tf.sigmoid,
                 reuse=None,
                 name=None,
                 dtype=None):
        super(StupidRNNCell, self).__init__(num_units ,activation, reuse, name, dtype)
        self._inner_activation = inner_activation
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"% inputs_shape)

        input_depth = inputs_shape[1].value
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth -1, 2])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[2],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self._hidden_kernel = self.add_variable(
            'hidden'+_WEIGHTS_VARIABLE_NAME,
            shape=[input_depth -1, input_depth -1])
        self._hidden_bias = self.add_variable(
            'hidden'+_BIAS_VARIABLE_NAME,
            shape=[input_depth -1],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):

        rnn_coeff = math_ops.matmul(inputs[:,1:], self._hidden_kernel)
        rnn_coeff = nn_ops.bias_add(rnn_coeff, self._hidden_bias)
        rnn_coeff = self._inner_activation(rnn_coeff)
        rnn_coeff = math_ops.matmul(rnn_coeff, self._kernel)
        rnn_coeff = nn_ops.bias_add(rnn_coeff, self._bias)
        rnn_coeff = self._inner_activation(rnn_coeff)

        gate_inputs = tf.multiply( array_ops.concat([inputs[:,:1], state], 1) , rnn_coeff)
        print(gate_inputs.shape)
        output = tf.reduce_sum( gate_inputs, axis=1, keepdims=True)

        return output, output
class StupidLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def call(self, inputs, state):
        """Long short-term memory cell (LSTM)."""
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
          c, h = state
        else:
          c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))
        LSTMStateTuple=tf.contrib.rnn.LSTMStateTuple
        if self._state_is_tuple:
          new_state = LSTMStateTuple(new_c, new_h)
        else:
          new_state = array_ops.concat([new_c, new_h], 1)
        return new_c, new_state
