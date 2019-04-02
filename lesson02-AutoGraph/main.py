import tensorflow as tf

import timeit


cell = tf.keras.layers.LSTMCell(10)


@tf.function
def fn(input, state):
    """
    use static graph to compute LSTM
    :param input:
    :param state:
    :return:
    """

    return cell(input, state)


input_t = tf.zeros([10, 10])
state_t = [tf.zeros([10, 10])] * 2

# warmup
cell(input_t, state_t)
fn(input_t, state_t)


dynamic_graph_time = timeit.timeit(lambda: cell(input_t, state_t), number=100)
static_graph_time = timeit.timeit(lambda: fn(input_t, state_t), number=100)
print('dynamic_graph_time:', dynamic_graph_time)
print('static_graph_time:', static_graph_time)
