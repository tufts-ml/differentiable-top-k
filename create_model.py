from functools import partial

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Multiply, Layer
from tensorflow.keras.models import Model
from perturbations import perturbed


def tf_topk(x, k=1):
    # Get the indices of the top-k elements
    values, indices = tf.nn.top_k(x, k=k)

    # Create a one-hot vector with the shape of `x`
    one_hot = tf.one_hot(indices, tf.shape(x)[-1], on_value=1.0, off_value=0.0, dtype=tf.float32)

    # Sum along the last axis to get a one-hot vector with shape (batch_size, num_classes)
    one_hot = tf.reduce_max(one_hot, axis=-2)

    return one_hot




def create_model(num_vals=9, hidden_sizes=[72,36,18],
                 hidden_activation='relu', use_topk=True,
                 num_perturbation_samples=1000, sigma_perturb=0.1, l2_reg=0.001):

    # values + RGB
    num_dims = num_vals + num_vals*3

    input_shape = (num_dims,)
    scorer_inputs = Input(shape=input_shape)
    # just colors
    x = scorer_inputs[:,num_vals:]

    for size in hidden_sizes:
        this_layer = Dense(size, activation=hidden_activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        x = this_layer(x)

    final_mlp_layer = Dense(num_vals, activation='sigmoid')
    x=final_mlp_layer(x)

    if use_topk:
        this_topk = partial(tf_topk, k=2)
        pert_topk = perturbed(this_topk,
                                        num_samples=num_perturbation_samples,
                                        sigma=sigma_perturb,
                                        noise='normal',
                                        batched=True)
        class TopKLayer(tf.keras.layers.Layer):
            def __init__(self):
                super(TopKLayer, self).__init__()
            def call(self, x):
                return pert_topk(x)  # you don't need to explicitly define the custom gradient

        topk_layer = TopKLayer()

        x = topk_layer(x)

    # this MLP is the scorer:
    scorer = Model(inputs=scorer_inputs, outputs=x)

    # Now create a new model that uses the scorer

    inputs = Input(shape=input_shape)
    multiplication_layer = Multiply()
    scores = scorer(inputs)
    vals = inputs[:,:num_vals]
    scores_times_vals = multiplication_layer([scores, vals])
    summed = tf.reduce_sum(scores_times_vals, axis=1)

    model = Model(inputs=inputs, outputs=summed)

    return model



