import numpy as np
import tensorflow as tf

from pisa.models.core.net import feedforward, normalize


def sinusoidal_positional_embedding(dim, maxlen, dtype=tf.float32):
    """
    Sinusoidal positional embedding
    :param dim:
    :param maxlen:
    :param dtype:
    :return:
    """
    encoded_vec = np.array([pos/np.power(10000, 2*i/np.float32(dim))
                            for pos in range(maxlen)
                            for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([maxlen, dim]),
                                dtype=dtype)


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        with_qk=False):
    """Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      with_qk:
    Returns
      A 3d tensor with shape of (N, T_q, C)
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.compat.v1.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        K = tf.compat.v1.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        V = tf.compat.v1.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1),
                            [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.compat.v1.layers.dropout(outputs,
                                              rate=dropout_rate,
                                              training=tf.convert_to_tensor(is_training))
        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

    if with_qk:
        return Q, K
    else:
        return outputs


def multi_head_attention_blocks(input_seq, num_blocks, embedding_dim,
                                num_heads, dropout_rate, mask,
                                out_embedding_dim=-1,
                                causality=True,
                                reuse=None,
                                is_training=False,
                                nonscale_inseq=None,
                                name=''):
    """
    Multi head attention blocks
    :param name:
    :param input_seq:
    :param num_blocks:
    :param embedding_dim:
    :param out_embedding_dim:
    :param num_heads:
    :param dropout_rate:
    :param is_training:
    :param mask:
    :param causality:
    :param reuse:
    :param nonscale_inseq:
    :return:
    """
    seq = input_seq
    for i in range(num_blocks):
        with tf.compat.v1.variable_scope(f'{name}_num_blocks_{i}'):
            # Self-attention
            seq = multihead_attention(queries=normalize(seq),
                                      keys=seq,
                                      num_units=embedding_dim,
                                      num_heads=num_heads,
                                      dropout_rate=dropout_rate,
                                      is_training=is_training,
                                      causality=causality,
                                      reuse=reuse,
                                      scope=f'{name}_self_attention{i}')
            if i == num_blocks - 1 and out_embedding_dim > 0:
                num_units = [embedding_dim, out_embedding_dim]
            else:
                num_units = [embedding_dim, embedding_dim]
            # Feed forward net
            seq = feedforward(
                normalize(seq),
                num_units=num_units,
                dropout_rate=dropout_rate,
                is_training=is_training)
            seq *= mask
            if nonscale_inseq is not None:
                seq += nonscale_inseq
    return seq
