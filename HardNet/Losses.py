import sys
import tensorflow as tf
import numpy as np


def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    # Sum of squares, then adding dimension back
    d1_sq = tf.expand_dims(tf.math.reduce_sum(anchor * anchor, axis=1), axis=1)
    d2_sq = tf.expand_dims(tf.math.reduce_sum(positive * positive, axis=1), axis=1)

    eps = 1e-6

    # *_sq are single scalars, tile them to form 1x128
    d1_rep = tf.tile(d1_sq, (1, tf.shape(positive)[0]))
    d2_rep = tf.tile(d2_sq, (1, tf.shape(anchor)[0]))
    m_a_p = tf.matmul(anchor, positive, transpose_b=True)

    # Reshape is effectively transpose so 1x128 * 128x1
    return tf.math.sqrt((d1_rep + d2_rep - 2.0 * m_a_p) + eps)


def distance_vectors_pairwise(anchor, positive, negative=None):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    a_sq = tf.math.reduce_sum(anchor * anchor, axis=1)
    p_sq = tf.math.reduce_sum(positive * positive, axis=1)

    eps = 1e-8
    d_a_p = tf.math.sqrt(a_sq + p_sq - 2 * tf.math.reduce_sum(anchor * positive, axis=1) + eps)
    if negative is not None:
        n_sq = tf.math.reduce_sum(negative * negative, axis=1)
        d_a_n = tf.math.sqrt(a_sq + n_sq - 2 * tf.math.reduce_sum(anchor * negative, axis=1) + eps)
        d_p_n = tf.math.sqrt(p_sq + n_sq - 2 * tf.math.reduce_sum(positive * negative, axis=1) + eps)
        return d_a_p, d_a_n, d_p_n
    return d_a_p


def loss_random_sampling(anchor, positive, negative, anchor_swap=False, margin=1.0, loss_type="triplet_margin"):
    """Loss with random sampling (no hard in batch).
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    (pos, d_a_n, d_p_n) = distance_vectors_pairwise(anchor, positive, negative)
    if anchor_swap:
        min_neg = tf.math.reduce_min(d_a_n, d_p_n)
    else:
        min_neg = d_a_n

    if loss_type == "triplet_margin":
        loss = tf.nn.relu(margin + pos - min_neg)
    elif loss_type == 'softmax':
        exp_pos = tf.math.exp(2.0 - pos)
        exp_den = exp_pos + tf.math.exp(2.0 - min_neg) + eps
        loss = - tf.math.log(exp_pos / exp_den)
    elif loss_type == 'contrastive':
        loss = tf.nn.relu(margin - min_neg, min=0.0) + pos
    else:
        print('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = tf.math.reduce_mean(loss)
    return loss


def loss_L2Net(anchor, positive, anchor_swap=False, margin=1.0, loss_type="triplet_margin"):
    """L2Net losses: using whole batch as negatives, not only hardest.
    """

    anchor_shape = anchor.get_shape().as_list()
    positive_shape = positive.get_shape().as_list()
    assert anchor_shape == positive_shape, \
        "Input sizes between positive and negative must be equal."

    assert len(anchor_shape) == 2, "Inputd must be a 2D matrix."

    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive)

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = tf.linalg.diag_part(dist_matrix)

    if loss_type == 'softmax':
        exp_pos = tf.math.exp(2.0 - pos1)
        exp_den = tf.math.reduce_sum(tf.math.exp(2.0 - dist_matrix), 1) + eps
        loss = -tf.math.log(exp_pos / exp_den)
        if anchor_swap:
            exp_den1 = tf.math.reduce_sum(tf.math.exp(2.0 - dist_matrix), 0) + eps
            loss += -tf.math.log(exp_pos / exp_den1)
    else:
        print('Only softmax loss works with L2Net sampling')
        sys.exit(1)
    loss = tf.math.reduce_mean(loss)
    return loss


def loss_HardNet(anchor, positive, anchor_swap=False, anchor_ave=False,
                 margin=1.0, batch_reduce='min', loss_type="triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    anchor_shape = anchor.get_shape().as_list()
    positive_shape = positive.get_shape().as_list()
    assert anchor_shape == positive_shape, \
        "Input sizes between positive and negative must be equal."

    # NOTE: First dimension is batch. Will be like [?, 128]
    assert len(anchor_shape) == 2, "Inputd must be a 2D matrix."

    eps = 1e-8

    dist_matrix = distance_matrix_vector(anchor, positive)
    dist_matrix += eps

    eye = tf.dtypes.cast(tf.eye(tf.shape(dist_matrix)[1]), dtype=tf.float32)

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = tf.linalg.diag_part(dist_matrix)
    dist_without_min_on_diag = dist_matrix + eye * 10.0
    mask = -1.0 * (tf.to_float(tf.math.greater_equal(dist_without_min_on_diag, 0.008)) - 1.0)
    # mask = tf.dtypes.cast(mask, dtype=dist_without_min_on_diag.dtype) * 10
    mask = 10.0 * mask
    dist_without_min_on_diag = dist_without_min_on_diag + mask

    print("Batch reduce = {}, loss type = {}".format(batch_reduce, loss_type))

    if batch_reduce == 'min':
        # --- This is the one commonly used ---
        min_neg = tf.math.reduce_min(dist_without_min_on_diag, 1)

        if anchor_swap:
            min_neg2 = tf.math.reduce_min(dist_without_min_on_diag, 0)
            min_neg = tf.math.minimum(min_neg, min_neg2)

        # The following line does nothing...
        min_neg = min_neg
        pos = pos1

    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1, 1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1, 1)
        if anchor_swap:
            min_neg2 = dist_without_min_on_diag.T.contiguous().view(-1, 1)
            min_neg = tf.math.minimum(min_neg, min_neg2)

        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        n = tf.shape(anchor)[0]
        idxs = tf.Variable(tf.random.shuffle(np.arange(n)))
        min_neg = dist_without_min_on_diag.gather(1, idxs.view(-1, 1))
        if anchor_swap:
            min_neg2 = dist_without_min_on_diag.T.gather(1, idxs.view(-1, 1))
            min_neg = tf.math.reduce_min(min_neg, min_neg2)
        min_neg = min_neg.T.squeeze(0)
        pos = pos1
    else:
        print('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    if loss_type == "triplet_margin":
        loss = tf.nn.relu(margin + pos - min_neg)
    elif loss_type == 'softmax':
        exp_pos = tf.math.exp(2.0 - pos)
        exp_den = exp_pos + tf.math.exp(2.0 - min_neg) + eps
        loss = - tf.math.log(exp_pos / exp_den)
    elif loss_type == 'contrastive':
        loss = tf.nn.relu(margin - min_neg) + pos
    else:
        print('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)

    loss = tf.math.reduce_mean(loss)
    return loss


def global_orthogonal_regularization(anchor, negative):
    neg_dis = tf.math.reduce_sum(torch.mul(anchor, negative), 1)
    dim = anchor.size(1)
    gor = torch.pow(tf.math.reduce_mean(neg_dis), 2) + tf.nn.relu(torch.mean(torch.pow(neg_dis, 2)) - 1.0 / dim)

    return gor
