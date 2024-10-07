import sys
import tensorflow as tf
sys.path.append(".")

from .logging import get_logger


class Triplet:
    """
    Abstract model
    """
    # Supported optimizers.
    ADADELTA = 'Adadelta'
    SGD = 'SGD'
    ADAM = 'Adam'

    def __init__(self, sess, params, n_users, n_items):
        self.logger = get_logger()
        self.sess = sess
        self.learning_rate = params.get('learning_rate', 0.001)
        self.embedding_dim = params.get('embedding_dim', 128)
        self.model_dir = params.get('model_dir', 'exp/model')
        self.clip_norm = params['model']['params'].get('clip_norm', 1.0)
        self.margin = params['model']['params'].get('margin', 1.0)
        self.n_negatives = params['model']['params'].get('n_negatives', 1)
        self.l2_reg = params['model']['params'].get('l2_reg', False)
        self.max_to_keep = params.get('max_to_keep', 1)
        self.optimizer = params['optimizer']
        self.n_users = n_users
        self.n_items = n_items
        self.checkpoint = None
        self.saver = None
        self.norm_item_embeddings = None

    def build_feedict(self, batch, is_training=True):
        feedict = {
            self.anc_ids: batch[0]
        }
        if is_training is True:
            feedict[self.pos_ids] = batch[1]
            feedict[self.neg_ids] = batch[2]
        return feedict

    def build_graph(self, name=None):
        """
        Build model computation graph
        :return:
        """
        self._create_placeholders()
        self._create_variables(reuse=tf.compat.v1.AUTO_REUSE)
        self._create_inference(name, reuse=tf.compat.v1.AUTO_REUSE)
        self._create_loss()
        self._create_train_ops()
        self.saver = tf.compat.v1.train.Saver(
            max_to_keep=self.max_to_keep)

    def save(self, save_path, global_step):
        """
        Save the model to directory
        :param save_path:
        :param global_step:
        :return:
        """
        self.saver.save(self.sess, save_path=save_path,
                        global_step=global_step)

    def restore(self, name=None):
        """
        Restore the model if it already exists
        :return:
        """
        self.checkpoint = tf.compat.v1.train.get_checkpoint_state(
            self.model_dir)
        if self.checkpoint is not None:
            self.logger.info(f'Load {self.__class__} model '
                             f'from {self.model_dir}')
            self.build_graph(name=name)
            self.saver.restore(self.sess,
                               self.checkpoint.model_checkpoint_path)

    def get_embeddings(self):
        if self.l2_reg is False:
            self.logger.info('Unnormalized Item Embeddings')
            embeddings = self.sess.run(self.item_embeddings)
        else:
            self.logger.info('NORM Item Embeddings')
            embeddings = self.sess.run(self.norm_item_embeddings)
        return embeddings

    def _create_placeholders(self):
        """
        Build input graph
        :return:
        """
        self.logger.debug('--> Create triplet placeholders')
        with tf.name_scope('input_data'):
            # batch of anchor user ids
            self.anc_ids = tf.compat.v1.placeholder(name='anchor_ids',
                                                    dtype=tf.int32,
                                                    shape=[None])
            # batch of positive track ids (tracks listened by anchor users)
            self.pos_ids = tf.compat.v1.placeholder(name='pos_ids',
                                                    dtype=tf.int32,
                                                    shape=[None])
            # batch of negative item ids (tracks not listened by anchor users)
            self.neg_ids = tf.compat.v1.placeholder(name='neg_ids',
                                                    dtype=tf.int32,
                                                    shape=[None, self.n_negatives])

    def _create_variables(self, reuse=None):
        """
        Build variables
        :return:
        """
        self.logger.info('--> Create User/Item embeddings')
        with tf.compat.v1.variable_scope('user_item_embeddings',
                                         reuse=reuse):
            # user embeddings
            self.user_embeddings = tf.compat.v1.get_variable(
                name='user_embedding_matrix',
                shape=[self.n_users, self.embedding_dim],
                initializer=tf.random_normal_initializer(
                    0., stddev=1. / (self.embedding_dim ** 0.5)),
                dtype=tf.float32
            )
            # item embeddings
            self.item_embeddings = tf.compat.v1.get_variable(
                name='item_embedding_matrix',
                shape=[self.n_items, self.embedding_dim],
                initializer=tf.random_normal_initializer(
                    0., stddev=1. / (self.embedding_dim ** 0.5)),
                dtype=tf.float32
            )
            if self.l2_reg is True:
                self.logger.info('--> L2 NORM activate')
                self.norm_item_embeddings = tf.math.l2_normalize(self.item_embeddings,
                                                                 axis=1)

    def _create_inference(self, name, reuse=None):
        """
        Build inference graph
        :return:
        """
        self.logger.debug('--> Get user/pos_items/neg_items vectors '
                          'for a batch')
        with tf.name_scope('inference'):
            # anchor user vectors [batch_size, dim]
            self.anchors = tf.nn.embedding_lookup(self.user_embeddings,
                                                  self.anc_ids,
                                                  name='batch_anc_vectors')
            # positive item vectors [batch_size, dim]
            self.positives = tf.nn.embedding_lookup(self.item_embeddings,
                                                    self.pos_ids,
                                                    name='batch_positive_vectors')
            # negative item vectors [batch_size, dim, n_negs]
            negatives = tf.nn.embedding_lookup(self.item_embeddings, self.neg_ids)
            self.negatives = tf.transpose(negatives, (0, 2, 1),
                                          name='batch_negative_embeddings')

    def _create_loss(self):
        self.logger.debug('--> Define Triplet loss')
        # positive distances
        self.pos_distances = self._pos_distances()

        # negative distances
        self.neg_distances = self._neg_distances()

        # get only the closest negative distance to the anchor
        min_neg_distances = tf.reduce_min(self.neg_distances, axis=1,
                                          name='min_neg_distances')
        loss = tf.maximum(self.pos_distances - min_neg_distances + self.margin,
                          0.0, name="pair_loss")
        self.loss = tf.reduce_sum(loss, name='embedding_loss')

    def _create_train_ops(self):
        """
        Train operations
        :return:
        """
        self.logger.debug('--> Define training operators')
        optimizer = self._build_optimizer(self.learning_rate)
        ops = [optimizer.minimize(self.loss)]
        with tf.control_dependencies(ops):
            self.train_ops = ops + self._clip_by_norm_op()

    def _clip_by_norm_op(self):
        """
        Clip operation by norm
        :return:
        """
        self.logger.debug('----> Define clip norm operators (regularization)')
        return [
            tf.compat.v1.assign(self.user_embeddings,
                                tf.clip_by_norm(self.user_embeddings,
                                                self.clip_norm,
                                                axes=[1])),
            tf.compat.v1.assign(self.item_embeddings,
                                tf.clip_by_norm(self.item_embeddings,
                                                self.clip_norm,
                                                axes=[1]))]

    def _build_optimizer(self, lr):
        """ Builds an optimizer instance from internal parameter values.
        Default to AdamOptimizer if not specified.

        :returns: Optimizer instance from internal configuration.
        """
        self.logger.debug('----> Define optimizer')
        if self.optimizer == self.ADADELTA:
            return tf.compat.v1.train.AdadeltaOptimizer()
        if self.optimizer == self.SGD:
            return tf.compat.v1.train.GradientDescentOptimizer(lr)
        elif self.optimizer == self.ADAM:
            return tf.compat.v1.train.AdamOptimizer(lr)
        else:
            raise ValueError(f'Unknown optimizer type {self.optimizer}')

    def _pos_distances(self):
        self.logger.debug('--> Define Triplet positive distances')
        distances = tf.reduce_sum(
            tf.compat.v1.squared_difference(self.anchors, self.positives),
            axis=1,
            name='pos_distances')
        return tf.maximum(distances, 0.0)

    def _neg_distances(self):
        self.logger.debug('--> Define Triplet negative distances')
        expanded_anchors = tf.expand_dims(self.anchors, -1,
                                          name='expanded_anchors')
        distances = tf.reduce_sum(
            tf.compat.v1.squared_difference(expanded_anchors, self.negatives),
            axis=1,
            name='neg_distances')
        return tf.maximum(distances, 0.0)
