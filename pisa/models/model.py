import numpy as np
import tensorflow as tf

from pisa import PisaError
from pisa.logging import get_logger
from pisa.constants import SESSION_LEN
from pisa.models.core import embedding, normalize, \
    multi_head_attention_blocks


class Model:
    """
    Abstract model
    """
    # Supported optimizers.
    ADADELTA = 'Adadelta'
    SGD = 'SGD'
    ADAM = 'Adam'

    def __init__(self, sess, params, n_users, n_items, pretrained_embs):
        """
        Initialize a model
        :param sess: global session
        :param params: model parameters
        :param n_users: number of users
        :param n_items: number of items
        :param pretrained_embs: pretrained embedding
        """
        self.logger = get_logger()
        self.sess = sess
        model_params = params['model']['params']
        self.learning_rate = params.get('learning_rate', 0.001)
        self.embedding_dim = params.get('embedding_dim', 128)
        self.model_dir = params.get('model_dir', 'exp/model')
        self.model_name = params['model']['name']
        self.n_epochs = params.get('n_epochs', 20)
        self.seqlen = model_params.get('seqlen', 50)
        self.batch_size = params.get('batch_size', 256)
        self.n_users = n_users
        self.n_items = n_items
        self.optimizer = params['optimizer']
        self.checkpoint = None
        self.saver = None
        self.max_to_keep = params.get('max_to_keep', 1)
        self.dropout_rate = model_params.get('dropout_rate', 0.2)
        # loss
        self.loss = None
        # prediction
        self.test_logits = None
        self.item_ids = None
        self.item_embeddings = None
        self.initializers = pretrained_embs
        # parameters for Self-Attention Module
        self.input_scale = model_params.get('input_scale', False)
        self.num_blocks = model_params['sab'].get('num_blocks', 2)
        self.num_heads = model_params['sab'].get('num_heads', 1)
        self.causality = model_params['sab'].get('causality', True)
        self.use_reg = model_params.get('use_reg', True)
        self.l2_emb = model_params.get('l2_emb', 0.0)
        self.seq = None

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

    def build_feedict(self, batch, is_training=True):
        feedict = {
            self.is_training: is_training,
            self.seqin_ids: batch[0]
        }
        if is_training is True:
            feedict[self.pos_ids] = batch[-2]
            feedict[self.neg_ids] = batch[-1]
        return feedict

    def set_item_embeddings(self, item_embeddings):
        self.item_ids = np.array(list(item_embeddings.keys()))
        self.item_embeddings = np.array(list(item_embeddings.values()))

    def predict(self, feed_dict, top_n=50):
        return self.sess.run(self.test_logits, feed_dict)

    def _create_placeholders(self):
        """
        Build input graph
        :return:
        """
        self.logger.debug('--> Create input placeholders')
        with tf.name_scope('input_data'):
            # boolean to check if training, used for dropout
            self.is_training = tf.compat.v1.placeholder(
                name='is_training',
                dtype=tf.bool,
                shape=())
            # batch of history sequence of sessions
            self.seqin_ids = tf.compat.v1.placeholder(
                name='seq_input_ids', dtype=tf.int32,
                shape=[None, self.seqlen, SESSION_LEN])
            # batch of target sessions
            self.pos_ids = tf.compat.v1.placeholder(name='pos_ids',
                                                    dtype=tf.int32,
                                                    shape=[None, self.seqlen, SESSION_LEN])
            # batch of negative sessions, with negative items
            self.neg_ids = tf.compat.v1.placeholder(name='neg_ids',
                                                    dtype=tf.int32,
                                                    shape=[None, self.seqlen, SESSION_LEN])

    def _create_variables(self, reuse=None):
        """
        Build variables
        :return:
        """
        self.logger.debug('--> Create variables')
        initializer = self.initializers['item_embeddings'] \
            if self.initializers is not None else None
        # item embedding
        self.item_embedding_table, self.item_embeddings = \
            embedding(vocab_size=self.n_items,
                      embedding_dim=self.embedding_dim,
                      zero_pad=True,
                      use_reg=self.use_reg,
                      l2_reg=self.l2_emb,
                      scope='item_embedding_table',
                      initializer=initializer,
                      reuse=reuse)
        # positional embedding, shared for both K&V
        self.position_embedding_table = embedding(vocab_size=self.seqlen,
                                                  embedding_dim=self.embedding_dim,
                                                  zero_pad=False,
                                                  use_reg=self.use_reg,
                                                  l2_reg=self.l2_emb,
                                                  scope='position_embedding_table',
                                                  reuse=reuse)

    def _create_inference(self, name, reuse=None):
        """
        Build inference graph
        :return:
        """
        raise NotImplementedError('_create_inference method should be '
                                  'implemented in concrete model')

    def _create_loss(self):
        """
        Build loss graph
        :return:
        """
        raise NotImplementedError('_create_loss method should be '
                                  'implemented in concrete model')

    def _create_train_ops(self):
        """
        Train operations
        :return:
        """
        self.logger.debug('--> Define training operators')
        self.step = tf.compat.v1.Variable(0, trainable=False)
        optimizer = self._build_optimizer(self.learning_rate, self.step)
        self.train_ops = [optimizer.minimize(self.loss, global_step=self.step)]

    def _build_optimizer(self, lr, step):
        """ Builds an optimizer instance from internal parameter values.
        Default to AdamOptimizer if not specified.

        :returns: Optimizer instance from internal configuration.
        """
        self.logger.debug('----> Define optimizer')
        if self.optimizer == self.ADADELTA:
            return tf.compat.v1.train.AdadeltaOptimizer(lr)
        if self.optimizer == self.SGD:
            return tf.compat.v1.train.GradientDescentOptimizer(lr)
        elif self.optimizer == self.ADAM:
            return tf.compat.v1.train.AdamOptimizer(lr, beta2=0.98)
        else:
            raise PisaError(f'Unknown optimizer type {self.optimizer}')

    @classmethod
    def _activation(cls, act_type='None'):
        if act_type == 'none':
            return None
        elif act_type == 'relu':
            return tf.nn.relu
        elif act_type == 'leaky_relu':
            return tf.nn.leaky_relu
        else:
            raise PisaError(f'Not support activation of type {act_type}')

    @classmethod
    def _get_mask(cls, seq):
        return tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(
            seq, 0)), -1)
