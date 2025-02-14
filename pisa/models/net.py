from collections import defaultdict
import numpy as np
import tensorflow as tf

from pisa.constants import SESSION_LEN
from pisa.models.model import Model
from pisa.models.core import embedding, normalize, \
    multi_head_attention_blocks


class PISA(Model):
    # This model incorporate ACT-R framework to derive session embedding
    # Session embeddings is calculated through attention-based framework
    # where attention weights are ACT-R's track activations
    # PISA combines both BPR Loss (track level) & Regressions Loss (session level)
    # to learn the model's parameters
    def __init__(self, sess, params, n_users, n_items, pretrained_embs):
        super(PISA, self).__init__(sess, params, n_users, n_items,
                                   pretrained_embs)
        model_params = params['model']['params']
        # ACTR
        self.spread_activate = model_params['actr']['spread'].get('activate', True)
        self.pm_activate = model_params['actr']['pm'].get('activate', True)
        self.pm_embtype = model_params['actr']['pm'].get('emb', 'learn')
        self.flatten_actr = model_params['actr'].get('flatten_actr', 1)
        # alwasy true for BLL
        self.num_active_comp = sum([True, self.spread_activate, self.pm_activate])
        self.n_last_sess = model_params['actr']['spread'].get('n_last_sess', 0)

        self.lbda_task = model_params.get('lbda_task', 0.9)
        self.lbda_pos = model_params.get('lbda_pos', 0.)
        self.lbda_ls = model_params.get('lbda_ls', 0.)
        self.num_favs = model_params.get('num_favs', 0)

    def build_feedict(self, batch, is_training=True):
        feedict = super().build_feedict(batch, is_training)
        feedict[self.seqin_actr_bla] = batch[1]
        feedict[self.seqin_actr_spread] = batch[2]
        if is_training is True:
            feedict[self.pos_actr_bla] = batch[3]
            feedict[self.pos_actr_spread] = batch[4]
            if self.num_favs > 0:
                feedict[self.lt_item_ids] = batch[5]
                feedict[self.lt_item_blls] = batch[6]
                feedict[self.user_ids] = batch[-3]
        else:
            feedict[self.lt_item_ids] = batch[3]
            feedict[self.lt_item_blls] = batch[4]
            feedict[self.user_ids] = batch[-2]
        return feedict

    def predict(self, feed_dict, top_n=50):
        item_ids = feed_dict['item_ids']
        pred_embeddings = self.st_rep[:, -1, :]
        scores = -tf.matmul(
            pred_embeddings, self.item_embeddings, transpose_b=True)
        if self.lbda_ls > 0:
            long_embeddings = self.lt_rep[:, -1, :]
            long_scores = -tf.matmul(
                long_embeddings, self.item_embeddings, transpose_b=True)
            scores += self.lbda_ls * long_scores
        scores = self.sess.run(scores, feed_dict['model_feed'])
        reco_items = defaultdict(list)
        for i, (uid, u_scores) in enumerate(zip(feed_dict['user_ids'], scores)):
            topn_indices = np.argsort(scores[i])[:top_n]
            reco_items[uid].append([item_ids[idx] for idx in topn_indices])
        return reco_items

    def get_actr_weights(self):
        actr_weights = self.sess.run(self.actr_comp_weights)
        return actr_weights

    def _create_placeholders(self):
        super(PISA, self)._create_placeholders()
        self.seqin_actr_bla = tf.compat.v1.placeholder(
            name='seq_in_actr_base_level', dtype=tf.float32,
            shape=[None, self.seqlen, SESSION_LEN])
        self.seqin_actr_spread = tf.compat.v1.placeholder(
            name='seq_in_actr_spreading', dtype=tf.float32,
            shape=[None, self.seqlen, SESSION_LEN])
        self.pos_actr_bla = tf.compat.v1.placeholder(
            name='pos_actr_base_level', dtype=tf.float32,
            shape=[None, self.seqlen, SESSION_LEN])
        self.pos_actr_spread = tf.compat.v1.placeholder(
            name='pos_actr_spreading', dtype=tf.float32,
            shape=[None, self.seqlen, SESSION_LEN])
        self.lt_item_ids = tf.compat.v1.placeholder(
            name='long_term_user_items', dtype=tf.int32,
            shape=[None, self.num_favs])
        self.lt_item_blls = tf.compat.v1.placeholder(
            name='long_term_user_item_blls', dtype=tf.float32,
            shape=[None, self.num_favs])
        self.user_ids = tf.compat.v1.placeholder(
            name='user_ids', dtype=tf.int32,
            shape=(None,))

    def _create_variables(self, reuse=None):
        super()._create_variables(reuse=reuse)
        self.user_embedding_table = \
            embedding(vocab_size=self.n_users,
                      embedding_dim=self.embedding_dim,
                      zero_pad=False,
                      use_reg=self.use_reg,
                      l2_reg=self.l2_emb,
                      initializer='random_normal',
                      scope='user_embedding_table',
                      reuse=reuse)
        # ACT-R component weights
        if self.num_active_comp > 1:
            self.actr_comp_weights = tf.compat.v1.Variable(
                0.1 * tf.ones(self.num_active_comp), trainable=True,
                name=f'actr_comp_weights', dtype=tf.float32)
        if self.pm_activate and self.pm_embtype != 'learn':
            self.logger.info(f'----> Create Pretrained Embeddings Table for PM')
            initializer = self.initializers['item_embeddings'] \
                if self.initializers is not None else None
            self.pretrained_item_embedding_table, _ = \
                embedding(vocab_size=self.n_items,
                          embedding_dim=self.embedding_dim,
                          zero_pad=True,
                          scope='pretrained_item_embedding_table',
                          initializer=initializer,
                          reuse=reuse,
                          trainable=False)
        self.ls_weights = tf.compat.v1.get_variable(
            name=f'long_short_fused_weights',
            dtype=tf.float32,
            shape=2,
            initializer=tf.random_normal_initializer(0., stddev=1.))

    def _create_inference(self, name, reuse=None):
        """
        Build inference graph
        :return:
        """
        self.logger.debug('--> Create inference')
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            # ACT-R weights
            if self.spread_activate:
                self.logger.info('----> ACTR-SPREAD Activate')
                seqin_actr_weights = tf.concat([tf.expand_dims(self.seqin_actr_bla, axis=-1),
                                                tf.expand_dims(self.seqin_actr_spread, axis=-1)],
                                               axis=-1)
                pos_actr_weights = tf.concat([tf.expand_dims(self.pos_actr_bla, axis=-1),
                                              tf.expand_dims(self.pos_actr_spread, axis=-1)],
                                             axis=-1)
            else:
                self.logger.info('----> ACTR-SPREAD Off')
                seqin_actr_weights = tf.expand_dims(self.seqin_actr_bla, axis=-1)
                pos_actr_weights = tf.expand_dims(self.pos_actr_bla, axis=-1)
            if self.n_last_sess > 0:
                prev_in_seq_ids = None
                prev_pos_seq_ids = self.seqin_ids
                prev_neg_seq_ids = self.seqin_ids
            else:
                prev_in_seq_ids = self.seqin_ids
                prev_pos_seq_ids = self.pos_ids
                prev_neg_seq_ids = self.neg_ids

            # input sequence
            self.input_seq, input_seq_nelems = \
                self._get_sess_representation(
                    self.seqin_ids, prev_seq_ids=prev_in_seq_ids,
                    seq_actr_weights=seqin_actr_weights)

            # positive output sequences
            self.weighted_pos_seq, pos_seq_nelems, self.pos_seq = \
                self._get_sess_representation(self.pos_ids,
                                              prev_seq_ids=prev_pos_seq_ids,
                                              seq_actr_weights=pos_actr_weights,
                                              output_item_emb=True)
            # we don't need negative session representation
            _, _, self.neg_seq = self._get_sess_representation(
                self.neg_ids, prev_seq_ids=prev_neg_seq_ids,
                output_item_emb=True, neg_seq=True)

            # ignore padding items (0)
            self.istarget = tf.reshape(tf.compat.v1.to_float(tf.not_equal(pos_seq_nelems, 0)),
                                       [tf.shape(self.seqin_ids)[0] * self.seqlen])

            if self.input_scale is True:
                self.logger.info('Scale input sequence')
                self.scaled_input_seq = self.input_seq * (self.embedding_dim ** 0.5)
            else:
                self.logger.info('DO NOT scale input')
                self.scaled_input_seq = self.input_seq
            # mask
            mask = self._get_mask(input_seq_nelems)
            avoid_div_by_zero = tf.cast(mask < 0.5, tf.float32) * 0.0000001

            # short-term user representation by Transformers
            st_rep = self._seq_representation(self.scaled_input_seq, mask=mask,
                                              dropout_rate=self.dropout_rate,
                                              name='user_short_term_rep')
            st_rep = st_rep / (tf.expand_dims(tf.norm(
                st_rep + avoid_div_by_zero, ord=2, axis=-1), -1))

            # long-term user representation
            lt_rep = self._long_term_user_representation(
                lt_item_ids=self.lt_item_ids, lt_item_blls=self.lt_item_blls,
                item_embedding_table=self.item_embedding_table,
                name='lt_track_user_reps')
            self.lt_rep = lt_rep / (tf.expand_dims(tf.norm(
                lt_rep + avoid_div_by_zero, ord=2, axis=-1), -1))

            # long-short fusion
            fused_rep, self.lt_w = self._long_short_fusion(self.lt_rep, st_rep)
            # skip connection
            self.st_rep = fused_rep + self.input_seq
            # renormalize
            self.st_rep = self.st_rep / (tf.expand_dims(tf.norm(
                self.st_rep + avoid_div_by_zero, ord=2, axis=-1), -1))
            self.weighted_pos_seq = self.weighted_pos_seq / (
                tf.expand_dims(tf.norm(
                    self.weighted_pos_seq, ord=2, axis=-1), -1))

    def _create_loss(self):
        """
        Build loss graph
        :return:
        """
        self.logger.debug('--> Create loss')
        mask = tf.compat.v1.to_float(tf.not_equal(self.pos_ids, 0))
        # The output of short-term & positive session representations should be
        # closed to positive items presented in positive sessions
        seq_bpr_loss = self._create_bpr_loss(self.st_rep, self.pos_seq,
                                             self.neg_seq, mask)
        pos_bpr_loss = self._create_bpr_loss(self.weighted_pos_seq,
                                             self.pos_seq, self.neg_seq, mask)
        bpr_loss = self.lbda_pos * pos_bpr_loss + (1. - self.lbda_pos) * seq_bpr_loss
        # The output of short-term should be closed to the positive sessions
        regr_loss = self._create_regression_loss(self.st_rep,
                                                 self.weighted_pos_seq)
        # long-term
        if self.lbda_ls > 0:
            long_bpr_loss = self._create_bpr_loss(self.lt_rep,
                                                  self.pos_seq, self.neg_seq, mask)
            bpr_loss += self.lbda_ls * long_bpr_loss
        # combine song-level loss & session-level loss
        self.loss = self.lbda_task * bpr_loss + (1. - self.lbda_task) * regr_loss

    @classmethod
    def _create_bpr_loss(cls, seq, pos_seq, neg_seq, mask,
                         w_seq=None):
        pos_score = tf.squeeze(tf.matmul(tf.expand_dims(seq, axis=-2),
                                         pos_seq, transpose_b=True), axis=-2)
        neg_score = tf.squeeze(tf.matmul(tf.expand_dims(seq, axis=-2),
                                         neg_seq, transpose_b=True), axis=-2)
        posneg_score = -tf.math.log(tf.nn.sigmoid(pos_score - neg_score))
        posneg_score = posneg_score * mask
        # avoid NaN loss in the case only BLL
        posneg_score = tf.where(tf.math.is_nan(posneg_score), 0., posneg_score)
        if w_seq is not None:
            posneg_score = w_seq * posneg_score
        loss = tf.reduce_mean(posneg_score)
        return loss

    def _create_regression_loss(self, seq, weighted_pos_seq):
        sim = weighted_pos_seq * seq
        loss = tf.reduce_mean((1.0 - tf.reshape(
            tf.reduce_sum(sim, axis=-1),
            shape=[tf.shape(self.seqin_ids)[0] * self.seqlen])) * self.istarget
                                  ) / tf.reduce_sum(self.istarget)
        # avoid NaN loss in the case only BLL
        loss = tf.where(tf.math.is_nan(loss), 0., loss)
        return loss

    def _get_sess_representation(self, seq_ids, prev_seq_ids,
                                 **kwargs):
        item_type = kwargs['item_type'] if 'item_type' in kwargs else 'track'
        item_embedding_table = self.item_embedding_table
        seq_emb = tf.nn.embedding_lookup(item_embedding_table, seq_ids)
        if prev_seq_ids is None:
            first_seq_ids = seq_ids[:, 0:1, :]
            prev_seq_ids = seq_ids[:, 0:self.seqlen - 1, :]
            prev_seq_ids = tf.concat([first_seq_ids, prev_seq_ids], axis=1)
        prev_seq_emb = tf.nn.embedding_lookup(item_embedding_table, prev_seq_ids)
        if self.pm_activate and self.pm_embtype != 'learn':
            self.logger.info('----> PM Similarity from PRETRAINED')
            sim_seq_emb = tf.nn.embedding_lookup(self.pretrained_item_embedding_table,
                                                 seq_ids)
            prev_seq_emb = tf.nn.embedding_lookup(
                self.pretrained_item_embedding_table, prev_seq_ids)
        else:
            sim_seq_emb = seq_emb
            prev_seq_emb = prev_seq_emb
        seq_ids = tf.reshape(
            seq_ids,
            shape=[tf.shape(seq_emb)[0] * self.seqlen * SESSION_LEN])
        seq_n_elems = tf.compat.v1.to_float(tf.not_equal(seq_ids, 0))
        seq_n_elems = tf.reshape(
            seq_n_elems,
            shape=[tf.shape(seq_emb)[0], self.seqlen, SESSION_LEN])
        seq_n_elems = tf.reduce_sum(seq_n_elems, axis=-1)
        if 'neg_seq' not in kwargs:
            seq_actr_weights = kwargs['seq_actr_weights']
            # partial matching
            if self.pm_activate:
                self.logger.info('----> ACTR-PM Activate')
                sim = tf.matmul(sim_seq_emb, tf.transpose(prev_seq_emb, perm=[0, 1, 3, 2]))
                zeros = tf.zeros(shape=[tf.shape(seq_emb)[0], self.seqlen, SESSION_LEN],
                                 dtype=tf.float32)
                sim = tf.reduce_sum(tf.linalg.set_diag(sim, zeros), axis=-1)
                seq_actr_weights = tf.concat([seq_actr_weights, tf.expand_dims(sim, axis=-1)],
                                             axis=-1)
            else:
                self.logger.info('----> ACTR-PM Off')
            # feed forward network
            with tf.compat.v1.variable_scope(f'{item_type}_shared_FFN_ACT-R_weights',
                                             reuse=tf.compat.v1.AUTO_REUSE):
                self.logger.info(f'{item_type} POSITIVE+NORM FFN for ACT-R weights')
                if self.num_active_comp > 1:
                    # positive constraints for the weights of ACT-R components
                    actr_comp_weights = tf.nn.relu(self.actr_comp_weights) + 1e-8
                    actr_comp_weights = tf.expand_dims(tf.expand_dims(tf.expand_dims(
                        actr_comp_weights, axis=0), axis=0), axis=0)
                    actr_comp_weights = tf.tile(actr_comp_weights,
                                                [tf.shape(self.seqin_ids)[0], self.seqlen, SESSION_LEN, 1])

                    seq_actr_weights = tf.nn.relu(seq_actr_weights) + 1e-8
                    seq_actr_weights = tf.reduce_sum(seq_actr_weights * actr_comp_weights,
                                                     axis=-1)
                    # positive constraints for attention weights
                    seq_actr_weights = tf.nn.relu(seq_actr_weights) + 1e-8
                    # normalization
                    seq_actr_weights = tf.math.pow(seq_actr_weights, self.flatten_actr) / \
                                       tf.reduce_sum(tf.math.pow(seq_actr_weights, self.flatten_actr),
                                                     axis=-1, keepdims=True)
            if self.num_active_comp > 1:
                weighted_seq_emb = tf.reduce_sum(seq_emb * tf.expand_dims(
                    seq_actr_weights, axis=-1), axis=2)
            else:
                seq_actr_weights = tf.nn.relu(seq_actr_weights) + 1e-8
                weighted_seq_emb = tf.reduce_sum(seq_emb * seq_actr_weights,
                                                 axis=2)
        else:
            weighted_seq_emb = tf.reduce_mean(seq_emb, axis=2)
        output = weighted_seq_emb, seq_n_elems
        if 'output_item_emb' in kwargs and kwargs['output_item_emb'] is True:
            output = output + (seq_emb,)
        return output

    def _seq_representation(self, seq, mask, name, nonscale_inseq=None,
                            dropout_rate=0.,
                            reuse=None):
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            # learnable absolute position embedding
            self.abs_position = self._learnable_abs_position_embedding(
                self.position_embedding_table)
            seq = seq + self.abs_position
            seq *= mask
            # self-attention block
            seq = multi_head_attention_blocks(
                input_seq=seq,
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                embedding_dim=self.embedding_dim,
                dropout_rate=dropout_rate,
                mask=mask,
                reuse=reuse,
                causality=self.causality,
                is_training=self.is_training,
                nonscale_inseq=nonscale_inseq,
                name=name)
        return seq

    def _long_term_user_representation(self, lt_item_ids, lt_item_blls,
                                       item_embedding_table,
                                       name='long_term_user_representation'):
        with tf.compat.v1.variable_scope(
                name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
            lt_items_emb = tf.nn.embedding_lookup(item_embedding_table,
                                                  lt_item_ids)
            lt_user_rep = tf.reduce_sum(
                tf.expand_dims(lt_item_blls, axis=-1) * lt_items_emb,
                axis=1, keepdims=True)
            lt_user_rep = tf.tile(lt_user_rep, [1, self.seqlen, 1])
            return lt_user_rep

    def _learnable_abs_position_embedding(self, position_embedding_table):
        """
        Lookup embedding for positions
        :param position_embedding_table:
        :return:
        """
        position_ids = tf.tile(
            tf.expand_dims(tf.range(tf.shape(self.seqin_ids)[1]), 0),
            [tf.shape(self.seqin_ids)[0], 1])
        position = tf.nn.embedding_lookup(position_embedding_table,
                                          position_ids)
        return position

    @classmethod
    def _long_short_fusion(cls, lt_rep, st_rep, name='ls_fusion'):
        with tf.compat.v1.variable_scope(
                name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
            multi_alpha = tf.concat([lt_rep, st_rep], axis=-1)
            multi_alpha = tf.compat.v1.layers.dense(multi_alpha, 2, name=name)
            multi_alpha = tf.nn.softmax(multi_alpha, axis=-1)

            multi_alpha_0 = tf.expand_dims(multi_alpha[:, :, 0], -1)
            multi_alpha_1 = tf.expand_dims(multi_alpha[:, :, 1], -1)
            fused_rep = multi_alpha_0 * lt_rep + multi_alpha_1 * st_rep
        return fused_rep, multi_alpha_0
