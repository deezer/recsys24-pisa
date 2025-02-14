from collections import defaultdict
import numpy as np
import tensorflow as tf

from pisa.constants import SESSION_LEN
from pisa.models.net import PISA
from pisa.models.core import embedding


class PISA_ART(PISA):
    # This model incorporate ACT-R framework to derive session embedding
    # Session embeddings is calculated through attention-based framework
    # where attention weights are ACT-R's track activations
    # PISA combines both BPR Loss (track level) & Regressions Loss (session level)
    # to learn the model's parameters
    # Integrate artist-level interactions into PISA
    def __init__(self, sess, params, n_users, n_items, pretrained_embs):
        super(PISA_ART, self).__init__(sess, params, n_users, n_items,
                                   pretrained_embs)
        model_params = params['model']['params']
        self.flatten_actr_art = model_params['actr'].get('flatten_actr_art', 1)
        self.lbda_art_loss = model_params.get('lbda_art_loss', 0.5)
        self.lbda_cross = params['model']['params'].get('lbda_cross', 0.)
        self.lbda_narts = params['model']['params'].get('lbda_narts', 0.)
        self.n_artists = params['n_artists']
        self.num_favs_art = model_params.get('num_favs_art', 5)
        self.lbda_task_art = model_params.get('lbda_task', 0.9)
        self.lbda_pos_art = model_params.get('lbda_pos', 0.5)
        self.lbda_ls_art = model_params.get('lbda_ls', 0.5)
        self.dropout_rate_art = 0

    def build_feedict(self, batch, is_training=True):
        feedict = super().build_feedict(batch, is_training)
        if is_training is True:
            feedict[self.art_seqin_ids] = batch[7]
            feedict[self.art_seqin_actr_bla] = batch[8]
            feedict[self.art_seqin_actr_spread] = batch[9]
            feedict[self.art_pos_actr_bla] = batch[10]
            feedict[self.art_pos_actr_spread] = batch[11]
            feedict[self.lt_art_ids] = batch[12]
            feedict[self.lt_art_blls] = batch[13]
            feedict[self.art_pos_ids] = batch[14]
            feedict[self.art_neg_ids] = batch[15]
            feedict[self.art_seqin_nitems] = batch[16]
            feedict[self.art_pos_nitems] = batch[17]
            feedict[self.art_neg_nitems] = batch[18]
        else:
            feedict[self.art_seqin_ids] = batch[5]
            feedict[self.art_seqin_actr_bla] = batch[6]
            feedict[self.art_seqin_actr_spread] = batch[7]
            feedict[self.lt_art_ids] = batch[8]
            feedict[self.lt_art_blls] = batch[9]
            feedict[self.art_seqin_nitems] = batch[10]
        return feedict

    def predict(self, feed_dict, top_n=50):
        # track
        item_ids = feed_dict['item_ids']
        pred_embeddings = self.track_lst_rep[:, -1, :]
        scores = -tf.matmul(
            pred_embeddings, self.item_embeddings, transpose_b=True)
        if self.lbda_ls > 0:
            long_embeddings = self.lt_rep[:, -1, :]
            long_scores = -tf.matmul(
                long_embeddings, self.item_embeddings, transpose_b=True)
            scores += self.lbda_ls * long_scores

        if self.lbda_cross > 0:
            cross_embeddings = self.art_lst_rep[:, -1, :]
            cross_scores = -tf.matmul(
                cross_embeddings, self.item_embeddings, transpose_b=True)
            scores += self.lbda_cross * cross_scores
        scores = self.sess.run(scores, feed_dict['model_feed'])
        reco_items = defaultdict(list)
        for i, (uid, u_scores) in enumerate(zip(feed_dict['user_ids'], scores)):
            topn_indices = np.argsort(scores[i])[:top_n]
            reco_items[uid].append([item_ids[idx] for idx in topn_indices])
        return reco_items

    def _create_placeholders(self):
        super(PISA_ART, self)._create_placeholders()
        # artist level
        self.art_seqin_ids = tf.compat.v1.placeholder(
            name='art_seq_input_ids', dtype=tf.int32,
            shape=[None, self.seqlen, SESSION_LEN])
        self.art_seqin_actr_bla = tf.compat.v1.placeholder(
            name='art_seq_in_actr_base_level', dtype=tf.float32,
            shape=[None, self.seqlen, SESSION_LEN])
        self.art_seqin_actr_spread = tf.compat.v1.placeholder(
            name='art_seq_in_actr_spreading', dtype=tf.float32,
            shape=[None, self.seqlen, SESSION_LEN])
        self.art_pos_actr_bla = tf.compat.v1.placeholder(
            name='art_pos_actr_base_level', dtype=tf.float32,
            shape=[None, self.seqlen, SESSION_LEN])
        self.art_pos_actr_spread = tf.compat.v1.placeholder(
            name='art_pos_actr_spreading', dtype=tf.float32,
            shape=[None, self.seqlen, SESSION_LEN])
        self.lt_art_ids = tf.compat.v1.placeholder(
            name='long_term_user_arts', dtype=tf.int32,
            shape=[None, self.num_favs_art])
        self.lt_art_blls = tf.compat.v1.placeholder(
            name='long_term_user_art_blls', dtype=tf.float32,
            shape=[None, self.num_favs_art])
        # batch of target sessions
        self.art_pos_ids = tf.compat.v1.placeholder(name='art_pos_ids',
                                                    dtype=tf.int32,
                                                    shape=[None, self.seqlen,
                                                           SESSION_LEN])
        # batch of negative sessions, with negative items
        self.art_neg_ids = tf.compat.v1.placeholder(name='art_neg_ids',
                                                    dtype=tf.int32,
                                                    shape=[None, self.seqlen,
                                                           SESSION_LEN])
        self.art_seqin_nitems = tf.compat.v1.placeholder(
            name='art_seq_input_nitems', dtype=tf.int32,
            shape=[None, self.seqlen])
        self.art_pos_nitems = tf.compat.v1.placeholder(
            name='art_pos_nitems', dtype=tf.int32,
            shape=[None, self.seqlen])
        self.art_neg_nitems = tf.compat.v1.placeholder(
            name='art_neg_nitems', dtype=tf.int32,
            shape=[None, self.seqlen])

    def _create_variables(self, reuse=None):
        super()._create_variables(reuse=reuse)
        art_initializer = self.initializers['artist_embeddings'] \
            if self.initializers is not None else None
        # artist embedding
        self.artist_embedding_table, self.artist_embeddings = \
            embedding(vocab_size=self.n_artists,
                      embedding_dim=self.embedding_dim,
                      zero_pad=True,
                      use_reg=self.use_reg,
                      l2_reg=self.l2_emb,
                      scope='artist_embedding_table',
                      initializer=None,
                      reuse=reuse)

        self.numart_embedding_table, self.numart_embeddings = \
            embedding(vocab_size=SESSION_LEN,
                      embedding_dim=self.embedding_dim,
                      zero_pad=True,
                      use_reg=self.use_reg,
                      l2_reg=self.l2_emb,
                      scope='numart_embedding_table',
                      initializer=None,
                      reuse=reuse)

    def _create_inference(self, name, reuse=None):
        """
        Build inference graph
        :return:
        """
        self.logger.debug('--> Create PISA-ART inference')
        (track_seqin_actr_weights, track_pos_actr_weights, track_prev_in_seq_ids,
        track_prev_pos_seq_ids, track_prev_neg_seq_ids) = self._actr_weights()
        (art_seqin_actr_weights, art_pos_actr_weights, art_prev_in_seq_ids,
        art_prev_pos_seq_ids, art_prev_neg_seq_ids) = self._actr_weights(
            item_type='artist')

        # track level
        self.input_seq, self.weighted_pos_seq, self.pos_seq, self.neg_seq, \
            input_seq_nelems, pos_seq_nelems = \
            self._input_output_seq(self.seqin_ids, self.pos_ids, self.neg_ids,
                                   track_prev_in_seq_ids,
                                   track_seqin_actr_weights,
                                   track_prev_pos_seq_ids,
                                   track_pos_actr_weights,
                                   track_prev_neg_seq_ids,
                                   flatten_actr=self.flatten_actr,
                                   item_type='track')

        # artist level
        self.art_input_seq, self.art_weighted_pos_seq, \
            self.art_pos_seq, self.art_neg_seq, \
            art_input_seq_nelems, art_pos_seq_nelems = \
            self._input_output_seq(self.art_seqin_ids, self.art_pos_ids,
                                   self.art_neg_ids, art_prev_in_seq_ids,
                                   art_seqin_actr_weights,
                                   art_prev_pos_seq_ids,
                                   art_pos_actr_weights,
                                   art_prev_neg_seq_ids,
                                   flatten_actr=self.flatten_actr_art,
                                   item_type='artist')
        # ignore padding items (0)
        self.istarget = tf.reshape(
            tf.compat.v1.to_float(tf.not_equal(pos_seq_nelems, 0)),
            [tf.shape(self.seqin_ids)[0] * self.seqlen])

        if self.input_scale is True:
            self.logger.info('Scale input sequence')
            self.scaled_input_seq = self.input_seq * (self.embedding_dim ** 0.5)
            self.art_scaled_input_seq = self.art_input_seq * (self.embedding_dim ** 0.5)
        else:
            self.logger.info('DO NOT scale input')
            self.scaled_input_seq = self.input_seq
            self.art_scaled_input_seq = self.art_input_seq
        # mask
        mask = self._get_mask(input_seq_nelems)

        # long-term & short-term user-artist representation
        self.art_lt_rep, self.art_st_rep = self._long_short_rep(
            input_seq=self.art_input_seq,
            scaled_input_seq=self.art_scaled_input_seq,
            mask=mask, lt_item_ids=self.lt_art_ids,
            lt_item_blls=self.lt_art_blls,
            item_embedding_table=self.artist_embedding_table,
            reuse=tf.compat.v1.AUTO_REUSE,
            dropout_rate=self.dropout_rate_art,
            item_type='artist')

        # short-term user-track representation
        # use shared long-term user representation from artist-level
        self.lt_rep, self.st_rep = self._long_short_rep(
            input_seq=self.input_seq,
            scaled_input_seq=self.scaled_input_seq,
            mask=mask, lt_item_ids=self.lt_item_ids,
            lt_item_blls=self.lt_item_blls,
            item_embedding_table=self.item_embedding_table,
            # lt_rep=self.art_lt_rep,
            reuse=tf.compat.v1.AUTO_REUSE,
            dropout_rate=self.dropout_rate,
            item_type='track')

        # fuse short-term representation on track & artist levels
        fused_art_track_st_rep = self._fusion(self.art_st_rep, self.st_rep,
                                              name='short_art_track_fusion')
        # long-short track fusion
        track_lst_rep, _ = self._long_short_fusion(
            self.lt_rep, fused_art_track_st_rep,
            name=f'track_long_short_term_fusion')
        # skip connection
        track_lst_rep = track_lst_rep + self.input_seq
        # renormalize
        self.track_lst_rep = track_lst_rep / (tf.expand_dims(tf.norm(
            track_lst_rep, ord=2, axis=-1), -1))

        # long-short artist fusion
        art_lst_rep, _ = self._long_short_fusion(
            self.art_lt_rep, fused_art_track_st_rep,
            name=f'artist_long_short_term_fusion')
        # skip connection
        art_lst_rep = art_lst_rep + self.art_input_seq
        # renormalize
        self.art_lst_rep = art_lst_rep / (tf.expand_dims(tf.norm(
            art_lst_rep, ord=2, axis=-1), -1))

        # represent sequence of number of items
        art_nitems_seq = self._art_nitems_seq_rep(mask, self.dropout_rate_art,
                                                  reuse)
        # art_nitems_seq = art_nitems_seq / (tf.expand_dims(tf.norm(
        #     art_nitems_seq, ord=2, axis=-1), -1))
        self.art_nitems_seq = art_nitems_seq + self.art_lst_rep + self.track_lst_rep
        # self.art_nitems_seq = tf.multiply(
        #     art_nitems_seq, tf.multiply(self.art_lst_rep, self.track_lst_rep))
        self.art_pos_nitems_emb = tf.nn.embedding_lookup(
            self.numart_embedding_table, self.art_pos_nitems)
        self.art_neg_nitems_emb = tf.nn.embedding_lookup(
            self.numart_embedding_table, self.art_neg_nitems)

        # normalize weighted outputs
        self.art_weighted_pos_seq = self.art_weighted_pos_seq / (
            tf.expand_dims(tf.norm(
                self.art_weighted_pos_seq, ord=2, axis=-1), -1))

    def _create_loss(self):
        """
        Build loss graph
        :return:
        """
        self.logger.debug('--> Create TRACK loss')
        track_loss = self._create_item_loss(st_rep=self.track_lst_rep,
                                            lt_rep=self.lt_rep,
                                            pos_ids=self.pos_ids,
                                            pos_seq=self.pos_seq,
                                            neg_seq=self.neg_seq,
                                            weighted_pos_seq=self.weighted_pos_seq,
                                            lbda_pos=self.lbda_pos,
                                            lbda_task=self.lbda_task,
                                            lbda_ls=self.lbda_ls)
        if self.lbda_cross > 0:
            cross_loss = self._create_item_loss(st_rep=self.art_lst_rep,
                                                lt_rep=self.art_lt_rep,
                                                pos_ids=self.pos_ids,
                                                pos_seq=self.pos_seq,
                                                neg_seq=self.neg_seq,
                                                weighted_pos_seq=self.weighted_pos_seq,
                                                lbda_pos=self.lbda_pos,
                                                lbda_task=self.lbda_task,
                                                lbda_ls=self.lbda_ls)
            track_loss = (1. - self.lbda_cross) * track_loss + \
                         self.lbda_cross * cross_loss

        if self.lbda_narts > 0:
            self.logger.debug('--> Create NUM ARTISTS loss')
            mask = tf.compat.v1.to_float(tf.not_equal(self.art_pos_nitems, 0))
            numart_loss = self._create_nitems_loss(self.art_nitems_seq,
                                                   self.art_pos_nitems_emb,
                                                   self.art_neg_nitems_emb, mask)
        else:
            numart_loss = 0.
        self.logger.debug('--> Create ARTIST loss')
        art_loss_1 = self._create_item_loss(st_rep=self.art_lst_rep,
                                            lt_rep=self.art_lt_rep,
                                            pos_ids=self.art_pos_ids,
                                            pos_seq=self.art_pos_seq,
                                            neg_seq=self.art_neg_seq,
                                            weighted_pos_seq=self.art_weighted_pos_seq,
                                            lbda_pos=self.lbda_pos_art,
                                            lbda_task=self.lbda_task_art,
                                            lbda_ls=self.lbda_ls_art)
        # artist representation should be predicted from
        # track representation as well
        art_loss_2 = self._create_item_loss(st_rep=self.st_rep,
                                            lt_rep=self.lt_rep,
                                            pos_ids=self.art_pos_ids,
                                            pos_seq=self.art_pos_seq,
                                            neg_seq=self.art_neg_seq,
                                            weighted_pos_seq=self.art_weighted_pos_seq,
                                            lbda_pos=self.lbda_pos_art,
                                            lbda_task=self.lbda_task_art,
                                            lbda_ls=self.lbda_ls_art)
        art_loss = (1. - self.lbda_narts) * (art_loss_1 + art_loss_2) + \
                    self.lbda_narts * numart_loss
        self.loss = (1 - self.lbda_art_loss) * track_loss + \
                    self.lbda_art_loss * art_loss

    def _create_item_loss(self, st_rep, lt_rep, pos_ids, pos_seq, neg_seq,
                          weighted_pos_seq, lbda_pos, lbda_task, lbda_ls):
        """
        Build loss graph
        :return:
        """
        mask = tf.compat.v1.to_float(tf.not_equal(pos_ids, 0))
        # The output of short-term & positive session representations should be
        # closed to positive items presented in positive sessions
        seq_bpr_loss = self._create_bpr_loss(st_rep, pos_seq,
                                             neg_seq, mask)
        pos_bpr_loss = self._create_bpr_loss(weighted_pos_seq,
                                             pos_seq, neg_seq, mask)
        bpr_loss = lbda_pos * pos_bpr_loss + (1. - lbda_pos) * seq_bpr_loss
        # The output of short-term should be closed to the positive sessions
        regr_loss = self._create_regression_loss(st_rep, weighted_pos_seq)
        # long-term
        if lbda_ls > 0:
            long_bpr_loss = self._create_bpr_loss(lt_rep, pos_seq,
                                                  neg_seq, mask)
            bpr_loss += lbda_ls * long_bpr_loss
        # combine item-level loss & session-level loss
        loss = lbda_task * bpr_loss + (1. - lbda_task) * regr_loss
        return loss

    @classmethod
    def _create_nitems_loss(cls, seq, pos_seq, neg_seq, mask):
        pos_score = tf.reduce_sum(seq * pos_seq, axis=-1)
        neg_score = tf.reduce_sum(seq * neg_seq, axis=-1)
        posneg_score = -tf.math.log(tf.nn.sigmoid(pos_score - neg_score))
        posneg_score = posneg_score * mask
        # avoid NaN loss in the case only BLL
        posneg_score = tf.where(tf.math.is_nan(posneg_score), 0., posneg_score)
        loss = tf.reduce_mean(posneg_score)
        return loss

    def _art_nitems_seq_rep(self, mask, dropout_rate, reuse=None):
        # input sequence
        numarts_seqin = tf.nn.embedding_lookup(self.numart_embedding_table,
                                               self.art_seqin_nitems)
        if self.input_scale is True:
            numarts_seqin = numarts_seqin * (self.embedding_dim ** 0.5)
        numarts_seq = self._seq_representation(numarts_seqin, mask=mask,
                                               dropout_rate=dropout_rate,
                                               reuse=reuse,
                                               name=f'numarts_seq_rep')
        return numarts_seq

    def _actr_weights(self, item_type='track'):
        if item_type == 'track':
            seqin_actr_bla = self.seqin_actr_bla
            seqin_actr_spread = self.seqin_actr_spread
            pos_actr_bla = self.pos_actr_bla
            pos_actr_spread = self.pos_actr_spread
            seqin_ids = self.seqin_ids
            pos_ids = self.pos_ids
            neg_ids = self.neg_ids
        else:
            seqin_actr_bla = self.art_seqin_actr_bla
            seqin_actr_spread = self.art_seqin_actr_spread
            pos_actr_bla = self.art_pos_actr_bla
            pos_actr_spread = self.art_pos_actr_spread
            seqin_ids = self.art_seqin_ids
            pos_ids = self.art_pos_ids
            neg_ids = self.art_neg_ids
        # ACT-R weights
        if self.spread_activate:
            self.logger.info(f'----> ACTR-SPREAD {item_type} Activate')
            seqin_actr_weights = tf.concat(
                [tf.expand_dims(seqin_actr_bla, axis=-1),
                 tf.expand_dims(seqin_actr_spread, axis=-1)],
                axis=-1)
            pos_actr_weights = tf.concat(
                [tf.expand_dims(pos_actr_bla, axis=-1),
                 tf.expand_dims(pos_actr_spread, axis=-1)],
                axis=-1)
        else:
            self.logger.info(f'----> ACTR-SPREAD {item_type} Off')
            seqin_actr_weights = tf.expand_dims(seqin_actr_bla, axis=-1)
            pos_actr_weights = tf.expand_dims(pos_actr_bla, axis=-1)
        if self.n_last_sess > 0:
            prev_in_seq_ids = None
            prev_pos_seq_ids = seqin_ids
            prev_neg_seq_ids = seqin_ids
        else:
            prev_in_seq_ids = seqin_ids
            prev_pos_seq_ids = pos_ids
            prev_neg_seq_ids = neg_ids

        return (seqin_actr_weights, pos_actr_weights, prev_in_seq_ids,
                prev_pos_seq_ids, prev_neg_seq_ids)

    def _input_output_seq(self, seqin_ids, pos_ids, neg_ids,
                          prev_in_seq_ids, seqin_actr_weights,
                          prev_pos_seq_ids, pos_actr_weights,
                          prev_neg_seq_ids, flatten_actr, item_type='track'):
        # input sequence
        input_seq, input_seq_nelems = \
            self._get_sess_representation(
                seqin_ids, prev_seq_ids=prev_in_seq_ids,
                flatten_actr=flatten_actr,
                seq_actr_weights=seqin_actr_weights,
                item_type=item_type)

        # positive output sequences
        weighted_pos_seq, pos_seq_nelems, pos_seq = \
            self._get_sess_representation(pos_ids,
                                          prev_seq_ids=prev_pos_seq_ids,
                                          seq_actr_weights=pos_actr_weights,
                                          flatten_actr=flatten_actr,
                                          output_item_emb=True,
                                          item_type=item_type)
        # we don't need negative session representation
        _, _, neg_seq = self._get_sess_representation(
            neg_ids, prev_seq_ids=prev_neg_seq_ids,
            output_item_emb=True, neg_seq=True,item_type=item_type)
        return (input_seq, weighted_pos_seq, pos_seq, neg_seq,
                input_seq_nelems, pos_seq_nelems)

    def _long_short_rep(self, input_seq, scaled_input_seq, mask,
                        lt_item_ids, lt_item_blls, item_embedding_table,
                        dropout_rate=0.,
                        item_type='track',
                        lt_rep=None,
                        reuse=None):
        """
        Long-term, short-term user representations
        """
        avoid_div_by_zero = tf.cast(mask < 0.5, tf.float32) * 1e-12

        # short-term user-track representation by Transformers
        st_rep = self._seq_representation(
            scaled_input_seq, mask=mask, reuse=reuse,
            name=f'{item_type}_user_short_term_rep')
        st_rep = st_rep / (tf.expand_dims(tf.norm(
            st_rep, ord=2, axis=-1), -1))

        # long-term user representation
        if lt_rep is None:
            lt_rep = self._long_term_user_representation(
                lt_item_ids=lt_item_ids, lt_item_blls=lt_item_blls,
                item_embedding_table=item_embedding_table,
                name=f'lt_{item_type}_user_reps')
            lt_rep = lt_rep / (tf.expand_dims(tf.norm(
                lt_rep, ord=2, axis=-1), -1))
        return lt_rep, st_rep

    @classmethod
    def _fusion(cls, art_rep, it_rep, name='art_track_fusion'):
        with tf.compat.v1.variable_scope(
                name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
            # multi_alpha = tf.concat([art_rep, it_rep], axis=-1)
            # multi_alpha = tf.compat.v1.layers.dense(multi_alpha, 2, name=name)
            # multi_alpha = tf.nn.softmax(multi_alpha, axis=-1)
            # multi_alpha_0 = tf.expand_dims(multi_alpha[:, :, 0], -1)
            # multi_alpha_1 = tf.expand_dims(multi_alpha[:, :, 1], -1)
            # fused_rep = multi_alpha_0 * art_rep + multi_alpha_1 * it_rep
            # test 1: addition
            fused_rep = art_rep + it_rep
            # test 2: Hadamard
            # fused_rep = tf.multiply(art_rep, it_rep)
        return fused_rep

    def _get_sess_representation(self, seq_ids, prev_seq_ids,
                                 flatten_actr=1., **kwargs):
        item_type = kwargs['item_type'] if 'item_type' in kwargs else 'track'
        item_embedding_table = self.item_embedding_table \
            if item_type == 'track' else self.artist_embedding_table
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
                    pow_seq_actr_weights = tf.math.pow(seq_actr_weights,
                                                       flatten_actr)
                    seq_actr_weights = pow_seq_actr_weights / \
                                       tf.reduce_sum(pow_seq_actr_weights,
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
