import os
import pickle
import numpy as np
import scipy.sparse as sp
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from pisa.logging import get_logger
from pisa.constants import *
from pisa.data.datasets.actr_weights import load_actr_spread_weights, \
    load_actr_spread_weights_for_posout, load_actr_bll_weights


class Dataset:
    """
    Dataset
    """
    def __init__(self, params):
        self.command = params['command']
        cache_params = params['cache']
        self.dataset_params = params['dataset']
        self.model_name = params['training']['model']['name']
        self.min_sessions = self.dataset_params.get('min_sessions', 250)
        self.recent_hist = self.dataset_params.get('recent_history', -1)
        self.cache_path = os.path.join(cache_params['path'],
                                       self.dataset_params['name'],
                                       f'min{self.min_sessions}sess')
        self.embedding_dim = params['training'].get('embedding_dim', 128)
        self.samples_step = self.dataset_params.get('samples_step', 5)
        self.normalize_embedding = params['training'].get(
            'normalize_embedding', False)
        model_params = params['training']['model']['params']
        self.seqlen = model_params.get('seqlen', 20)

        # ACTR
        self.bll_type = model_params['actr']['bll'].get('type', 'ts')
        self.hop = model_params['actr']['spread'].get('hop', 1)
        self.n_last_sess = model_params['actr']['spread'].get('n_last_sess', 1)

        self.negsam_strategy = model_params.get('negsam_strategy', NEGSAM_UNIFORM)
        self.neg_alpha = model_params.get('neg_alpha', 1.0)

        # train/val/test data split
        data_split = self.dataset_params.get('train_val_test_split', '-1:10:10')
        n_train_sess, n_val_sess, n_test_sess = data_split.split(':')
        self.data_split = {'train': int(n_train_sess),
                           'valid': int(n_val_sess),
                           'test': int(n_test_sess)}
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.entities_path = os.path.join(
            self.cache_path, f'{self.dataset_params["name"]}_entities.npz')
        self.track_artist_map_path = os.path.join(
            self.cache_path, f'{self.dataset_params["name"]}_track_artist.pkl')
        self.logger = get_logger()

    def fetch_data(self):
        # load track embeddings
        track_embeddings, track_ids, art_ids, track_art_map = \
            self._load_track_embeddings()
        # load user sessions
        user_sessions = self._load_stream_sessions()
        user_ids = np.array(list(user_sessions.keys()))

        # mapping entities (user, track, artist) into internal ids
        track_ids_map = {tid: idx + 1 for idx, tid in enumerate(track_ids)}
        art_ids_map = {aid: idx + 1 for idx, aid in enumerate(art_ids)}
        user_ids_map = {uid: idx for idx, uid in enumerate(user_ids)}

        # train sessions indexes
        split_idx = -(self.data_split['valid'] + self.data_split['test'])
        train_session_indexes = self._load_train_session_indexes(
            user_sessions, split_idx=split_idx)

        train_user_tracks, valid_user_tracks, test_user_tracks = \
            self._load_split_user_tracks(user_sessions, data_split=self.data_split)
        # calculate track popularities
        pers_track_pops = self._load_pers_track_pops(user_sessions=user_sessions)
        glob_track_pops = self._load_glob_track_pops(
            user_tracks=[train_user_tracks, valid_user_tracks])
        norm_track_pops = None
        if self.negsam_strategy == NEGSAM_POP:
            norm_track_pops = self._normalize_track_popularities(glob_track_pops,
                                                                 track_ids,
                                                                 self.neg_alpha)
        # repeat consumptions dict
        repeat_consumptions = None
        if self.command == 'eval':
            repeat_consumptions = self._load_repeat_consumptions(
                user_tracks=test_user_tracks, pers_track_pops=pers_track_pops)

        # ACT-R BLL weights
        track_bll_weights = {
            'train': load_actr_bll_weights(self.cache_path, user_sessions,
                                           seqlen=self.seqlen,
                                           train_session_indexes=train_session_indexes,
                                           recent_hist=self.recent_hist,
                                           bll_type=self.bll_type,
                                           data_split=self.data_split, item_type='track',
                                           mode='train', samples_step=self.samples_step,
                                           logger=self.logger),
            'test': load_actr_bll_weights(self.cache_path, user_sessions,
                                          seqlen=self.seqlen,
                                          train_session_indexes=train_session_indexes,
                                          recent_hist=self.recent_hist,
                                          bll_type=self.bll_type,
                                          data_split=self.data_split, item_type='track',
                                          mode='test', samples_step=self.samples_step,
                                          logger=self.logger)
        }
        # ACT-R Spread weights
        track_spread_weights = load_actr_spread_weights(
            self.cache_path, user_sessions, train_session_indexes,
            recent_hist=self.recent_hist, data_split=self.data_split,
            item_type='track', track_ids=track_ids, art_ids=art_ids,
            logger=self.logger, seqlen=self.seqlen,
            item_embeddings=track_embeddings, hop=self.hop,
            samples_step=self.samples_step,
            n_last_sess=self.n_last_sess)
        track_pos_spread_weights = load_actr_spread_weights_for_posout(
            self.cache_path, user_sessions, train_session_indexes,
            recent_hist=self.recent_hist, seqlen=self.seqlen,
            item_type='track', track_ids=track_ids, art_ids=art_ids,
            logger=self.logger, item_embeddings=track_embeddings,
            hop=self.hop, samples_step=self.samples_step,
            n_last_sess=self.n_last_sess)

        self.data = {
            'user_sessions': user_sessions,
            'track_embeddings': track_embeddings,
            'user_ids': user_ids,
            'track_ids': track_ids,
            'art_ids': art_ids,
            'user_ids_map': user_ids_map,
            'track_ids_map': track_ids_map,
            'art_ids_map': art_ids_map,
            'track_art': track_art_map,
            'train_session_indexes': train_session_indexes,
            'data_split': self.data_split,
            'n_users': len(user_sessions),
            'n_items': len(track_ids),
            'n_artists': len(art_ids),
            'glob_track_popularities': glob_track_pops,
            'pers_track_popularities': pers_track_pops,
            'norm_track_popularities': norm_track_pops,
            'user_tracks': {
                'train': train_user_tracks,
                'valid': valid_user_tracks,
                'test': test_user_tracks
            },
            'samples_step': self.samples_step,
            'recent_hist': self.recent_hist,
            'repeat_consumptions': repeat_consumptions,
            'track_spread_weights': track_spread_weights,
            'track_pos_spread_weights': track_pos_spread_weights,
            'track_bll_weights': track_bll_weights
        }

    def _load_track_embeddings(self):
        raise NotImplementedError('_load_track_embeddings should be '
                                  'implemented in concrete class')

    def _load_stream_sessions(self):
        raise NotImplementedError('_load_stream_sessions should be '
                                  'implemented in concrete class')

    def _load_train_session_indexes(self, user_sessions, split_idx):
        train_session_indexes_path = os.path.join(
            self.cache_path,
            f'train_session_indexes_recenthist{self.recent_hist}_'
            f'samples-step{self.samples_step}_seqlen{self.seqlen}_'
            f'{self.data_split["valid"]}v_{self.data_split["test"]}t.pkl')
        if not os.path.exists(train_session_indexes_path):
            self.logger.info('Extract training session indexes')
            train_session_indexes = []
            for uid, sessions in user_sessions.items():
                last_idx = len(sessions) + (split_idx - 2)
                first_idx = last_idx - self.recent_hist if 0 < self.recent_hist < last_idx else 0
                # train_session_indexes.append((uid, last_idx))
                if self.samples_step > 0:
                    offsets = list(range(last_idx, first_idx + self.seqlen - 1,
                                         -self.samples_step))
                    for offset in offsets:
                        train_session_indexes.append((uid, offset))
            pickle.dump(train_session_indexes,
                        open(train_session_indexes_path, 'wb'))
        else:
            self.logger.info(f'Load training session indexes from '
                             f'{train_session_indexes_path}')
            train_session_indexes = pickle.load(open(train_session_indexes_path, 'rb'))
        return train_session_indexes

    def _load_split_user_tracks(self, user_sessions, data_split):
        train_user_tracks_path = os.path.join(
            self.cache_path,
            f'train_user_tracks_recenthist{self.recent_hist}_'
            f'{data_split["valid"]}v_{data_split["test"]}t.pkl')
        valid_user_tracks_path = os.path.join(
            self.cache_path,
            f'valid_user_tracks_{data_split["valid"]}v_{data_split["test"]}t.pkl')
        test_user_tracks_path = os.path.join(
            self.cache_path,
            f'test_user_tracks_{data_split["valid"]}v_{data_split["test"]}t.pkl')
        if not os.path.exists(train_user_tracks_path) or not \
                os.path.exists(valid_user_tracks_path) or not \
                os.path.exists(test_user_tracks_path):
            self.logger.info('Extract user tracks...')
            train_user_tracks = defaultdict(set)
            valid_user_tracks = test_user_tracks = defaultdict(dict)
            split_idx = data_split['valid'] + data_split['test']
            for uid, sessions in user_sessions.items():
                start_idx = -split_idx - self.recent_hist if self.recent_hist > 0 else 0
                for s in sessions[start_idx:-split_idx]:
                    for tid in s['track_ids']:
                        train_user_tracks[uid].add(tid)
                for s in sessions[-split_idx:-data_split['test']]:
                    valid_user_tracks[uid][s['session_id']] = set(s['track_ids'])
                for s in sessions[-data_split['test']:]:
                    test_user_tracks[uid][s['session_id']] = set(s['track_ids'])
            pickle.dump(train_user_tracks, open(train_user_tracks_path, 'wb'))
            pickle.dump(valid_user_tracks, open(valid_user_tracks_path, 'wb'))
            pickle.dump(test_user_tracks, open(test_user_tracks_path, 'wb'))
        else:
            self.logger.info('Load user tracks...')
            train_user_tracks = pickle.load(open(train_user_tracks_path, 'rb'))
            valid_user_tracks = pickle.load(open(valid_user_tracks_path, 'rb'))
            test_user_tracks = pickle.load(open(test_user_tracks_path, 'rb'))
        return train_user_tracks, valid_user_tracks, test_user_tracks

    def _load_glob_track_pops(self, user_tracks):
        glob_track_pops_path = os.path.join(
            self.cache_path,
            f'glob_track_popularities_recenthist{self.recent_hist}_'
            f'{self.data_split["valid"]}v_{self.data_split["test"]}t.pkl')
        if not os.path.exists(glob_track_pops_path):
            self.logger.info('Calculate global track popularities...')
            train_user_tracks, valid_user_tracks = user_tracks
            glob_track_pops = defaultdict(float)
            for uid, track_ids in train_user_tracks.items():
                for tid in track_ids:
                    glob_track_pops[tid] += 1.0
            for uid, track_dict in valid_user_tracks.items():
                for _, track_ids in track_dict.items():
                    for tid in track_ids:
                        glob_track_pops[tid] += 1.0
            n_users = len(train_user_tracks)
            glob_track_pops = {tid: pop / n_users for tid, pop in glob_track_pops.items()}
            pickle.dump(glob_track_pops, open(glob_track_pops_path, 'wb'))
        else:
            self.logger.info(f'Load global track popularities from {glob_track_pops_path}...')
            glob_track_pops = pickle.load(open(glob_track_pops_path, 'rb'))
        return glob_track_pops

    def _load_pers_track_pops(self, user_sessions):
        pers_track_pops_path = os.path.join(
            self.cache_path,
            f'pers_track_popularities_recenthist{self.recent_hist}_'
            f'{self.data_split["valid"]}v_{self.data_split["test"]}t.pkl')
        if not os.path.exists(pers_track_pops_path):
            pers_track_pops = defaultdict(dict)
            thres_num_sess = self.data_split['valid'] + \
                             self.data_split['test'] + self.recent_hist
            for uid, sessions in tqdm(user_sessions.items(),
                                      desc='Calculate personal track popularities...'):
                start_idx = -thres_num_sess if self.recent_hist > 0 and \
                                               len(sessions) > thres_num_sess else 0
                for ss in sessions[start_idx:-self.data_split['test']]:
                    for tid in ss['track_ids']:
                        if tid not in pers_track_pops[uid]:
                            pers_track_pops[uid][tid] = 1.0
                        else:
                            pers_track_pops[uid][tid] += 1.0
            for uid, popdict in pers_track_pops.items():
                sum_pop = sum(list(popdict.values()))
                pers_track_pops[uid] = defaultdict(
                    float, {tid: pop / sum_pop for tid, pop in popdict.items()})
            pickle.dump(pers_track_pops, open(pers_track_pops_path, 'wb'))
        else:
            self.logger.info(f'Load personal track popularities from {pers_track_pops_path}...')
            pers_track_pops = pickle.load(open(pers_track_pops_path, 'rb'))
        return pers_track_pops

    def _normalize_track_popularities(self, glob_track_pops, track_ids,
                                      alpha=1.0):
        norm_track_pops_path = os.path.join(
            self.cache_path,
            f'norm_track_popularities_recenthist{self.recent_hist}_'
            f'{self.data_split["valid"]}v_{self.data_split["test"]}t_alpha{alpha}.npy')
        if not os.path.exists(norm_track_pops_path):
            self.logger.info('Normalize global track popularities')
            if alpha != 1.0:
                glob_track_pops = {tid: np.power(freq, alpha)
                                   for tid, freq in glob_track_pops.items()}
            total_count = np.sum(list(glob_track_pops.values()))
            item_popularities = np.zeros(len(track_ids), dtype=np.float32)
            for idx in range(len(track_ids)):
                tid = track_ids[idx]
                if tid in glob_track_pops:
                    item_popularities[idx] = glob_track_pops[tid] / total_count
            with open(norm_track_pops_path, 'wb') as f:
                np.save(f, item_popularities)
        else:
            self.logger.info(f'Load normalized track from {norm_track_pops_path}...')
            with open(norm_track_pops_path, 'rb') as f:
                item_popularities = np.load(f)
        return item_popularities

    def _load_repeat_consumptions(self, user_tracks, pers_track_pops):
        repeat_consumption_path = os.path.join(
            self.cache_path,
            f'repeat_consumptions_recenthist{self.recent_hist}_'
            f'{self.data_split["valid"]}v_{self.data_split["test"]}t.pkl')
        if not os.path.exists(repeat_consumption_path):
            self.logger.info('Extract repeat consumptions...')
            repeat_consumptions = defaultdict(list)
            for uid, track_dict in user_tracks.items():
                repeat_tracks = []
                for sid, track_ids in track_dict.items():
                    for tid in track_ids:
                        if tid in pers_track_pops[uid]:
                            repeat_tracks.append(tid)
                    repeat_consumptions[uid].append(set(repeat_tracks))
            pickle.dump(repeat_consumptions, open(repeat_consumption_path, 'wb'))
        else:
            self.logger.info(f'Load repeat consumptions from {repeat_consumption_path}')
            repeat_consumptions = pickle.load(open(repeat_consumption_path, 'rb'))
        return repeat_consumptions
