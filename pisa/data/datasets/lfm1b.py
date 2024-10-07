import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

from pisa.data.datasets.dataset import Dataset
from pisa.utils.tempo import high_level_tempo_from_ts


class LFMDataset(Dataset):

    def __init__(self, params):
        super(LFMDataset, self).__init__(params)
        self.ndays_recent = self.dataset_params.get('ndays_recent', 365)
        self.ndays_min = self.dataset_params.get('ndays_min', 365)
        self.ustreams = self.dataset_params.get('min_user_nstreams', 1000)
        self.istreams = self.dataset_params.get('min_item_nstreams', 1500)

    def _load_track_embeddings(self):
        track_embs_prefix = self.dataset_params['files']['track_embeddings']
        track_embeddings_path = os.path.join(
            self.dataset_params['path'], self.dataset_params['name'],
            f'{track_embs_prefix}_recent{self.ndays_recent}days_'
            f'histmin{self.ndays_min}days_unstreams{self.ustreams}_'
            f'instreams{self.istreams}_nsess{self.min_sessions}.pkl')
        self.logger.info(f'Load track embeddings from {track_embeddings_path}')
        track_embeddings = pickle.load(open(track_embeddings_path, 'rb'))
        if not os.path.exists(self.entities_path):
            data_path = os.path.join(
                self.dataset_params['path'], self.dataset_params['name'],
                f'user_sessions_recent{self.ndays_recent}days_'
                f'histmin{self.ndays_min}days_unstreams{self.ustreams}_'
                f'instreams{self.istreams}_nsess{self.min_sessions}.csv')
            data = pd.read_csv(data_path, names=self.dataset_params["col_names"])
            data = data[['track_id', 'art_id']].drop_duplicates()
            art_ids = data['art_id'].tolist()
            track_art_map = dict(zip(data['track_id'].tolist(), art_ids))
            art_ids = list(set(art_ids))
            track_ids = list(track_embeddings.keys())
            pickle.dump(track_art_map, open(self.track_artist_map_path, 'wb'))
            np.savez(self.entities_path, track_ids=track_ids, art_ids=art_ids)
        else:
            entities = np.load(self.entities_path, allow_pickle=True)
            track_ids = entities['track_ids']
            art_ids = entities['art_ids']
            track_art_map = pickle.load(open(self.track_artist_map_path, 'rb'))
        return track_embeddings, track_ids, art_ids, track_art_map

    def _load_stream_sessions(self):
        output_path = os.path.join(self.cache_path, 'user_sessions.pkl')
        if not os.path.exists(output_path):
            data_path = os.path.join(
                self.dataset_params['path'], self.dataset_params['name'],
                f'user_sessions_recent{self.ndays_recent}days_'
                f'histmin{self.ndays_min}days_unstreams{self.ustreams}_'
                f'instreams{self.istreams}_nsess{self.min_sessions}.csv')
            streams_df = pd.read_csv(data_path, names=self.dataset_params["col_names"])
            grouped_streams = streams_df.sort_values(['ts']).groupby(
                ['user_id', 'session_id'])
            user_sessions = defaultdict(list)
            user_sess_index_map = defaultdict(dict)

            # get context infos
            self.logger.info('Get context infos')
            first_rows_grouped_streams = grouped_streams.first().reset_index()
            first_rows_grouped_streams = first_rows_grouped_streams[
                ['user_id', 'session_id', 'ts']]
            first_rows_grouped_streams = first_rows_grouped_streams.sort_values(
                ['ts']).groupby('user_id')

            for user_id, df_group in first_rows_grouped_streams:
                session_ids = df_group['session_id'].tolist()
                user_sess_index_map[user_id] = {sid: idx for idx, sid in enumerate(session_ids)}
                timestamps = df_group['ts'].tolist()
                for idx, (sid, ts) in enumerate(zip(session_ids, timestamps)):
                    if idx == 0:
                        time_since_last_session = 0
                    else:
                        time_since_last_session = ts - timestamps[idx - 1]
                    day_of_week, hour_of_day = high_level_tempo_from_ts(ts)
                    user_sessions[user_id].append({
                        'session_id': sid,
                        'context': {
                            'time_since_last_session': time_since_last_session,
                            'ts': ts,
                            'day_of_week': day_of_week,
                            'hour_of_day': hour_of_day
                        }
                    })
            # tracks infos
            self.logger.info('Get track list in each session')
            for group_name, df_group in grouped_streams:
                user_id, session_id = group_name
                track_ids = df_group['track_id'].astype('int32').tolist()
                idx = user_sess_index_map[user_id][session_id]
                user_sessions[user_id][idx]['track_ids'] = track_ids
            # write result to cache
            self.logger.info(f'Write user session streams to {output_path}')
            pickle.dump(user_sessions, open(output_path, 'wb'))
        else:
            self.logger.info(f'Load user session streams from {output_path}')
            user_sessions = pickle.load(open(output_path, 'rb'))
        return user_sessions
