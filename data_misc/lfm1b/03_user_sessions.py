import numpy as np
import pandas as pd

from utils import *


params = load_configuration('configs/lfm1b.json')
dataset_params = params['dataset']
cache_path = os.path.join('cache', dataset_params['name'],
                          f'min{dataset_params["min_sessions"]}sess')
min_user_nstreams = dataset_params.get('min_user_nstreams', 1000)
min_item_nstreams = dataset_params.get('min_item_nstreams', 1500)
data_path = os.path.join(
    dataset_params['path'], dataset_params['name'],
    f'interactions_recent{dataset_params["ndays_recent"]}days_'
    f'histmin{dataset_params["ndays_min"]}days_'
    f'unstreams{min_user_nstreams}_instreams{min_item_nstreams}.csv')

MIN_SESSIONS_IN_HISTORY = 50

output_path = os.path.join(
    dataset_params['path'], dataset_params['name'],
    f'user_sessions_recent{dataset_params["ndays_recent"]}days_'
    f'histmin{dataset_params["ndays_min"]}days_'
    f'unstreams{min_user_nstreams}_instreams{min_item_nstreams}_'
    f'nsess{MIN_SESSIONS_IN_HISTORY}.csv')

logger = get_logger()

if not os.path.exists(output_path):
    logger.info(f'Load interactions data from {data_path}')
    data = pd.read_csv(data_path, names=["user_id", "art_id", "track_id", "ts"])

    logger.info(f'Organize data into sessions')
    # calculate delta t
    data['previous_ts'] = data.sort_values(['user_id', 'ts'],
                                           ascending=True).groupby('user_id')['ts'].shift()
    data['delta_t'] = data.ts - data.previous_ts
    data['is_nan'] = np.where(data['delta_t'].isna(), 1, 0)
    # new session if delta_t > 20m or beginning of a session
    data['is_new_session'] = np.where(data['delta_t'] > 60*20, 1, 0)
    data['is_new_session'] = data.is_new_session + data.is_nan
    data = data[["user_id", "art_id", "track_id", "ts", "is_new_session"]]
    # create session id
    data['session_id'] = data.sort_values(['user_id', 'ts'], ascending=True).groupby(
        'user_id')['is_new_session'].cumsum()
    # do not use repetition in a session
    data = data.sort_values(["user_id", "ts"], ascending=True).groupby(
        ["user_id", "art_id", "track_id", "session_id"]).first().reset_index()

    # get sessions with length > 10
    MIN_TRACKS_IN_SESSION = 10
    logger.info(f'Filter sessions less than {MIN_TRACKS_IN_SESSION}')
    data['count'] = data.groupby(['user_id', 'session_id'])['is_new_session'].transform('count')
    data = data.drop(columns=['is_new_session'])
    data = data[data['count'] >= MIN_TRACKS_IN_SESSION]

    # get only first 10 tracks in each session
    truncated_data = data.sort_values(['user_id', 'ts'], ascending=True).groupby(
        ['user_id', 'session_id']).head(MIN_TRACKS_IN_SESSION)

    # get only user more than 50 sessions
    user_sess_counts = truncated_data[['user_id', 'session_id']].drop_duplicates().groupby(
        'user_id').size().reset_index(name='counts')
    valid_users = user_sess_counts[user_sess_counts.counts >= MIN_SESSIONS_IN_HISTORY]
    valid_user_sessions = pd.merge(truncated_data, valid_users, on='user_id')
    user_sessions = valid_user_sessions[["user_id", "art_id", "track_id", "ts", "session_id"]]
    logger.info(f'Write user sessions data to {output_path}')
    user_sessions.to_csv(output_path, sep=',', header=False, index=False)
else:
    logger.info(f'Read user sessions data from {output_path}')
    user_sessions = pd.read_csv(output_path, names=dataset_params['col_names'])
logger.info(f'Number of interactions: {len(user_sessions)}')
logger.info(f'Number of users: {len(user_sessions["user_id"].unique())}')
logger.info(f'Number of tracks: {len(user_sessions["track_id"].unique())}')
logger.info(f'Number of artists: {len(user_sessions["art_id"].unique())}')
