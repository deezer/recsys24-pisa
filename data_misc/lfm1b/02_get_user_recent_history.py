import pickle
import pandas as pd
import gc

from utils import *


# read configuration file
params = load_configuration('configs/lfm1b.json')
dataset_params = params['dataset']
cache_path = os.path.join('cache', dataset_params['name'],
                          f'min{dataset_params["min_sessions"]}sess')
min_user_nstreams = dataset_params.get('min_user_nstreams', 1000)
min_item_nstreams = dataset_params.get('min_item_nstreams', 1500)
valid_users_path = os.path.join(
    cache_path, f'users_hist_longer_than_'
                f'{dataset_params["ndays_min"]}days_morethan{min_user_nstreams}-ints.pkl')
valid_items_path = os.path.join(
    cache_path, f'items_in_hist_longer_than_'
                f'{dataset_params["ndays_min"]}days_morethan{min_item_nstreams}-ints.pkl')
# filter recent history for valid users
valid_users_recent_history_path = os.path.join(
    dataset_params['path'], dataset_params['name'],
    f'interactions_recent{dataset_params["ndays_recent"]}days_'
    f'histmin{dataset_params["ndays_min"]}days_'
    f'unstreams{min_user_nstreams}_instreams{min_item_nstreams}.csv')
data_path = os.path.join(dataset_params['path'], dataset_params['name'],
                         dataset_params['files']['streams'])
ts_thres = dataset_params["ndays_recent"] * 24 * 3600
n_users = 0
last_user_id = -1
data = []
processed_users = dict()
logger = get_logger()

if not os.path.exists(valid_users_recent_history_path):
    logger.info(f'Start get recent history for valid users')
    logger.info(f'Load valid users from {valid_users_path}')
    valid_users = pickle.load(open(valid_users_path, 'rb'))
    logger.info(f'Load valid items from {valid_items_path}')
    valid_items = pickle.load(open(valid_items_path, 'rb'))
    with open(data_path) as file:
        for line in file:
            arr = line.split(sep=dataset_params.get('sep', ','))
            user_id, artist_id, track_id, timestamp = \
                int(arr[0]), int(arr[1]), int(arr[3]), int(arr[4])
            if user_id not in valid_users or track_id not in valid_items:
                continue
            if valid_users[user_id]['max_ts'] - timestamp > ts_thres:
                continue
            if user_id != last_user_id:
                if n_users > 1 and n_users % 3000 == 0:
                    logger.info(f'Process data for {n_users} users')
                    data = pd.DataFrame(data, columns=dataset_params['col_names'])
                    # push chunk to file
                    mode = 'w' if not os.path.exists(valid_users_recent_history_path) \
                        else 'a'
                    data.to_csv(valid_users_recent_history_path, sep=',',
                                header=False, mode=mode, index=False)
                    # explicit remove data due to memory leak, don't know why
                    del data
                    gc.collect()
                    # refrech data
                    data = []
                if user_id not in processed_users:
                    n_users += 1
                    processed_users[user_id] = True
                last_user_id = user_id
            data.append((user_id, artist_id, track_id, timestamp))
    # push last chunk to file
    data = pd.DataFrame(data, columns=dataset_params['col_names'])
    data.to_csv(valid_users_recent_history_path, sep=',',
                header=False, mode='a', index=False)
else:
    logger.info(f'Read user recent history from {valid_users_recent_history_path}')
    data = pd.read_csv(valid_users_recent_history_path,
                       names=dataset_params['col_names'])
print(f'Number of interactions: {len(data)}')
print(f'Number of users: {len(data["user_id"].unique())}')
print(f'Number of items: {len(data["track_id"].unique())}')
