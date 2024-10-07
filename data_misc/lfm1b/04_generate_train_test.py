import pickle
import numpy as np
import pandas as pd

from utils import *


params = load_configuration('configs/lfm1b_pretrain_usertrack.json')
dataset_params = params['dataset']
training_params = params['training']
cache_params = params['cache']
model_params = training_params['model']
cache_path = os.path.join('cache', dataset_params['name'],
                          f'min{dataset_params["min_sessions"]}sess')
min_user_nstreams = dataset_params.get('min_user_nstreams', 1000)
min_item_nstreams = dataset_params.get('min_item_nstreams', 1500)

MIN_SESSIONS_IN_HISTORY = 50

data_path = os.path.join(
    dataset_params['path'], dataset_params['name'],
    f'user_sessions_recent{dataset_params["ndays_recent"]}days_'
    f'histmin{dataset_params["ndays_min"]}days_'
    f'unstreams{min_user_nstreams}_instreams{min_item_nstreams}_'
    f'nsess{MIN_SESSIONS_IN_HISTORY}.csv')

logger = get_logger()

# read dataset
data = pd.read_csv(data_path, names=dataset_params['col_names'])
data = data[['user_id', 'track_id']].drop_duplicates()
data = data.rename(columns={'user_id': 'org_user',
                            'track_id': 'org_item'})
user_ids = data.org_user.unique()
item_ids = data.org_item.unique()

# mapping to internal ids
user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}
data.loc[:, 'user'] = data.org_user.apply(lambda x: user_id_map[x])
data.loc[:, 'item'] = data.org_item.apply(lambda x: item_id_map[x])

cache_path = os.path.join(cache_path, model_params['name'])
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

train_interactions_name = cache_params['train_interactions']
test_interactions_name = cache_params['test_interactions']
entities_name = cache_params.get('entities', 'entities')

train_interactions_path = os.path.join(cache_path,
                                       f'{train_interactions_name}.npz')
test_interactions_path = os.path.join(cache_path,
                                      f'{test_interactions_name}.npz')
entities_path = os.path.join(cache_path,
                             f'{entities_name}.npz')

if not os.path.exists(train_interactions_path) or not \
        os.path.exists(test_interactions_path) or not \
        os.path.exists(entities_path):

    # split data
    logger.info('Split data into train/test')
    train_set, test_set = split_data(data,
                                     test_size=dataset_params['test_size'],
                                     random_state=dataset_params['random_state'])
    binary = dataset_params.get('binary', True)
    train_interactions = df_to_mat(train_set,
                                   n_rows=len(user_ids),
                                   n_cols=len(item_ids),
                                   binary=binary)
    test_interactions = df_to_mat(test_set,
                                  n_rows=len(user_ids),
                                  n_cols=len(item_ids),
                                  binary=binary)
    # save to file
    logger.debug('Save data to cache')
    sp.save_npz(train_interactions_path, train_interactions)
    sp.save_npz(test_interactions_path, test_interactions)
    np.savez(entities_path, user_ids=user_ids, item_ids=item_ids)
else:
    logger.info('Load interaction matrices')
    train_interactions = sp.load_npz(train_interactions_path)
    test_interactions = sp.load_npz(test_interactions_path)
    entity_ids = np.load(entities_path, allow_pickle=True)
    user_ids = entity_ids['user_ids']
    item_ids = entity_ids['item_ids']

logger.info(f'Num users: {len(user_ids)}')
logger.info(f'Num items: {len(item_ids)}')
logger.info(f'Num of train interactions: '
            f'{train_interactions.count_nonzero()}')
logger.info(f'Num of test interactions: '
            f'{test_interactions.count_nonzero()}')
