import pickle
from tqdm import tqdm
from collections import defaultdict
from utils import *


# read configuration file
params = load_configuration('configs/lfm1b.json')
dataset_params = params['dataset']
data_path = os.path.join(dataset_params['path'], dataset_params['name'],
                         dataset_params['files']['interactions'])
output_path = os.path.join('cache', dataset_params['name'],
                           f'min{dataset_params["min_sessions"]}sess')
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

total_users_path = os.path.join(output_path, 'user_minmax_timestamps.pkl')
valid_users_path = os.path.join(output_path,
                                f'user_hist_longer_than_'
                                f'{dataset_params["ndays_min"]}days.pkl')
user_numstreams_path = os.path.join(
    output_path, f'user_histmin{dataset_params["ndays_min"]}days_'
                 f'nstreams.pkl')
item_numstreams_path = os.path.join(
    output_path, f'item_in_histmin{dataset_params["ndays_min"]}days_'
                 f'nstreams.pkl')
min_user_nstreams = dataset_params.get('min_user_nstreams', 1000)
min_item_nstreams = dataset_params.get('min_item_nstreams', 1500)

logger = get_logger()
user_numstreams = defaultdict(int)
item_numstreams = defaultdict(dict)

last_user_id = -1
total_users = defaultdict(dict)

if not os.path.exists(valid_users_path):
    if not os.path.exists(total_users_path):
        logger.info(f'Start processing {total_users_path}...')
        with open(data_path) as file:
            for _, line in enumerate(tqdm(file)):
                arr = line.split(sep=dataset_params.get('sep', ','))
                user_id, artist_id, track_id, timestamp = \
                    int(arr[0]), int(arr[1]), int(arr[3]), int(arr[4])
                if user_id != last_user_id:
                    last_user_id = user_id
                if user_id not in total_users:
                    total_users[user_id] = {'min_ts': timestamp,
                                            'max_ts': timestamp}
                else:
                    if timestamp < total_users[user_id]['min_ts']:
                        total_users[user_id]['min_ts'] = timestamp
                    if timestamp > total_users[user_id]['max_ts']:
                        total_users[user_id]['max_ts'] = timestamp
        pickle.dump(total_users, open(total_users_path, 'wb'))
    else:
        logger.info(f'Load user history timstamps from {total_users_path}...')
        total_users = pickle.load(open(total_users_path, 'rb'))
    logger.info(f'Total number of users: {len(total_users)}')
    logger.info(f'Filter out users with history less than '
                f'{dataset_params["ndays_min"]}')
    valid_users = {uid: minmax_ts for uid, minmax_ts in total_users.items()
                   if delta_ts(minmax_ts['min_ts'],
                               minmax_ts['max_ts']) >= dataset_params['ndays_min']}
    pickle.dump(valid_users, open(valid_users_path, 'wb'))
else:
    logger.info(f'Load user history >= {dataset_params["ndays_min"]} days '
                f'from {valid_users_path}...')
    valid_users = pickle.load(open(valid_users_path, 'rb'))
logger.info(f'Number of valid users: {len(valid_users)}')

if not os.path.exists(user_numstreams_path) or \
        not os.path.exists(item_numstreams_path):
    with open(data_path) as file:
        for _, line in enumerate(tqdm(file)):
            arr = line.split(sep=dataset_params.get('sep', ','))
            user_id, artist_id, track_id, timestamp = \
                int(arr[0]), int(arr[1]), int(arr[3]), int(arr[4])
            if user_id not in valid_users:
                continue
            user_numstreams[user_id] += 1
            if user_id in item_numstreams[track_id]:
                item_numstreams[track_id][user_id] += 1
            else:
                item_numstreams[track_id][user_id] = 1
    pickle.dump(user_numstreams, open(user_numstreams_path, 'wb'))
    pickle.dump(item_numstreams, open(item_numstreams_path, 'wb'))
else:
    user_numstreams = pickle.load(open(user_numstreams_path, 'rb'))
    item_numstreams = pickle.load(open(item_numstreams_path, 'rb'))

output_valid_users_path = os.path.join(
    output_path, f'users_hist_longer_than_'
                 f'{dataset_params["ndays_min"]}days_morethan{min_user_nstreams}-ints.pkl')
output_valid_items_path = os.path.join(
    output_path, f'items_in_hist_longer_than_'
                 f'{dataset_params["ndays_min"]}days_morethan{min_item_nstreams}-ints.pkl')

if not os.path.exists(output_valid_users_path):
    valid_user_numstreams = {uid: ns for uid, ns in user_numstreams.items()
                             if ns >= min_user_nstreams}
    valid_users = {uid: minmax_ts for uid, minmax_ts in valid_users.items()
                   if uid in valid_user_numstreams}
    pickle.dump(valid_users, open(output_valid_users_path, 'wb'))
else:
    logger.info(f'Load valid users from {output_valid_users_path}')
    valid_users = pickle.load(open(output_valid_users_path, 'rb'))

if not os.path.exists(output_valid_items_path):
    item_numstreams = {iid: len(users) for iid, users in item_numstreams.items()}
    valid_items = {iid: ns for iid, ns in item_numstreams.items() if ns >= min_item_nstreams}
    pickle.dump(valid_items, open(output_valid_items_path, 'wb'))
else:
    logger.info(f'Load valid items from {output_valid_items_path}')
    valid_items = pickle.load(open(output_valid_items_path, 'rb'))

logger.info(f'Number of final users: {len(valid_users)}')
logger.info(f'Number of final items: {len(valid_items)}')
