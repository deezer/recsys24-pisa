import os
import pickle
from collections import defaultdict
from tqdm import tqdm

dataset = 'lfm1b'
minsess = 50
cache_path = f'cache/{dataset}/min{minsess}sess'
data_path = f'exp/data/{dataset}'

NTOP = 5
N_VALID = 5
N_TEST = 10
RECENT_HIST = -1
SAMPLES_STEP = 5
SEQLEN = 20

# load user sessions
user_sess_path = os.path.join(cache_path, 'user_sessions.pkl')
print(f'Load user session streams from {user_sess_path}')
user_sessions = pickle.load(open(user_sess_path, 'rb'))

# load train interaction indexes
train_session_indexes_path = os.path.join(
            cache_path,
            f'train_session_indexes_recenthist{RECENT_HIST}_'
            f'samples-step{SAMPLES_STEP}_seqlen{SEQLEN}_'
            f'{N_VALID}v_{N_TEST}t.pkl')
print(f'Load session indexes from {train_session_indexes_path}')
train_session_indexes = pickle.load(open(train_session_indexes_path, 'rb'))

# load track-artist map
track_artist_map_path = os.path.join(
            cache_path, f'{dataset}_track_artist.pkl')
track_art_map = pickle.load(open(track_artist_map_path, 'rb'))

nartists_hist = defaultdict(int)
for uid, nxt_idx in tqdm(train_session_indexes,
                         desc='Num artists in session histogram...'):
    idx = SEQLEN - 1
    for sess in reversed(user_sessions[uid][:nxt_idx]):
        item_ids = [track_art_map[tid] for tid in sess['track_ids']]
        n_arts = len(set(item_ids))
        nartists_hist[n_arts] += 1
        idx -= 1
        if idx == -1:
            break
print(nartists_hist)
