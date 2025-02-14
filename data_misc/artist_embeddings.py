import os
import pickle
import numpy as np
from collections import defaultdict

dataset = 'lfm1b'
minsess = 50
cache_path = f'cache/{dataset}/min{minsess}sess'
data_path = f'exp/data/{dataset}'

NTOP = 5
N_VALID = 5
N_TEST = 10
RECENT_HIST = -1
EMBEDDING_DIM = 128

art_toptracks_path = os.path.join(cache_path, f'artist_toptracks_recenthist{RECENT_HIST}_'
                                              f'{N_VALID}v_{N_TEST}t.pkl')

art_embeddings_path = os.path.join(
    data_path,
    'normalized_pretrained_arttrack_embeddings_recent365days_'
    'histmin365days_unstreams1000_instreams1500_nsess50.pkl')

if not os.path.exists(art_toptracks_path):
    # load art_ids, track_ids
    entities_path = os.path.join(cache_path, f'{dataset}_entities.npz')
    print(f'Load artists, tracks from {entities_path}...')
    entities = np.load(entities_path, allow_pickle=True)
    track_ids = entities['track_ids']
    art_ids = entities['art_ids']
    art_ids_map = {aid: idx for idx, aid in enumerate(art_ids)}
    n_artists = len(art_ids)

    # load track_artist mapping
    track_artist_map_path = os.path.join(cache_path, f'{dataset}_track_artist.pkl')
    print(f'Load track:artist mapping from {track_artist_map_path}...')
    track_art_map = pickle.load(open(track_artist_map_path, 'rb'))

    # load track popularities
    glob_track_pops_path = os.path.join(cache_path,
                                        f'glob_track_popularities_recenthist{RECENT_HIST}_'
                                        f'{N_VALID}v_{N_TEST}t.pkl')
    print(f'Load global track popularities from {glob_track_pops_path}...')
    glob_track_pops = pickle.load(open(glob_track_pops_path, 'rb'))

    # build artist top tracks
    print(f'Build artist top {NTOP} tracks and writ to {art_toptracks_path}')
    art_toptracks = defaultdict(list)
    for tid, aid in track_art_map.items():
        pop = glob_track_pops[tid] if tid in glob_track_pops else 0.
        art_toptracks[aid].append((tid, pop))

    # reorganize top tracks
    for aid, tracks in art_toptracks.items():
        tracks.sort(key=lambda i: i[1], reverse=True)
        top_tracks = tracks[:NTOP]
        art_toptracks[aid] = top_tracks
    pickle.dump(art_toptracks, open(art_toptracks_path, 'wb'))
else:
    print(f'Load artist top tracks from {art_toptracks_path}...')
    art_toptracks = pickle.load(open(art_toptracks_path, 'rb'))

# calculate artist embeddings from top 5 tracks
if not os.path.exists(art_embeddings_path):
    # load track embeddings
    if dataset == 'deezer':
        track_embeddings_path = os.path.join(cache_path,
                                             f'normalized_track_svd_embeddings.pkl')
    else:
        track_embeddings_path = os.path.join(
            'exp/data/lfm1b',
            'normalized_pretrained_user_track_embeddings_recent365days_'
            'histmin365days_unstreams1000_instreams1500_nsess50.pkl')
    track_embeddings = pickle.load(open(track_embeddings_path, 'rb'))
    artist_embeddings = {}
    for aid, toptracks in art_toptracks.items():
        toptracks_embs = [track_embeddings[tid] for tid, _ in toptracks]
        artist_embeddings[aid] = np.mean(toptracks_embs, axis=0)
    pickle.dump(artist_embeddings, open(art_embeddings_path, 'wb'))
else:
    print(f'Load artist embeddings from {art_embeddings_path}...')
    artist_embeddings = pickle.load(open(art_embeddings_path, 'rb'))

print(list(artist_embeddings.items())[:1])
