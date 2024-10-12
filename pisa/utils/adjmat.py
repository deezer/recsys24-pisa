from collections import defaultdict
import itertools
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp


def build_sparse_adjacency_matrix(user_sessions, level, recent_hist,
                                  track_ids, mode, data_split,
                                  **kwargs):
    track_art_map = None
    if kwargs['track_art_map'] is None:
        item_dict = {tid: idx for idx, tid in enumerate(track_ids)}
        n_items = len(track_ids)
    else:
        track_art_map = kwargs['track_art_map']
        item_ids = list(set(kwargs['art_ids']))
        item_dict = {iid: idx for idx, iid in enumerate(item_ids)}
        n_items = len(item_ids)
    pairs = {}
    split_idx = -(data_split['test'] + data_split['valid']) \
        if mode == 'train' else -data_split['test']
    n_last_sess = kwargs['n_last_sess']
    if n_last_sess > 0:
        start_idx = split_idx - recent_hist + 1 if recent_hist > 0 else 1
    else:
        start_idx = split_idx - recent_hist if recent_hist > 0 else 0
    if level == 'sess':
        if n_last_sess > 0:
            for uid, sessions in tqdm(user_sessions.items(),
                                      desc=f'Calculate session-level {n_last_sess} '
                                           f'last sess co-occurence frequencies...'):
                for idx in range(start_idx, len(sessions) + split_idx):
                    sess = sessions[idx]
                    if n_last_sess > 1:
                        prev_idx = idx - n_last_sess if idx - n_last_sess >= 0 else 0
                        prev_sessions = sessions[prev_idx:idx]
                    else:
                        prev_sessions = [sessions[idx - 1]]
                    if track_art_map is None:
                        id_list = [item_dict[tid] for tid in sess['track_ids']]
                        prev_id_list = get_id_list(prev_sessions, item_dict)
                    else:
                        id_list = [item_dict[track_art_map[tid]] for tid in sess['track_ids']]
                        prev_id_list = get_id_list(prev_sessions, item_dict,
                                                   track_art_map)
                    for t in list(itertools.product(prev_id_list, id_list)):
                        add_tuple(t, pairs)
        else:
            for uid, sessions in tqdm(user_sessions.items(),
                                      desc='Calculate session-level co-occurence frequencies...'):
                for sess in sessions[start_idx:split_idx]:
                    if track_art_map is None:
                        id_list = [item_dict[tid] for tid in sess['track_ids']]
                    else:
                        id_list = [item_dict[track_art_map[tid]] for tid in
                                   sess['track_ids']]
                    for t in list(itertools.product(id_list, id_list)):
                        add_tuple(t, pairs)
    else:
        user_hist = defaultdict(set)
        for uid, sessions in tqdm(user_sessions.items(),
                                  desc='Extract user history...'):
            for sess in sessions[start_idx-1:split_idx]:
                for tid in sess['track_ids']:
                    if track_art_map is None:
                        user_hist[uid].add(item_dict[tid])
                    else:
                        user_hist[uid].add(item_dict[track_art_map[tid]])
        for uid, hist in tqdm(user_hist.items(),
                              desc='Calculate user-level co-occurence frequencies...'):
            id_list = list(hist)
            for t in list(itertools.product(id_list, id_list)):
                add_tuple(t, pairs)
    return create_sparse_matrix(pairs, n_items)


def add_tuple(t, pairs):
    assert len(t) == 2
    if t[0] != t[1]:
        if t not in pairs:
            pairs[t] = 1
        else:
            pairs[t] += 1


def create_sparse_matrix(pairs, n_items):
    row = [p[0] for p in pairs]
    col = [p[1] for p in pairs]
    data = [pairs[p] for p in pairs]

    adj_matrix = sp.csc_matrix((data, (row, col)), shape=(n_items, n_items),
                               dtype="float32")
    nb_nonzero = len(pairs)
    density = nb_nonzero * 1.0 / n_items / n_items
    print(f"Density: {density:.6f}")
    return sp.csc_matrix(adj_matrix, dtype="float32")


def normalize_adj(adj_matrix):
    """Symmetrically normalize adjacency matrix."""
    row_sum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized_matrix = adj_matrix.dot(d_mat_inv_sqrt) \
        .transpose().dot(d_mat_inv_sqrt)
    return normalized_matrix.tocsr()


def remove_diag(adj_matrix):
    new_adj_matrix = sp.csr_matrix(adj_matrix)
    new_adj_matrix.setdiag(0.0)
    new_adj_matrix.eliminate_zeros()
    return new_adj_matrix


def get_id_list(sessions, item_dict, track_art_map=None):
    id_list = []
    for sess in sessions:
        if track_art_map is None:
            id_list += [item_dict[tid] for tid in sess['track_ids']]
        else:
            id_list += [item_dict[track_art_map[tid]] for tid in
                        sess['track_ids']]
    return list(set(id_list))
