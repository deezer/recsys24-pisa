import os
import pickle
from scipy.special import softmax

from pisa.constants import SESSION_LEN
from pisa.utils.adjmat import *


def load_actr_bll_weights(cache_path, user_sessions, data_split, seqlen,
                          train_session_indexes=None,
                          recent_hist=-1,
                          item_type='track', bll_type='ts',
                          mode='train',
                          decay=0.5, **kwargs):
    logger = kwargs['logger']
    samples_step = kwargs['samples_step']
    bll_weights_path = os.path.join(cache_path,
                                    f'{mode}_{item_type}_recenthist{recent_hist}_'
                                    f'blltype{bll_type}_blldecay{decay}_weights_'
                                    f'seqlen{seqlen}_step{samples_step}.pkl')
    if not os.path.exists(bll_weights_path):
        logger.info(f'Build {mode} {item_type} BLL weights dictionary...')
        # t_refs for ACT-R
        t_refs = _load_t_refs(user_sessions, data_split,
                              train_session_indexes=train_session_indexes,
                              mode=mode, bll_type=bll_type)
        track_art_map = None if 'track_art_map' not in kwargs else kwargs['track_art_map']
        if mode == 'train':
            user_item_seq_hist = _load_train_user_item_seq_hist(
                cache_path, user_sessions, data_split,
                recent_hist=recent_hist,
                seqlen=seqlen,
                samples_step=samples_step,
                train_session_indexes=train_session_indexes,
                logger=logger,
                item_type=item_type,
                track_art_map=track_art_map)
        else:
            user_item_seq_hist = _load_test_user_item_seq_hist(
                cache_path,
                user_sessions,
                data_split,
                recent_hist=recent_hist,
                logger=logger,
                item_type=item_type,
                track_art_map=track_art_map,
                seqlen=seqlen,
                samples_step=samples_step)
        bll_weights = {}
        if mode == 'test':
            for uid, item_seq_hist in tqdm(user_item_seq_hist.items(),
                                           desc=f'Calculate {mode}-{item_type} bll-{bll_type} weights...'):
                user_bll_weights = {iid: _calculate_bll_weights(seq_hist, t_refs[uid],
                                                                decay, bll_type)
                                    for iid, seq_hist in item_seq_hist.items()}
                # normalize
                k = list(user_bll_weights.keys())
                v = np.array(list(user_bll_weights.values()))
                v = softmax(v)
                bll_weights[uid] = dict(zip(k, v))
        else:
            for uid, item_seq_hist in tqdm(user_item_seq_hist.items(),
                                           desc=f'Calculate {mode}-{item_type} bll-{bll_type} weights...'):
                user_bll_weights = {}
                for nxt_idx, hist in item_seq_hist.items():
                    user_nxtidx_bll_weights = {
                        iid: _calculate_bll_weights(seq_hist, t_refs[uid][nxt_idx],
                                                    decay, bll_type)
                        for iid, seq_hist in hist.items()}
                    # normalize
                    k = list(user_nxtidx_bll_weights.keys())
                    v = np.array(list(user_nxtidx_bll_weights.values()))
                    v = softmax(v)
                    user_bll_weights[nxt_idx] = dict(zip(k, v))
                bll_weights[uid] = user_bll_weights
        pickle.dump(bll_weights, open(bll_weights_path, 'wb'))
    else:
        logger.info(f'Load {mode}-{item_type} bll-{bll_type} weights '
                    f'step {samples_step} dictionary...')
        bll_weights = pickle.load(open(bll_weights_path, 'rb'))
    return bll_weights


def load_actr_spread_weights(cache_path, user_sessions, train_session_indexes,
                             recent_hist, data_split, item_type='track', **kwargs):
    logger = kwargs['logger']
    hop = kwargs['hop']
    samples_step = kwargs['samples_step']
    seqlen = kwargs['seqlen']
    n_last_sess = kwargs['n_last_sess']
    track_ids = kwargs['track_ids']
    art_ids = kwargs['art_ids']
    spread_weights_path = os.path.join(cache_path,
                                       f'{item_type}_spread_weights_'
                                       f'recenthist{recent_hist}_{hop}hop_'
                                       f'seqlen{seqlen}_step{samples_step}_'
                                       f'nlastsess{n_last_sess}.pkl')
    if not os.path.exists(spread_weights_path):
        logger.info(f'Load {item_type} {hop}-hop adjacency matrices...')
        train_adj_matrix_path = os.path.join(cache_path,
                                             f'train_{item_type}_sess-level_adj_matrix_'
                                             f'recenthist{recent_hist}_{hop}hop_'
                                             f'last{n_last_sess}sessctx.npz')
        test_adj_matrix_path = os.path.join(cache_path,
                                            f'test_{item_type}_sess-level_adj_matrix_'
                                            f'recenthist{recent_hist}_{hop}hop_'
                                            f'last{n_last_sess}sessctx.npz')
        if os.path.exists(train_adj_matrix_path):
            train_adj_matrix = sp.load_npz(train_adj_matrix_path).toarray()
        else:
            logger.info(f'Build train {item_type} session-'
                        f'level adjacency matrix...')
            train_adj_matrix = _build_adj_matrix(mode='train',
                                                 user_sessions=user_sessions,
                                                 level='sess',
                                                 recent_hist=-1,
                                                 data_split=data_split,
                                                 track_art_map=None,
                                                 track_ids=track_ids,
                                                 n_last_sess=n_last_sess,
                                                 art_ids=art_ids,
                                                 hop=hop)
            sp.save_npz(train_adj_matrix_path, train_adj_matrix)
            logger.info(f'Save train {item_type} {hop}-hop '
                        f'adjacency matrix to {train_adj_matrix_path}...')
        if os.path.exists(test_adj_matrix_path):
            test_adj_matrix = sp.load_npz(test_adj_matrix_path).toarray()
        else:
            logger.info(f'Build test {item_type} session-'
                        f'level adjacency matrix...')
            test_adj_matrix = _build_adj_matrix(mode='test',
                                                user_sessions=user_sessions,
                                                level='sess',
                                                recent_hist=-1,
                                                data_split=data_split,
                                                track_art_map=None,
                                                track_ids=track_ids,
                                                n_last_sess=n_last_sess,
                                                art_ids=art_ids,
                                                hop=hop)
            sp.save_npz(test_adj_matrix_path, test_adj_matrix)
            logger.info(f'Save test {item_type} {hop}-hop '
                        f'adjacency matrix to {test_adj_matrix_path}...')

        spread_weights = {'train': {}, 'test': {}}
        # Train corpus
        for uid, nxt_idx in tqdm(
                train_session_indexes,
                desc=f'Build train spread weights from {item_type} {hop}-hop adj matrix...'):
            if uid not in spread_weights['train']:
                spread_weights['train'][uid] = {
                    nxt_idx: _build_spread_weights_for_user_index(
                        nxt_idx, user_sessions[uid], item_type, train_adj_matrix, **kwargs)}
            else:
                spread_weights['train'][uid][nxt_idx] = \
                    _build_spread_weights_for_user_index(
                        nxt_idx, user_sessions[uid], item_type, train_adj_matrix, **kwargs)
        # Val + Test corpus
        for uid, sessions in tqdm(
                user_sessions.items(),
                desc=f'Build test spread weights from {item_type} {hop}-hop adj matrix...'):
            for nxt_idx in range(-(data_split['valid'] + data_split['test']),
                                 -data_split['test']):
                if uid not in spread_weights['test']:
                    spread_weights['test'][uid] = {
                        nxt_idx: _build_spread_weights_for_user_index(
                            nxt_idx, sessions, item_type, train_adj_matrix, **kwargs)}
                else:
                    spread_weights['test'][uid][nxt_idx] = \
                        _build_spread_weights_for_user_index(
                            nxt_idx, sessions, item_type, train_adj_matrix, **kwargs)
            for nxt_idx in range(-data_split['test'], 0):
                if uid not in spread_weights['test']:
                    spread_weights['test'][uid] = {
                        nxt_idx: _build_spread_weights_for_user_index(
                            nxt_idx, sessions, item_type, test_adj_matrix, **kwargs)}
                else:
                    spread_weights['test'][uid][nxt_idx] = \
                        _build_spread_weights_for_user_index(
                            nxt_idx, sessions, item_type, test_adj_matrix, **kwargs)
        pickle.dump(spread_weights, open(spread_weights_path, 'wb'))
    else:
        logger.info(f'Load {item_type} spread weights '
                    f'from {spread_weights_path}...')
        spread_weights = pickle.load(open(spread_weights_path, 'rb'))
    return spread_weights


def load_actr_spread_weights_for_posout(cache_path, user_sessions,
                                        train_session_indexes, recent_hist,
                                        item_type='track', **kwargs):
    """This is a patch for the function _load_actr_spread_weights """
    logger = kwargs['logger']
    hop = kwargs['hop']
    samples_step = kwargs['samples_step']
    seqlen = kwargs['seqlen']
    n_last_sess = kwargs['n_last_sess']
    spread_weights_path = os.path.join(cache_path,
                                       f'pos_{item_type}_spread_weights_'
                                       f'recenthist{recent_hist}_{hop}hop_'
                                       f'seqlen{seqlen}_step{samples_step}_'
                                       f'nlastsess{n_last_sess}.pkl')
    if not os.path.exists(spread_weights_path):
        logger.info(f'Load train {item_type} adjacency matrices...')
        adj_matrix_path = os.path.join(cache_path,
                                       f'train_{item_type}_sess-level_adj_matrix_'
                                       f'recenthist{recent_hist}_{hop}hop_'
                                       f'last{n_last_sess}sessctx.npz')
        adj_matrix = sp.load_npz(adj_matrix_path).toarray()
        spread_weights = {}
        for uid, nxt_idx in tqdm(
                train_session_indexes,
                desc=f'Build train POS-OUTPUT spread weights '
                     f'from {item_type} adj matrix...'):
            if uid not in spread_weights:
                spread_weights[uid] = {
                    nxt_idx: _build_spread_weights_for_user_next_index(
                        nxt_idx, user_sessions[uid], item_type, adj_matrix, **kwargs)}
            else:
                spread_weights[uid][nxt_idx] = \
                    _build_spread_weights_for_user_next_index(
                        nxt_idx, user_sessions[uid], item_type, adj_matrix, **kwargs)
        pickle.dump(spread_weights, open(spread_weights_path, 'wb'))
    else:
        logger.info(f'Load POS OUTPUT {item_type} spread weights '
                    f'from {spread_weights_path}...')
        spread_weights = pickle.load(open(spread_weights_path, 'rb'))
    return spread_weights


def _load_t_refs(user_sessions, data_split, train_session_indexes=None,
                 mode='train', bll_type='ts'):
    t_refs = defaultdict(dict) if mode == 'train' else {}
    if mode == 'train':
        for uid, nxt_idx in tqdm(train_session_indexes, desc='Extract train t_ref...'):
            if bll_type == 'ts':
                t_refs[uid][nxt_idx] = user_sessions[uid][nxt_idx]['context']['ts']
            else:
                t_refs[uid][nxt_idx] = nxt_idx
    else:
        ref_idx = -data_split['test']
        for uid, sessions in tqdm(user_sessions.items(), desc='Extract test t_ref...'):
            if bll_type == 'ts':
                t_refs[uid] = sessions[ref_idx]['context']['ts']
            else:
                t_refs[uid] = len(sessions) + ref_idx
    return t_refs


def _calculate_bll_weights(seq_hist, t_ref, d, bll_type='ts'):
    weight = 0.
    for idx, ts in seq_hist:
        if bll_type == 'ts':
            weight += np.power(t_ref - ts, -d)
        else:
            weight += np.power(t_ref - idx, -d)
    return np.log(weight)


def _load_train_user_item_seq_hist(cache_path, user_sessions, data_split, recent_hist,
                                   seqlen, train_session_indexes,
                                   item_type='track', **kwargs):
    logger = kwargs['logger']
    user_item_interactions_path = os.path.join(
        cache_path,
        f'train_user_{item_type}_interactions_dict_recenthist{recent_hist}_'
        f'{data_split["valid"]}v_{data_split["test"]}t_seqlen{seqlen}_step{kwargs["samples_step"]}.pkl')
    if not os.path.exists(user_item_interactions_path):
        logger.info(f'Extract train user-{item_type} sequential '
                    f'history for ACT-R...')
        output = defaultdict(dict)
        # calculate for each nxt_idx
        for uid, nxt_idx in tqdm(train_session_indexes,
                                 desc=f'Build user-{item_type} history...'):
            sessions = user_sessions[uid]
            for idx, ss in enumerate(sessions[nxt_idx - seqlen:nxt_idx]):
                for tid in ss['track_ids']:
                    rid = tid if item_type == 'track' else kwargs['track_art_map'][tid]
                    if uid not in output or nxt_idx not in output[uid]:
                        output[uid][nxt_idx] = {}
                    if rid not in output[uid][nxt_idx]:
                        output[uid][nxt_idx][rid] = [(idx + nxt_idx - seqlen, ss['context']['ts'])]
                    else:
                        output[uid][nxt_idx][rid].append((idx + nxt_idx - seqlen, ss['context']['ts']))
        # merge by order of nxt_idx
        user_item_hist = defaultdict(dict)
        for uid, item_hist_dict in tqdm(output.items(), desc='Merge history...'):
            nxt_indices = sorted(list(item_hist_dict.keys()))
            if len(nxt_indices) == 1:
                curr_nxt_idx = nxt_indices[0]
                curr_hist = item_hist_dict[curr_nxt_idx]
                user_item_hist[uid][curr_nxt_idx] = curr_hist
            else:
                for i in range(0, len(nxt_indices) - 1):
                    curr_nxt_idx = nxt_indices[i]
                    curr_hist = item_hist_dict[curr_nxt_idx]
                    if curr_nxt_idx not in user_item_hist[uid]:
                        user_item_hist[uid][curr_nxt_idx] = curr_hist
                    else:
                        curr_hist = user_item_hist[uid][curr_nxt_idx]
                    nxt_nxt_idx = nxt_indices[i + 1]
                    nxt_hist = item_hist_dict[nxt_nxt_idx]
                    user_item_hist[uid][nxt_nxt_idx] = _dict_merge(curr_hist, nxt_hist)
        pickle.dump(user_item_hist, open(user_item_interactions_path, 'wb'))
    else:
        logger.info(f'Load train user-{item_type} sequential history for ACT-R from '
                    f'{user_item_interactions_path}')
        user_item_hist = pickle.load(open(user_item_interactions_path, 'rb'))
    return user_item_hist


def _dict_merge(dict_a, dict_b):
    output = defaultdict(list)
    merged_keys = set(list(dict_a.keys()) + list(dict_b.keys()))
    for iid in merged_keys:
        if iid in dict_a:
            output[iid] = dict_a[iid]
        if iid in dict_b:
            output[iid] = list(set(output[iid] + dict_b[iid]))
    return output


def _load_test_user_item_seq_hist(cache_path, user_sessions, data_split, recent_hist,
                                  item_type='track', **kwargs):
    logger = kwargs['logger']
    split_idx = -data_split['test']
    user_item_interactions_path = os.path.join(
        cache_path,
        f'test_user_{item_type}_interactions_dict_recenthist{recent_hist}_'
        f'{data_split["valid"]}v_{data_split["test"]}t_seqlen{kwargs["seqlen"]}_step{kwargs["samples_step"]}.pkl')
    if not os.path.exists(user_item_interactions_path):
        logger.info(f'Extract test user-{item_type} sequential '
                    f'history for ACT-R...')
        output = {}
        for uid, sessions in tqdm(user_sessions.items(),
                                  desc=f'Build user-{item_type} history...'):
            output[uid] = defaultdict(list)
            start_idx = split_idx - recent_hist if recent_hist > 0 else 0
            for idx, ss in enumerate(sessions[start_idx:split_idx]):
                for tid in ss['track_ids']:
                    if item_type == 'track':
                        output[uid][tid].append((idx, ss['context']['ts']))
                    else:
                        aid = kwargs['track_art_map'][tid]
                        output[uid][aid].append((idx, ss['context']['ts']))
        pickle.dump(output, open(user_item_interactions_path, 'wb'))
    else:
        logger.info(f'Load test user-{item_type} sequential history for ACT-R from '
                    f'{user_item_interactions_path}')
        output = pickle.load(open(user_item_interactions_path, 'rb'))
    return output


def _build_spread_weights_for_user_index(nxt_idx, user_sessions, item_type, adj_matrix,
                                         **kwargs):
    seqlen = kwargs['seqlen']
    sessions = user_sessions[nxt_idx - seqlen:nxt_idx]
    item_ids_list = _extract_item_ids(sessions, item_type, **kwargs)
    return _build_spread_weights(item_ids_list, adj_matrix)


def _build_spread_weights_for_user_next_index(nxt_idx, user_sessions,
                                              item_type, adj_matrix, **kwargs):
    sessions = [user_sessions[nxt_idx]]
    item_ids_list = _extract_item_ids(sessions, item_type, **kwargs)
    return _build_spread_weights(item_ids_list, adj_matrix)


def _extract_item_ids(sessions, item_type='track', **kwargs):
    output = []
    if item_type == 'track':
        track_ids_map = {tid: idx for idx, tid in enumerate(kwargs['track_ids'])}
        for ss in sessions:
            track_ids = [track_ids_map[tid] for tid in ss['track_ids']]
            output.append(track_ids)
    else:
        art_ids_map = {aid: idx for idx, aid in enumerate(kwargs['art_ids'])}
        track_art_map = kwargs['track_art_map']
        for ss in sessions:
            art_ids = [art_ids_map[track_art_map[tid]] for tid in ss['track_ids']]
            output.append(art_ids)
    return output


def _build_spread_weights(item_ids_list, adj_matrix):
    spread_weights = []
    for item_ids in item_ids_list:
        weights = np.zeros(SESSION_LEN, dtype=np.float32)
        for idx, iid_1 in enumerate(item_ids):
            w = 0.
            for iid_2 in item_ids:
                if iid_2 != iid_1:
                    w += adj_matrix[iid_1, iid_2]
            weights[idx] = w
        spread_weights.append(weights)
    return np.array(spread_weights, dtype=np.float32)


def _build_spread_weights_with_item_embs(item_ids_list, adj_matrix,
                                         item_embeddings):
    spread_weights = []
    embeddings = np.array(list(item_embeddings.values()))
    for item_ids in item_ids_list:
        weights = np.zeros(SESSION_LEN, dtype=np.float32)
        for idx, iid_1 in enumerate(item_ids):
            w = 0.
            for iid_2 in item_ids:
                if iid_2 != iid_1:
                    sim = np.dot(embeddings[iid_1, :],
                                 embeddings[iid_2, :])
                    w += sim * adj_matrix[iid_1, iid_2]
            weights[idx] = w
        spread_weights.append(weights)
    return np.array(spread_weights, dtype=np.float32)


def _build_adj_matrix(mode, user_sessions, level, recent_hist,
                      track_ids, data_split, track_art_map,
                      art_ids, n_last_sess, hop=1):
    adj_matrix = build_sparse_adjacency_matrix(user_sessions=user_sessions,
                                               level=level,
                                               recent_hist=recent_hist,
                                               track_ids=track_ids,
                                               mode=mode,
                                               data_split=data_split,
                                               track_art_map=track_art_map,
                                               art_ids=art_ids,
                                               n_last_sess=n_last_sess)
    adj_matrix = normalize_adj(adj_matrix)
    adj_matrix = adj_matrix.toarray()
    mul = adj_matrix
    w_mul = adj_matrix
    coeff = 1.0
    if hop > 1:
        for w in range(1, hop):
            coeff *= 0.85
            w_mul *= adj_matrix
            w_mul = remove_diag(w_mul)
            w_adj_matrix = normalize_adj(w_mul)
            mul += coeff * w_adj_matrix
    adj_matrix = mul
    adj_matrix = sp.csr_matrix(adj_matrix)
    return adj_matrix
