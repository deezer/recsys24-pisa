import numpy as np

from pisa.constants import *


def train_sample(uid, nxt_idx, dataset, seqlen, n_items, user_tracks,
                 **kwargs):
    user_ids_map = kwargs['user_ids_map']
    # track-level
    track_seq_in, track_seq_actr_bll, track_seq_actr_spread, \
        track_pos_actr_bll, track_pos_actr_spread, \
        track_topbll_ids, track_topbll_scores, \
        track_seq_pos, track_seq_neg = train_item_sample(
        uid, nxt_idx, dataset, seqlen, n_items,
        item_type='track', **kwargs)
    art_seq_in, art_seq_actr_bll, art_seq_actr_spread, \
        art_pos_actr_bll, art_pos_actr_spread, \
        art_topbll_ids, art_topbll_scores, \
        art_seq_pos, art_seq_neg, art_seqin_nitems, \
        art_pos_nitems, art_neg_nitems = train_item_sample(
        uid, nxt_idx, dataset, seqlen, dataset['n_artists'],
        item_type='artist', **kwargs)
    out = (track_seq_in, track_seq_actr_bll, track_seq_actr_spread,
           track_pos_actr_bll, track_pos_actr_spread,
           track_topbll_ids, track_topbll_scores,
           art_seq_in, art_seq_actr_bll, art_seq_actr_spread,
           art_pos_actr_bll, art_pos_actr_spread,
           art_topbll_ids, art_topbll_scores,
           art_seq_pos, art_seq_neg, art_seqin_nitems,
           art_pos_nitems, art_neg_nitems,
           user_ids_map[uid], track_seq_pos, track_seq_neg)
    return out


def test_sample(uid, data, seqlen, nxt_idx, **kwargs):
    user_ids_map = kwargs['user_ids_map']
    seq_in, seq_actr_bll, seq_actr_spread, \
        topbll_ids, topbll_scores = test_item_sample(
        uid, data, seqlen, nxt_idx, item_type='track',
        **kwargs)
    art_seq_in, art_seq_actr_bll, art_seq_actr_spread, \
        art_topbll_ids, art_topbll_scores, art_seqin_nitems = test_item_sample(
        uid, data, seqlen, nxt_idx, item_type='artist',
        **kwargs)
    out = (seq_in, seq_actr_bll, seq_actr_spread, topbll_ids, topbll_scores,
           art_seq_in, art_seq_actr_bll, art_seq_actr_spread,
           art_topbll_ids, art_topbll_scores, art_seqin_nitems,
           user_ids_map[uid], uid)
    return out


def train_item_sample(uid, nxt_idx, dataset, seqlen, n_items, item_type,
                      **kwargs):
    item_ids_map = kwargs[f'{item_type}_ids_map']
    bll_weights = kwargs[f'{item_type}_bll_weights']
    spread_weights = kwargs[f'{item_type}_spread_weights']
    pos_spread_weights = kwargs[f'{item_type}_pos_spread_weights']
    num_favs = kwargs['num_favs'] if item_type == 'track' \
        else kwargs['num_favs_art']
    internal_item_ids = np.arange(1, n_items + 1)
    negsam_strategy = kwargs['negsam_strategy']
    item_pops = kwargs[f'norm_{item_type}_popularities'] \
        if negsam_strategy == NEGSAM_POP else None

    # sequence of input session embeddings
    seq_in = np.zeros(shape=[seqlen, SESSION_LEN], dtype=np.int32)
    seq_actr_bll = np.zeros(shape=[seqlen, SESSION_LEN],
                            dtype=np.float32)
    seq_pos = np.zeros(shape=[seqlen, SESSION_LEN], dtype=np.int32)
    pos_actr_bll = np.zeros(shape=[seqlen, SESSION_LEN],
                            dtype=np.float32)
    seq_neg = np.zeros(shape=[seqlen, SESSION_LEN], dtype=np.int32)
    user_sessions = dataset['user_sessions'][uid]
    nxt = user_sessions[nxt_idx]
    idx = seqlen - 1

    seqin_nitems = np.zeros(shape=seqlen, dtype=np.float32)
    pos_nitems = np.zeros(shape=seqlen, dtype=np.float32)
    neg_nitems = np.zeros(shape=seqlen, dtype=np.float32)
    for sess in reversed(user_sessions[:nxt_idx]):
        if item_type == 'track':
            item_ids = list(set(sess['track_ids']))
            nxt_item_ids = list(set(nxt['track_ids']))
        else:
            item_ids = list(set([dataset['track_art'][tid]
                                 for tid in sess['track_ids']]))
            nxt_item_ids = list(set([dataset['track_art'][tid]
                                     for tid in nxt['track_ids']]))
            seqin_nitems[idx] = len(item_ids)
            pos_nitems[idx] = len(nxt_item_ids)
        seq_in_item_ids = [item_ids_map[iid] for iid in item_ids]
        seq_in[idx][:len(seq_in_item_ids)] = seq_in_item_ids
        seq_actr_bll[idx][:len(seq_in_item_ids)] = \
            [bll_weights[uid][nxt_idx][iid] if iid in bll_weights[uid][nxt_idx]
             else 0. for iid in item_ids]
        seq_pos[idx][:len(nxt_item_ids)] = [item_ids_map[iid]
                                            for iid in nxt_item_ids]
        pos_actr_bll[idx][:len(nxt_item_ids)] = \
            [bll_weights[uid][nxt_idx][iid] if iid in bll_weights[uid][nxt_idx]
             else 0. for iid in nxt_item_ids]
        nxt_set = set([item_ids_map[iid] for iid in nxt_item_ids])
        if negsam_strategy == NEGSAM_UNIFORM:
            neg_ids = np.random.choice(internal_item_ids, size=SESSION_LEN)
            for j, neg in enumerate(neg_ids):
                while neg in nxt_set:
                    neg_ids[j] = neg = np.random.choice(internal_item_ids)
            ne_nitem = np.random.choice(SESSION_LEN)
            while ne_nitem == pos_nitems[idx]:
                ne_nitem = np.random.choice(SESSION_LEN)
        else:
            neg_ids = np.random.choice(internal_item_ids, size=SESSION_LEN,
                                       p=item_pops)
            for j, neg in enumerate(neg_ids):
                while neg in nxt_set:
                    neg_ids[j] = neg = np.random.choice(internal_item_ids,
                                                        p=item_pops)
            ne_nitem = np.random.choice(SESSION_LEN)
            while ne_nitem == pos_nitems[idx]:
                ne_nitem = np.random.choice(SESSION_LEN)
        seq_neg[idx] = neg_ids
        neg_nitems[idx] = ne_nitem
        nxt = sess
        idx -= 1
        if idx == -1:
            break

    seq_actr_spread = spread_weights[uid][nxt_idx]
    pos_actr_spread = np.concatenate(
        (spread_weights[uid][nxt_idx][1:], pos_spread_weights[uid][nxt_idx]),
        axis=0)
    # sort personal items by descending BLL
    sorted_entries = sorted(bll_weights[uid][nxt_idx].items(),
                            key=lambda x: x[1], reverse=True)[:num_favs]

    if len(sorted_entries) < num_favs:
        topbll_ids = np.zeros(num_favs, dtype=np.int32)
        topbll_scores = np.zeros(num_favs, dtype=np.float32)
        topbll_ids[:len(sorted_entries)] = [item_ids_map[iid]
                                            for iid, _ in sorted_entries]
        topbll_scores[:len(sorted_entries)] = \
            [score for _, score in sorted_entries]
    else:
        topbll_ids = np.array([item_ids_map[iid] for iid, _ in sorted_entries])
        topbll_scores = np.array([score for _, score in sorted_entries])
    out = (seq_in, seq_actr_bll, seq_actr_spread,
           pos_actr_bll, pos_actr_spread,
           topbll_ids, topbll_scores, seq_pos, seq_neg)
    if item_type != 'track':
        out = out + (seqin_nitems, pos_nitems, neg_nitems)
    return out


def test_item_sample(uid, data, seqlen, nxt_idx, item_type, **kwargs):
    item_ids_map = kwargs[f'{item_type}_ids_map']
    bll_weights = kwargs[f'{item_type}_bll_weights']
    spread_weights = kwargs[f'{item_type}_spread_weights']
    num_favs = kwargs['num_favs'] if item_type == 'track' \
        else kwargs['num_favs_art']

    seq_in = np.zeros(shape=[seqlen, SESSION_LEN], dtype=np.int32)
    seq_actr_bll = np.zeros(shape=[seqlen, SESSION_LEN],
                            dtype=np.float32)
    seqin_nitems = np.zeros(shape=seqlen, dtype=np.float32)
    user_sessions = data['user_sessions'][uid]
    idx = seqlen - 1
    for sess in reversed(user_sessions[:nxt_idx]):
        if item_type == 'track':
            item_ids = list(set(sess['track_ids']))
        else:
            item_ids = list(set([kwargs['track_art_map'][tid]
                                 for tid in sess['track_ids']]))
            seqin_nitems[idx] = len(item_ids)
        seq_in_track_ids = [item_ids_map[iid] for iid in item_ids]
        seq_in[idx][:len(seq_in_track_ids)] = seq_in_track_ids
        seq_actr_bll[idx][:len(seq_in_track_ids)] = [
            bll_weights[uid][iid] if iid in bll_weights[uid] else 0.0
            for iid in item_ids]
        idx -= 1
        if idx == -1:
            break
    seq_actr_spread = spread_weights[uid][nxt_idx]
    # sort personal items by descending BLL
    sorted_entries = sorted(bll_weights[uid].items(),
                            key=lambda x: x[1], reverse=True)[:num_favs]
    if len(sorted_entries) < num_favs:
        topbll_ids = np.zeros(num_favs, dtype=np.int32)
        topbll_scores = np.zeros(num_favs, dtype=np.float32)
        topbll_ids[:len(sorted_entries)] = [item_ids_map[iid] for iid, _ in
                                            sorted_entries]
        topbll_scores[:len(sorted_entries)] = [score for _, score in
                                               sorted_entries]
    else:
        topbll_ids = np.array([item_ids_map[iid] for iid, _ in sorted_entries])
        topbll_scores = np.array([score for _, score in sorted_entries])
    out = (seq_in, seq_actr_bll, seq_actr_spread,
           topbll_ids, topbll_scores)
    if item_type != 'track':
        out = out + (seqin_nitems,)
    return out
