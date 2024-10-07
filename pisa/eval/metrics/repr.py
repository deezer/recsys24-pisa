import numpy as np
from pisa.eval.metrics.metric import Metric


class REPR(Metric):
    """
    Repetition ratio @k metric.
    """
    def __str__(self):
        return f'repr@{self.k}'

    def eval(self, reco_items, ref_user_items, mode='mean'):
        res = []
        fav_tracks = self.kwargs['user_tracks']
        stop_sess = 1 if mode == 'first' else None
        for uid, top_items_arr in reco_items.items():
            rep_hits = [np.array([1. if it in fav_tracks[uid]
                                  else 0. for it in top_items[:self.k]],
                                 dtype=np.float32)
                        for idx, top_items in enumerate(top_items_arr[:stop_sess])]
            rep_hits = np.mean([np.sum(hits) / self.k for hits in rep_hits])
            res.append(rep_hits)
        repeat_ratio = np.mean(res) if len(res) > 0 else 0.
        return repeat_ratio
