import numpy as np
from pisa.eval.metrics.metric import Metric


class NDCG(Metric):
    """
    nDCG@k score metric.
    """
    def __str__(self):
        return f'ndcg_{self.kwargs["consumption_mode"]}@{self.k}'

    @classmethod
    def dcg_at_k(cls, r, k):
        """
        Discounted Cumulative Gain calculation method
        :param r:
        :param k:
        :return: float, DCG value
        """
        assert k >= 1
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.

    def eval(self, reco_items, ref_user_items, mode='mean'):
        res = []
        consumption_mode = self.kwargs['consumption_mode']
        fav_tracks = self.kwargs['user_tracks'] \
            if consumption_mode == 'rep' or consumption_mode == 'exp' \
            else None
        stop_sess = 1 if mode == 'first' else None
        for user_id, top_items_arr in reco_items.items():
            user_res = []
            for idx, top_items in enumerate(top_items_arr[:stop_sess]):
                if idx < len(ref_user_items[user_id]):
                    ref_set = ref_user_items[user_id][idx]
                    if consumption_mode == 'rep':
                        ref_set = [it for it in ref_set if it in fav_tracks[user_id]]
                    elif consumption_mode == 'exp':
                        ref_set = [it for it in ref_set if it not in fav_tracks[user_id]]
                    if len(ref_set) > 0:
                        user_hits = np.array([1 if it in ref_set else 0 for it in top_items[:self.k]],
                                             dtype=np.float32)
                        ideal_rels = self._ideal_rels(ref_set)
                        dcg_k = self.dcg_at_k(user_hits, self.k)
                        ideal_dcg = self.dcg_at_k(ideal_rels, self.k)
                        if ideal_dcg > 0.:
                            ndcg = dcg_k / ideal_dcg
                            user_res.append(ndcg)
            if len(user_res) > 0:
                res.append(np.mean(user_res))
        glob_ndcg = np.mean(res) if len(res) > 0 else 0.
        return glob_ndcg

    def _ideal_rels(self, ref_set):
        if len(ref_set) >= self.k:
            ideal_rels = np.ones(self.k)
        else:
            ideal_rels = np.pad(np.ones(len(ref_set)),
                                (0, self.k - len(ref_set)),
                                'constant')
        return ideal_rels
