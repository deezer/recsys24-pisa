import numpy as np
from pisa.eval.metrics.metric import Metric


class RECALL(Metric):
    """
    recall@k score metric.
    """
    def __str__(self):
        return f'recall_{self.kwargs["consumption_mode"]}@{self.k}'

    @classmethod
    def user_rec(cls, user_hits, ref_len, k):
        score = 0.0
        user_hits = np.asfarray(user_hits)[:k]
        sum_hits = np.sum(user_hits)

        # in the case where the list contains no hit, return score 0.0 directly
        if sum_hits == 0:
            return score
        return float(sum_hits) / ref_len

    def eval(self, reco_items, ref_user_items, mode='mean'):
        """
        Compute the Top-K recall
        :param reco_items: reco items dictionary
        :param ref_user_items:
        :param mode
        :return: recall@k
        """
        recall = []
        consumption_mode = self.kwargs['consumption_mode']
        fav_tracks = self.kwargs['user_tracks'] \
            if consumption_mode == 'rep' or consumption_mode == 'exp' \
            else None
        stop_sess = 1 if mode == 'first' else None
        for user_id, top_items_arr in reco_items.items():
            user_recall = []
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
                        user_recall.append(self.user_rec(user_hits, len(ref_set), self.k))
            if len(user_recall) > 0:
                recall.append(np.mean(user_recall))
        return np.mean(recall) if len(recall) > 0 else 0.
