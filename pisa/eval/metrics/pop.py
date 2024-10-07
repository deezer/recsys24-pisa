import numpy as np
from pisa.eval.metrics.metric import Metric


class POP(Metric):
    """
    Median rank of recommended items @k score metric.
    """
    def __str__(self):
        return f'pop@{self.k}'

    def eval(self, reco_items, ref_user_items, mode='mean'):
        res = []
        pop_dict = self.kwargs['track_popularities']
        stop_sess = 1 if mode == 'first' else None
        for user_id, top_items_arr in reco_items.items():
            user_pops = [np.mean([pop_dict[it] for it in top_items[:self.k]])
                         for top_items in top_items_arr[:stop_sess]]
            res.append(np.mean(user_pops))
        return np.mean(res)
