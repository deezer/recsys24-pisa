class Metric:
    """
    Abstract class for a recommendation metric
    """
    def __init__(self, k, **kwargs):
        self.k = k
        self.kwargs = kwargs

    def eval(self, reco_items, ref_user_items, mode='mean'):
        """
        Abstract
        :param reco_items:
        :param ref_user_items:
        :param mode
        :return:
        """
        raise NotImplementedError(
            'eval method should be implemented in concrete model')
