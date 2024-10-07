from pisa.data.datasets.deezer import DeezerDataset
from pisa.data.datasets.lfm1b import LFMDataset


_SUPPORTED_DATASETS = {
    'deezer': DeezerDataset,
    'lfm1b': LFMDataset
}


def dataset_factory(params):
    """
    Factory that generate dataset
    :param params:
    :return:
    """
    dataset_name = params['dataset'].get('name', 'deezer')
    try:
        dataset = _SUPPORTED_DATASETS[dataset_name](params)
        dataset.fetch_data()
        return dataset.data
    except KeyError:
        raise KeyError(f'Not support {dataset_name} dataset')
