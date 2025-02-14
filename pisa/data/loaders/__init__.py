from pisa import PisaError
from pisa.data.loaders.train_loader import TrainDataLoader
from pisa.data.loaders.test_loader import TestDataLoader

_SUPPORTED_DATALOADERS = {
    'train': TrainDataLoader,
    'valid': TestDataLoader,
    'test': TestDataLoader
}


def dataloader_factory(data, batch_size, seqlen, mode='train',
                       random_seed=2023, **kwargs):
    kwargs['mode'] = mode
    kwargs['track_ids_map'] = data['track_ids_map']
    bll_mode = mode if mode != 'valid' else 'test'
    spread_mode = mode if mode != 'valid' else 'test'
    kwargs['track_bll_weights'] = data['track_bll_weights'][bll_mode]
    kwargs['track_spread_weights'] = data['track_spread_weights'][spread_mode]
    kwargs['user_ids_map'] = data['user_ids_map']
    if mode == 'train':
        kwargs['track_pos_spread_weights'] = data['track_pos_spread_weights']
        kwargs['train_session_indexes'] = data['train_session_indexes']

    if kwargs['model_name'] == 'pisa_art':
        kwargs['artist_bll_weights'] = data['artist_bll_weights'][bll_mode]
        kwargs['artist_spread_weights'] = data['artist_spread_weights'][
            spread_mode]
        kwargs['n_artists'] = data['n_artists']
        kwargs['artist_ids_map'] = data['art_ids_map']
        kwargs['track_art_map'] = data['track_art']
        if mode == 'train':
            kwargs['artist_pos_spread_weights'] = data[
                'artist_pos_spread_weights']
    try:
        return _SUPPORTED_DATALOADERS[mode](data,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'],
                                            batch_size=batch_size,
                                            seqlen=seqlen,
                                            random_seed=random_seed,
                                            **kwargs)
    except KeyError as err:
        raise PisaError(f'{err}')
