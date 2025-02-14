from pisa import PisaError


def one_train_sample(uid, nxt_idx, dataset, seqlen, n_items, user_tracks,
                     **kwargs):
    model_name = kwargs['model_name']
    if model_name == 'pisa':
        from pisa.data.samplers.sampler_pisa import train_sample
    elif model_name == 'pisa_art':
        from pisa.data.samplers.sampler_pisa_art import train_sample
    else:
        raise PisaError(f'Not support train sampler for '
                        f'{kwargs["model_name"]} model')
    return train_sample(uid, nxt_idx, dataset, seqlen, n_items, user_tracks,
                        **kwargs)


def one_test_sample(uid, data, seqlen, nxt_idx, **kwargs):
    model_name = kwargs['model_name']
    if model_name == 'pisa':
        from pisa.data.samplers.sampler_pisa import test_sample
    elif model_name == 'pisa_art':
        from pisa.data.samplers.sampler_pisa_art import test_sample
    else:
        raise PisaError(f'Not support test sampler for '
                        f'{kwargs["model_name"]} model')
    return test_sample(uid, data, seqlen, nxt_idx, **kwargs)
