from .baseline_config import get_baseline_params

def get_params():
    params = get_baseline_params()
    
    params.update({
        'experiment_name': 'CNN14_BiMamba',
        'model_type': 'CNN14_Conformer',
        'feature_type': 'PFOA',

        'decoder_type': 'bmamba',
        'num_decoder_layers': 2,

        'learning_rate': 1e-4,
        'weight_decay': 5e-6,
        'batch_size': 128,
        'nb_epochs': 120,
        'nb_workers': 14,
        
        'early_stopping_patience': 30,
    })
    
    params['decoder_kwargs'] = {
        'd_state': 64,
        'd_conv': 4,
        'expand': 2,
        'dropout': 0.05,
        'bias': True,
        'conv_bias': True,
    }
    
    return params

config = get_params() 