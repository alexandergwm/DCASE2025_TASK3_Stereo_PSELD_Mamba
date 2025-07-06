
from .baseline_config import get_baseline_params

def get_params():
    params = get_baseline_params()
    
    params.update({
        'experiment_name': 'CNN14_Conformer',
        'model_type':'CNN14_Conformer',
        'feature_type':'PFOA', 
        'decoder_type': 'conformer',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'batch_size': 256,
        'nb_epochs': 120,
        'nb_workers': 14,
        
        'num_decoder_layers': 2,
        
        'early_stopping_patience': 30,
    })
    params['decoder_kwargs'] = {
        'feed_forward_expansion_factor': 2,
        'conv_kernel_size': 7,
        'feed_forward_dropout_p': 0.05,
        'attention_dropout_p': 0.05,
        'conv_dropout_p': 0.05,
    }
        
    return params 