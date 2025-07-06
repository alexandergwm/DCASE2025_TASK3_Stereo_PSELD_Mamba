from .baseline_config import get_baseline_params

def get_params():
    params = get_baseline_params()
    
    params.update({
        'experiment_name': 'HTSAT',
        'model_type':'HTSAT',
        'feature_type':'PFOA',
        'learning_rate': 1e-4,
        'weight_decay': 5e-6,
        'batch_size': 128,
        'nb_epochs': 120,
        'nb_workers': 14,

        
        # Modify early stopping parameters
        'early_stopping_patience': 30,  # Increase early stopping patience
    })
    
    return params 