"""
baseline_config.py

This is the baseline configuration file containing default parameters for all experiments.
Other experimental configurations can modify specific parameters by inheriting from this file.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: January 2025
"""

def get_baseline_params():
    """
    Return baseline parameters dictionary
    """
    return {
        # choose task
        'modality': 'audio',  # 'audio' or 'audio_visual'
        'net_type': 'SELDnet',

        # data params
        'root_dir': '/root/CODE/DCASE_2025_task3/DCASE2025_DATASET',  # parent directory containing the audio, video and labels directory
        'feat_dir': '/root/CODE/DCASE_2025_task3/2025_my_pretrained/features',  # store extracted features here
        'HTS_AT_pretrained_dir': '/root/CODE/DCASE_2025_task3/2025_my_pretrained/PSELD_pretrained_ckpts/mACCDOA-HTSAT-0.567.ckpt',
        'CNN14_Conformer_pretrained_dir': '/root/CODE/DCASE_2025_task3/2025_my_pretrained/PSELD_pretrained_ckpts/mACCDOA-CNN14-Conformer-0.582.ckpt',
        'log_dir': 'logs',  # save all logs here like loss and metrics
        'checkpoints_dir': 'checkpoints',  # save trained model checkpoints and config
        'output_dir': 'outputs',  # save the predicted files here.
        'feature_type': 'PFOA', # 'mel_IPD_ILD', 'PFOA'
        'model_type':'CNN14_Conformer', # Fix: changed from 'SELD' to 'CNN14_Conformer'
        'device': 'cuda:0',

        # audio feature extraction params
        'sampling_rate': 24000,
        'hop_length_s': 0.02,    # 480 samples
        'nb_mels': 64,

        # model params
        'nb_conv_blocks': 3,
        'nb_conv_filters': 64,
        'f_pool_size': [4, 4, 2],
        't_pool_size': [5, 1, 1],
        'dropout': 0.05,

        'rnn_size': 128,
        'nb_rnn_layers': 2,

        'nb_self_attn_layers': 2,
        'nb_attn_heads': 8,

        'nb_transformer_layers': 2,

        'nb_fnn_layers': 1,
        'fnn_size':128,

        # decoder configuration
        'decoder_type': 'conformer',  # 'conformer', 'bmamba', 'gru', 'transformer'
        'num_decoder_layers': 3,      
        'decoder_channels': 1024,
        
        # BMAMBA specific parameters
        'bmamba_d_state': 64,        # State dimension for Mamba blocks
        'bmamba_d_conv': 4,          # Convolution kernel size
        'bmamba_expand': 2,          # Expansion factor
        'bmamba_dropout': 0.1,       # Dropout rate
        'bmamba_bias': False,        # Whether to use bias in linear layers
        'bmamba_conv_bias': True,    # Whether to use bias in conv layers
        
        # Conformer specific parameters
        'conformer_num_attention_heads': 8,
        'conformer_feed_forward_expansion_factor': 4,
        'conformer_conv_expansion_factor': 2,
        'conformer_feed_forward_dropout_p': 0.1,
        'conformer_attention_dropout_p': 0.1,
        'conformer_conv_dropout_p': 0.1,
        'conformer_conv_kernel_size': 31,
        'conformer_half_step_residual': True,

        'max_polyphony': 3,   # tracks for multiaccdoa
        'nb_classes': 13,
        'label_sequence_length': 50,  # 5 seconds with 100ms frames

        # loss params
        'multiACCDOA': True,
        'thresh_unify': 15,

        # training params
        'nb_epochs': 120,
        'batch_size': 128,
        'nb_workers': 10,
        'shuffle': True,
        'random_seed':42,

        # optimizer params
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,

        # folds for training, testing
        'dev_train_folds': ['fold1', 'fold3'],  # 'fold1' is the synthetic training data. You can skip that if you do not use the synthetic data to train.
        'dev_test_folds': ['fold4'],
        
        # early stopping params
        'early_stopping_patience': 30,
        'early_stopping_min_delta': 0.001,

        # metric params
        'average': 'macro',                  # Supports 'micro': sample-wise average and 'macro': class-wise average.
        'segment_based_metrics': False,      # If True, uses segment-based metrics, else uses event-based metrics.
        'lad_doa_thresh': 20,                # DOA error threshold for computing the detection metrics.
        'lad_dist_thresh': float('inf'),     # Absolute distance error threshold for computing the detection metrics.
        'lad_reldist_thresh': float('1.0'),  # Relative distance error threshold for computing the detection metrics.
        'lad_req_onscreen': False,           # Require correct on-screen estimation when computing the detection metrics.

        'use_jackknife': False,               # If True, uses jackknife to calc results of the best model on test/eval set.
                                              # CAUTION: Too slow to use jackknife

        'repeat_times': 1,                    # Number of repetitions
        'mixed_precision': 0,                # Whether to enable mixed precision training (0:no, 1:yes)
    }

def get_bmamba_params():
    """
    Return specific parameters for BMAMBA experiment
    Inherits baseline parameters and modifies decoder-related settings
    """
    params = get_baseline_params()
    
    # Modify decoder to BMAMBA
    params.update({
        'decoder_type': 'bmamba',
        'num_decoder_layers': 3,
        'decoder_channels': 1024,
        
        # Optimized BMAMBA parameters
        'bmamba_d_state': 64,
        'bmamba_d_conv': 4,
        'bmamba_expand': 2,
        'bmamba_dropout': 0.1,
        'bmamba_bias': False,
        'bmamba_conv_bias': True,
        
        # Adjust training parameters for BMAMBA
        'learning_rate': 1e-4,  # Slightly smaller learning rate
        'batch_size': 64,       # Moderate batch size
        'weight_decay': 1e-6,   # Smaller weight decay
        
        # Output directories
        'log_dir': 'logs/CNN14_BMAMBA_PFOA',
        'checkpoints_dir': 'checkpoints/CNN14_BMAMBA_PFOA',
        'output_dir': 'outputs/CNN14_BMAMBA_PFOA',
    })
    
    return params

def get_conformer_params():
    """
    Return specific parameters for Conformer experiment
    """
    params = get_baseline_params()
    
    # Ensure using Conformer
    params.update({
        'decoder_type': 'conformer',
        'num_decoder_layers': 3,
        'decoder_channels': 1024,
        
        # Output directories
        'log_dir': 'logs/CNN14_Conformer_PFOA',
        'checkpoints_dir': 'checkpoints/CNN14_Conformer_PFOA',
        'output_dir': 'outputs/CNN14_Conformer_PFOA',
    })
    
    return params

def get_params():
    """
    Return experiment parameters, directly returns baseline parameters here
    Other experiments can inherit this function and modify specific parameters
    """
    return get_baseline_params()
