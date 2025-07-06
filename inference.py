
"""
inference.py

This module provides functionality for SELD (Sound Event Localization and Detection) model inference.
It loads trained model weights and performs inference on test data to evaluate model performance.

Functions:
    get_exp_params: Dynamically imports and returns experiment parameters from configuration files.
    cleanup: Cleans up GPU memory and system resources.
    main: Main inference function that handles model loading, data preparation, and evaluation.

Key Features:
    - Supports multiple model types (HTSAT, CNN14_Conformer)
    - Handles both single-ACCDOA and multi-ACCDOA configurations
    - Provides comprehensive evaluation metrics including F1 score, angular error, distance error
    - Supports jackknife analysis for robust statistical evaluation
    - Includes resource cleanup and error handling

Usage:
    python inference.py --exp baseline_config --checkpoints_dir ./checkpoints --output_dir ./outputs

Author: Gavin
Date: June 2025
"""

import os
import torch
import gc
import utils
import argparse
import importlib
import sys
from model import *  # Or your actual model name
from main import val_epoch  #
from loss import SELDLossADPIT, SELDLossSingleACCDOA
from metrics import ComputeSELDResults
from data_generator_aug import DataGenerator
from torch.utils.data import DataLoader

def get_exp_params(exp_name):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    exp_module = importlib.import_module(f'experiments.{exp_name}')
    return exp_module.get_params()

def cleanup():
    """Clean up resources"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main(params):
    try:
        # 1. Set device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # 2. Build model
        if params['model_type'] == 'HTSAT':
            seld_model = HTSAT_multi(params=params).to(device)
        elif params['model_type'] == 'CNN14_Conformer':
            seld_model = ConvConformer_Multi(params=params).to(device)
        else:
            raise ValueError(f"Unsupported model type: {params['model_type']}")

        # 3. Load best weights
        checkpoints_folder = params['checkpoints_dir']
        print(f"Loading model weights: {checkpoints_folder}")
        best_model_ckpt = torch.load(os.path.join(checkpoints_folder, 'best_model.pth'), 
                                   map_location=device, 
                                   weights_only=False)  # Only load weights
        seld_model.load_state_dict(best_model_ckpt['seld_model'])
        seld_model.eval()  # Set to evaluation mode

        # 4. Build dataset and loss, evaluation functions
        print("Preparing dataset...")
        dev_test_dataset = DataGenerator(params=params, mode='dev_test')
        dev_test_iterator = DataLoader(
            dataset=dev_test_dataset,
            batch_size=params['batch_size'],
            num_workers=0,  # Simplified worker configuration
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )

        if params['multiACCDOA']:
            seld_loss = SELDLossADPIT(params=params).to(device)
        else:
            seld_loss = SELDLossSingleACCDOA(params=params).to(device)

        seld_metrics = ComputeSELDResults(
            params=params, 
            ref_files_folder=os.path.join(params['root_dir'], 'metadata_dev')
        )

        use_jackknife = params['use_jackknife']
        output_dir = params['output_dir']

        print("Starting inference...")
        with torch.no_grad():  # Use no_grad context
            test_loss, test_metric_scores = val_epoch(
                seld_model, dev_test_iterator, seld_loss, seld_metrics, 
                output_dir, device, params, is_jackknife=use_jackknife
            )
        
        test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr = test_metric_scores
        utils.print_results(test_f, test_ang_error, test_dist_error, test_rel_dist_error, 
                          test_onscreen_acc, class_wise_scr, params)
        
        print("Inference completed!")

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise e
    finally:
        cleanup()
        print("Resource cleanup completed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SELD inference script')
    parser.add_argument('--exp', type=str, required=True, help='Experiment configuration filename (without .py)')
    parser.add_argument('--checkpoints_dir', type=str, help='Model checkpoint directory')
    parser.add_argument('--feat_dir', type=str, help='Feature directory')
    parser.add_argument('--root_dir', type=str, help='Dataset directory')
    parser.add_argument('--output_dir', type=str, help='Inference output directory')
    args = parser.parse_args()
    params = get_exp_params(args.exp)
    if args.checkpoints_dir:
        params['checkpoints_dir'] = args.checkpoints_dir
    if args.feat_dir:
        params['feat_dir'] = args.feat_dir
    if args.root_dir:
        params['root_dir'] = args.root_dir
    if args.output_dir:
        params['output_dir'] = args.output_dir
    main(params)