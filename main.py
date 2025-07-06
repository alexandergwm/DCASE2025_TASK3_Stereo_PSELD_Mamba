"""
main.py

This is the main entry point for the SELD (Sound Event Localization and Detection) training pipeline.
It orchestrates the complete training workflow including data preparation, model training, validation,
and evaluation with support for data augmentation.

Key Components:
    EarlyStopping: Prevents overfitting by monitoring validation performance
    train_epoch: Handles one epoch of model training with optional mixed precision
    val_epoch: Performs validation and generates predictions for evaluation
    main: Main training function that coordinates the entire pipeline

Features:
    - Supports multiple model architectures (HTSAT, CNN14_Conformer)
    - Handles both single-ACCDOA and multi-ACCDOA configurations
    - Includes data augmentation with LRswap (left-right channel swapping)
    - Supports mixed precision training for improved performance
    - Implements early stopping and learning rate scheduling
    - Comprehensive evaluation metrics with optional jackknife analysis
    - Checkpoint saving and resuming capabilities

Functions:
    seed_everything: Sets random seeds for reproducibility
    train_epoch: Executes one training epoch with progress tracking
    val_epoch: Performs validation and metric computation
    main: Main training loop with model setup and training orchestration

Usage:
    python main.py
    Or import and call main() with custom parameters

Author: Gavin
Date: June 2025
"""

import os
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from model import SELDModel, HTSAT_multi, ConvConformer_Multi
from loss import SELDLossADPIT, SELDLossSingleACCDOA
from metrics import ComputeSELDResults
from data_generator_aug import DataGenerator
from torch.utils.data import DataLoader
from extract_features_aug import SELDFeatureExtractor
import utils
import pickle
from tqdm import tqdm
from torch.amp import GradScaler


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, mode='max'):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped
            min_delta (float): Minimum improvement threshold
            mode (str): 'min' or 'max', indicates whether the monitored metric should be minimized or maximized
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf if mode == 'min' else -np.inf

    def __call__(self, val_score):
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def seed_everything(seed):
    """
    Set all random seeds to ensure reproducibility of experiments
    
    Args:
        seed (int): Random seed value
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

def train_epoch(seld_model, dev_train_iterator, optimizer, seld_loss, device, params, scaler=None):
    seld_model.train()
    train_loss_per_epoch = 0

    pbar = tqdm(dev_train_iterator, desc='Training', leave=True)
    for i, (input_features, labels) in enumerate(pbar):
        optimizer.zero_grad()
        labels = labels.to(device)
        # Handling modalities
        if params['modality'] == 'audio':
            audio_features, video_features = input_features.to(device), None
        elif params['modality'] == 'audio_visual':
            audio_features, video_features = input_features[0].to(device), input_features[1].to(device)
        else:
            raise AssertionError("Modality should be one of 'audio' or 'audio_visual'.")

        # Forward pass
        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = seld_model(audio_features, video_features)
                loss = seld_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = seld_model(audio_features, video_features)
            loss = seld_loss(logits, labels)
            loss.backward()
            optimizer.step()

        # Track loss
        train_loss_per_epoch += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = train_loss_per_epoch / len(dev_train_iterator)
    return avg_train_loss

def val_epoch(seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir, device, params, is_jackknife=False):
    seld_model.eval()
    val_loss_per_epoch = 0
    
    pbar = tqdm(dev_test_iterator, desc='Validation', leave=True)
    with torch.no_grad():
        for j, (input_features, labels) in enumerate(pbar):
            labels = labels.to(device)

            # Handling modalities
            if params['modality'] == 'audio':
                audio_features, video_features = input_features.to(device), None
            elif params['modality'] == 'audio_visual':
                audio_features, video_features = input_features[0].to(device), input_features[1].to(device)
            else:
                raise AssertionError("Modality should be one of 'audio' or 'audio_visual'.")

            # Forward pass
            logits = seld_model(audio_features, video_features)

            # Compute loss
            loss = seld_loss(logits, labels)
            val_loss_per_epoch += loss.item()

            # save predictions to csv files for metric calculations
            utils.write_logits_to_dcase_format(logits, params, output_dir, dev_test_iterator.dataset.label_files[j * params['batch_size']: (j + 1) * params['batch_size']])
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_val_loss = val_loss_per_epoch / len(dev_test_iterator)

        metric_scores = seld_metrics.get_SELD_Results(pred_files_path=os.path.join(output_dir, 'dev-test'), is_jackknife=is_jackknife)

        return avg_val_loss, metric_scores

def main(params=None, restore_from_checkpoint=False, initial_checkpoint_path=None):
    """
    Main function
    
    Args:
        params (dict): Experiment parameters dictionary, uses default parameters if None
        restore_from_checkpoint (bool): Whether to restore from checkpoint
        initial_checkpoint_path (str): Checkpoint path
    """
    # If no parameters are provided, try to import default parameters from parameters
    if params is None:
        try:
            from parameters import params as default_params
            params = default_params
        except ImportError:
            raise ImportError("Parameter configuration not found. Please ensure:\n"
                            "1. Provide parameters through configuration files in the experiments directory, or\n"
                            "2. A parameters.py file exists in the root directory as default configuration")
    
    # Set random seed to ensure experiment reproducibility
    seed = params.get('random_seed', 42)
    seed_everything(seed)
    
    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    early_stopping = EarlyStopping(
        patience=params.get('early_stopping_patience', 10),
        min_delta=params.get('early_stopping_min_delta', 0.001),
        mode='max'
    )


    if restore_from_checkpoint:
        print('Loading params from the initial checkpoint')
        params_file = os.path.join(initial_checkpoint_path, 'config.pkl')
        f = open(params_file, "rb")
        loaded_params = pickle.load(f)
        params.clear()  # Clear the original params
        params.update(loaded_params)

    # Set up directories for storing model checkpoints, predictions(output_dir), and create a summary writer
    checkpoints_folder, output_dir, summary_writer = utils.setup(params)

    # Feature extraction code.
    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='dev')
    feature_extractor.extract_labels(split='dev')

    # Set up dev_train and dev_test data iterator
    dev_train_dataset = DataGenerator(params=params, mode='dev_train')
    dev_train_iterator = DataLoader(dataset=dev_train_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'], shuffle=params['shuffle'], pin_memory=True, prefetch_factor=2, persistent_workers= True, drop_last=True)

    dev_test_dataset = DataGenerator(params=params, mode='dev_test')
    dev_test_iterator = DataLoader(dataset=dev_test_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'], shuffle=False, pin_memory=True, prefetch_factor=2, persistent_workers= True, drop_last=False)

    # create model, optimizer, loss and metrics
    # seld_model = SELDModel(params=params).to(device)
    if params['model_type'] == 'HTSAT':
        seld_model = HTSAT_multi(params=params).to(device)
    elif params['model_type'] == 'CNN14_Conformer':
        seld_model = ConvConformer_Multi(params=params).to(device)
    
    optimizer = torch.optim.AdamW(
        params=seld_model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
        betas=(0.9, 0.999),  # AdamW default values
        eps=1e-8,           # AdamW default values
    )
    scaler = GradScaler() if params.get('mixed_precision', 0) == 1 else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=0.8, 
            patience=6
            )
    if params['multiACCDOA']:
        seld_loss = SELDLossADPIT(params=params).to(device)
    else:
        seld_loss = SELDLossSingleACCDOA(params=params).to(device)

    seld_metrics = ComputeSELDResults(params=params, ref_files_folder=os.path.join(params['root_dir'], 'metadata_dev'))

    start_epoch = 0
    best_f_score = float('-inf')

    if restore_from_checkpoint:
        print('Loading model weights and optimizer state dict from initial checkpoint...')
        model_ckpt = torch.load(os.path.join(initial_checkpoint_path, 'best_model.pth'), map_location=device, weights_only=False)
        seld_model.load_state_dict(model_ckpt['seld_model'])
        optimizer.load_state_dict(model_ckpt['opt'])
        start_epoch = model_ckpt['epoch'] + 1
        best_f_score = model_ckpt['best_f_score']

    for epoch in range(start_epoch, params['nb_epochs']):
        # ------------- Training -------------- #
        avg_train_loss = train_epoch(seld_model, dev_train_iterator, optimizer, seld_loss, device, params, scaler=scaler)
        # -------------  Validation -------------- #
        avg_val_loss, metric_scores = val_epoch(seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir, device, params)
        val_f, val_ang_error, val_dist_error, val_rel_dist_error, val_onscreen_acc, class_wise_scr = metric_scores
        scheduler.step(val_f)  # Use validation F-score to adjust learning rate

        # Check if early stopping is needed
        early_stopping(val_f)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        # ------------- Log losses and metrics ------------- #
        print(
            f"Epoch {epoch + 1}/{params['nb_epochs']} | "
            f"Train Loss: {avg_train_loss:.2f} | "
            f"Val Loss: {avg_val_loss:.2f} | "
            f"F-score: {val_f * 100:.2f} | "
            f"Ang Err: {val_ang_error:.2f} | "
            f"Dist Err: {val_dist_error:.2f} | "
            f"Rel Dist Err: {val_rel_dist_error:.2f}" +
            (f" | On-Screen Acc: {val_onscreen_acc:.2f}" if params['modality'] == 'audio_visual' else "")
        )
        # ------------- Save model if validation f score improves -------------#
        if val_f >= best_f_score:
            best_f_score = val_f
            net_save = {'seld_model': seld_model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch,
                        'best_f_score': best_f_score, 'best_ang_err': val_ang_error, 'best_rel_dist_err': val_rel_dist_error}
            if params['modality'] == 'audio_visual':
                net_save['best_onscreen_acc'] = val_onscreen_acc
            torch.save(net_save, checkpoints_folder + "/best_model.pth")

    # Evaluate the best model on dev-test.
    best_model_ckpt = torch.load(os.path.join(checkpoints_folder, 'best_model.pth'), map_location=device, weights_only=False)
    seld_model.load_state_dict(best_model_ckpt['seld_model'])
    use_jackknife = params['use_jackknife']
    test_loss, test_metric_scores = val_epoch(
        seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir, device, params, is_jackknife=use_jackknife
    )
    test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr = test_metric_scores
    utils.print_results(test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr, params)


if __name__ == '__main__':
    # Test code
    try:
        from parameters import params as test_params
    except ImportError:
        print("Warning: parameters.py not found, please provide parameters through configuration files in the experiments directory")
        exit(1)
        
    main(test_params)



