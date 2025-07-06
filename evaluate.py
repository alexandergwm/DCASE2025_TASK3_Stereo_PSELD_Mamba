"""
evaluate.py

This script evaluates the trained SELD models on the DCASE2025 evaluation dataset.
It loads pretrained models, extracts features, and generates predictions in DCASE format.

Classes:
    EvalDataGenerator: Dataset class for loading evaluation data

Functions:
    evaluate: Main evaluation function that performs model inference

Key Features:
    - Feature extraction for evaluation data
    - Model loading from checkpoints
    - Prediction generation in DCASE format
    - Support for both audio-only and audio-visual modalities

Author: Gavin
Date: June 2025
"""

import utils
from model import SELDModel
import pickle
import os
import torch
from torch.utils.data import DataLoader
from extract_features import SELDFeatureExtractor
from torch.utils.data import Dataset
import glob
from model import *


class EvalDataGenerator(Dataset):
    def __init__(self, params):
        """
        Initializes the EvalDataGenerator instance.
        Args:
            params (dict): Parameters for data generation.
        """
        super().__init__()
        self.params = params
        self.root_dir = params['root_dir']
        self.feat_dir = params['feat_dir']
        self.modality = params['modality']

        # self.video_files will be an empty [] if self.modality == 'audio'
        self.audio_files, self.video_files = self.get_feature_files()

    def __getitem__(self, item):
        """
        Returns the data for a given index.
        Args:
            item (int): Index of the data.
        Returns:
            tuple: A tuple containing audio features, video_features (for audio_visual modality).
        """
        audio_file = self.audio_files[item]
        audio_features = torch.load(audio_file)

        if self.modality == 'audio_visual':
            video_file = self.video_files[item]
            video_features = torch.load(video_file)
            return audio_features, video_features
        else:
            return audio_features

    def __len__(self):
        """
        Returns the number of data points.
        Returns:
            int: Number of data points.
        """
        return len(self.audio_files)

    def get_feature_files(self):
        """
        Collects the paths to the feature files based on the modality.
        Returns:
            tuple: A tuple containing lists of paths to audio feature files, video feature files.
        """
        audio_files, video_files = [], []

        # Loop through each fold and collect files

        audio_files += glob.glob(os.path.join(self.feat_dir, f'stereo_eval/*.pt'))

        # Only collect video files if modality is 'audio_video'
        if self.modality == 'audio_visual':
            video_files += glob.glob(os.path.join(self.feat_dir, f'video_eval/*.pt'))

        # Sort files to ensure corresponding audio, video, and label files are in the same order
        audio_files = sorted(audio_files, key=lambda x: x.split('/')[-1])

        # Sort video files only if modality is 'audio_visual'
        if self.modality == 'audio_visual':
            video_files = sorted(video_files, key=lambda x: x.split('/')[-1])

        # Return the appropriate files based on modality
        if self.modality == 'audio':
            return audio_files, []
        elif self.modality == 'audio_visual':
            return audio_files, video_files
        else:
            raise ValueError(f"Invalid modality: {self.modality}. Choose from ['audio', 'audio_visual'].")


def evaluate(params):

    reference = model_dir.split('/')[-1]
    output_dir = os.path.join(params['output_dir'], reference)
    os.makedirs(params['output_dir'], exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Feature extraction code.
    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='eval')

    if params['model_type'] == 'HTSAT':
        seld_model = HTSAT_multi(params=params).to(device)
    elif params['model_type'] == 'CNN14_Conformer':
        seld_model = ConvConformer_Multi(params=params).to(device)
    else:
        raise ValueError(f"Unsupported model type: {params['model_type']}")

    model_ckpt = torch.load(os.path.join(model_dir, 'best_model.pth'), map_location=device, weights_only=False)
    seld_model.load_state_dict(model_ckpt['seld_model'])

    eval_dataset = EvalDataGenerator(params=params)
    eval_iterator = DataLoader(dataset=eval_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'],
                                shuffle=False, drop_last=False)

    seld_model.eval()
    with torch.no_grad():
        for j, (input_features) in enumerate(eval_iterator):

            # Handling modalities
            if params['modality'] == 'audio':
                audio_features, video_features = input_features.to(device), None
            elif params['modality'] == 'audio_visual':
                audio_features, video_features = input_features[0].to(device), input_features[1].to(device)
            else:
                raise AssertionError("Modality should be one of 'audio' or 'audio_visual'.")

            # Forward pass
            logits = seld_model(audio_features, video_features)

            # save predictions to csv files for metric calculations
            utils.write_logits_to_dcase_format(logits, params, output_dir, eval_iterator.dataset.audio_files[j * params['batch_size']: (j + 1) * params['batch_size']], split='eval')


if __name__ == '__main__':
    try:
        from experiments.HTSAT_PFOA_YSWAP import get_params
        test_params = get_params()
    except ImportError:
        print("Warning: experiments/baseline_config.py not found, please ensure configuration file exists")
        exit(1)
    model_dir = "/root/CODE/2025_my_pretrained/experiments/HTSAT_PFOA_YSWAP_35.1Fscore_seed2/checkpoints/SELDnet_audio_multiACCDOA_20250609_200220"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    evaluate(test_params)
