"""
extract_features.py

This module defines the SELDFeatureExtractor class, which provides functionality to extract features from both audio
and audio augmented with LRswap. It also processes the labels to support MultiACCDOA (ADPIT).  It includes the following key components.

Classes:
    SELDFeatureExtractor: A class that supports the extraction of audio and video features. It extracts log Mel
    spectrogram from audio files and ResNet-based features from video frames. It also processes labels for MultiACCDOA.

    Methods:
        - extract_audio_features: Extracts audio features from a specified split of the dataset.w
        - extract_features: A high-level function to extract features based on the modality ('audio' or 'audio_visual').
        - extract_labels: converts labels to support multiACCDOA.

Author: Gavin
Date: June 2025
"""

import os
import glob
import torch
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import utils
import numpy as np

class SELDFeatureExtractor():
    def __init__(self, params=None):
        """
        Initializes the SELDFeatureExtractor with the provided parameters.
        Args:
            params (dict): A dictionary containing various parameters for audio/video feature extraction among others.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # If no parameters are provided, try to import from parameters
        if params is None:
            try:
                from parameters import params as default_params
                self.params = default_params
            except ImportError:
                raise ImportError("Parameter configuration not found. Please ensure:\n"
                                "1. Provide parameters through configuration files in the experiments directory, or\n"
                                "2. A parameters.py file exists in the root directory as default configuration")
        else:
            self.params = params
        self.root_dir = params['root_dir']
        self.feat_dir = params['feat_dir']

        self.modality = params['modality']

        # audio feature extraction
        self.sampling_rate = params['sampling_rate']
        self.hop_length = int(self.sampling_rate * params['hop_length_s'])
        self.win_length = 2 * self.hop_length
        self.n_fft = 2 ** (self.win_length - 1).bit_length()
        self.nb_mels = params['nb_mels']

        # label extraction
        self.nb_label_frames = params['label_sequence_length']
        self.nb_unique_classes = params['nb_classes']


    @staticmethod
    def normalize_audio(audio):
        """
        Peak normalization, scale audio to [-1, 1] range
        Args:
            audio: numpy.ndarray, shape=[channels, samples]
        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    def extract_audio_features(self, split):
        """
        Extracts audio features for a given split (dev/eval).
        Args:
            split (str): The split for which features need to be extracted ('dev' or 'eval').
        """

        if split == 'dev':
            audio_files = glob.glob(os.path.join(self.root_dir, 'stereo_dev', 'dev-*', '*.wav'))
        elif split == 'eval':
            audio_files = glob.glob(os.path.join(self.root_dir, 'stereo_eval', 'eval', '*.wav'))
        else:
            raise ValueError("Split must be either 'dev' or 'eval'.")

        os.makedirs(os.path.join(self.feat_dir, f'stereo_{split}'), exist_ok=True)

        for audio_file in tqdm(audio_files, desc=f"Processing audio files ({split})", unit="file"):
            basename = os.path.splitext(os.path.basename(audio_file))[0]
            
            # 1. Process original audio
            original_feature_path = os.path.join(self.feat_dir, f'stereo_{split}', f'{basename}.pt')
            if not os.path.exists(original_feature_path):
                # Load original audio
                audio, sr = utils.load_audio(audio_file, self.sampling_rate)
                audio = self.normalize_audio(audio)
                # Extract features
                if self.params['feature_type'] == 'PFOA':
                    audio_feat = utils.extract_PFOA(audio, sr, self.n_fft, self.hop_length, self.win_length, self.nb_mels)
                    # extract_PFOA already returns torch.Tensor, use directly
                    audio_feat = audio_feat.detach().float()
                torch.save(audio_feat, original_feature_path)

            # 2. Process augmented version (channel swapping)
            aug_feature_path = os.path.join(self.feat_dir, f'stereo_{split}', f'{basename}_LRswap.pt')
            if not os.path.exists(aug_feature_path):
                # Load audio
                audio, sr = utils.load_audio(audio_file, self.sampling_rate)
                audio = self.normalize_audio(audio) 
                # Swap channels
                audio_aug = audio.copy()
                temp = audio_aug[0].copy()  # Use copy() to ensure deep copy
                audio_aug[0] = audio_aug[1].copy()  # Use copy() to ensure deep copy
                audio_aug[1] = temp
                # Extract features
                if self.params['feature_type'] == 'PFOA':
                    audio_feat = utils.extract_PFOA(audio_aug, sr, self.n_fft, self.hop_length, self.win_length, self.nb_mels)
                    # extract_PFOA already returns torch.Tensor, use directly
                    audio_feat = audio_feat.detach().float()
                torch.save(audio_feat, aug_feature_path)


    def extract_features(self, split='dev'):
        """
        Extracts features based on the selected modality ('audio' or 'audio_visual').
        Args:
            split (str): The split for which features need to be extracted ('dev' or 'eval').
        """

        os.makedirs(self.feat_dir, exist_ok=True)

        if self.modality == 'audio':
            self.extract_audio_features(split)
        else:
            raise ValueError("Modality should be one of 'audio' or 'audio_visual'. You can set the modality in params.py")

    def extract_labels(self, split):

        os.makedirs(self.feat_dir, exist_ok=True)   # already created by extract_features method

        if split == 'dev':
            label_files = glob.glob(os.path.join(self.root_dir, 'metadata_dev', 'dev-*', '*.csv'))
        elif split == 'eval':  # only for organizers
            label_files = glob.glob(os.path.join(self.root_dir, 'metadata_eval', 'eval', '*.csv'))
        else:
            raise ValueError("Split must be either 'dev' or 'eval'.")

        os.makedirs(os.path.join(self.feat_dir, 'metadata_{}{}'.format(split, '_adpit' if self.params['multiACCDOA'] else '')), exist_ok=True)

        for label_file in tqdm(label_files, desc=f"Processing label files ({split})", unit="file"):
            filename = os.path.splitext(os.path.basename(label_file))[0] + '.pt'
            label_path = os.path.join(self.feat_dir, 'metadata_{}{}'.format(split, '_adpit' if self.params['multiACCDOA'] else ''), filename)

            # Check if the feature file already exists
            if os.path.exists(label_path):
                continue

            # If the feature file doesn't exist, perform extraction
            label_data = utils.load_labels(label_file)
            if self.params['multiACCDOA']:
                processed_labels = utils.process_labels_adpit(label_data, self.nb_label_frames, self.nb_unique_classes)
            else:
                processed_labels = utils.process_labels(label_data, self.nb_label_frames, self.nb_unique_classes)
            torch.save(processed_labels, label_path)


if __name__ == '__main__':
    # use this space to test if the SELDFeatureExtractor class works as expected.
    # All the classes will be called from the main.py for actual use.
    try:
        from experiments.EXP_HTSAT import get_params
        test_params = get_params()
    except ImportError:
        print("Warning: experiments/baseline_config.py not found, please ensure configuration file exists")
        exit(1)
        
    test_params['multiACCDOA'] = True
    feature_extractor = SELDFeatureExtractor(test_params)
    feature_extractor.extract_features(split='dev')
    feature_extractor.extract_labels(split='dev')


