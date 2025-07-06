"""
data_generator.py

This module handles the creation of data generators for efficient data loading and preprocessing during training, including LRswap augmentation.

Class:
    DataGenerator: A data generator for efficient data loading and preprocessing during training.

Methods:
    __init__(self, params=None, mode='dev_train'): Initializes the DataGenerator instance.
    __getitem__(self, item): Returns the data for a given index.
    __len__(self): Returns the number of data points.
    get_feature_files(self): Collects the paths to the feature files based on the selected folds and modality.
    get_folds(self): Returns the folds for the given data split.

Author: Gavin
Date: June 2025
"""

import os
import torch
import glob
from torch.utils.data.dataset import Dataset
import numpy as np
import utils
import torch




    
class DataGenerator(Dataset):
    def __init__(self, params=None, mode='dev_train'):
        """
        Initializes the DataGenerator instance.
        Args:
            params (dict): Parameters for data generation. If None, it will try to import from parameters.
            mode (str): data split ('dev_train', 'dev_test').
        """
        # If params are not provided, try to import from parameters
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
            
        self.mode = mode
        self.root_dir = self.params['root_dir']
        self.feat_dir = self.params['feat_dir']
        self.modality = self.params['modality']

        self.folds = self.get_folds()

        # self.video_files will be an empty [] if self.modality == 'audio'
        self.audio_files, self.video_files, self.label_files = self.get_feature_files()

    def __getitem__(self, item):
        """
        Returns the data for a given index.
        Args:
            item (int): Index of the data.
        Returns:
            tuple: A tuple containing audio features, video_features (for audio_visual modality), and labels.
        """
        audio_file = self.audio_files[item]
        label_file = self.label_files[item]
        
        # Load data
        audio_features = torch.load(audio_file, map_location='cpu', weights_only=False)
        labels = torch.load(label_file, map_location='cpu', weights_only=False)
        
        # Data validation
        if torch.isnan(audio_features).any() or torch.isinf(audio_features).any():
            raise ValueError(f"Audio features contain NaN or Inf values: {audio_file}")
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels)
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            raise ValueError(f"Labels contain NaN or Inf values: {label_file}")
            
        # If it's an augmented version, transform labels
        if '_LRswap.pt' in audio_file:
            labels = self.swap_label_channels(labels)
        
        # Original label processing logic
        if not self.params['multiACCDOA']:
            mask = labels[:, :self.params['nb_classes']]
            mask = mask.repeat(1, 4)
            labels = mask * labels[:, self.params['nb_classes']:]
        else:
            if self.params['multiACCDOA']:
                labels = labels[:, :, :-1, :]
            else:
                labels = labels[:, :-self.params['nb_classes']]
        
        # Final data validation
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            raise ValueError(f"Processed labels contain NaN or Inf values: {label_file}")
            
        return audio_features, labels

    def __len__(self):
        """
        Returns the number of data points.
        Returns:
            int: Number of data points.
        """
        #return 100
        return len(self.audio_files)

    def get_feature_files(self):
        """
        Get feature files, including augmented versions during training, only original versions during evaluation
        """
        audio_files, video_files, label_files = [], [], []
        label_file_map = {}  # Used to store label file paths

        for fold in self.folds:
            # Get label files
            label_path = os.path.join(self.feat_dir, f'metadata_dev{"_adpit" if self.params["multiACCDOA"] else ""}/{fold}*.pt')
            fold_label_files = glob.glob(label_path)
            
            # Get feature files
            audio_path = os.path.join(self.feat_dir, f'stereo_dev/{fold}*.pt')
            fold_audio_files = glob.glob(audio_path)
            
            # Find corresponding feature files for each label file
            for label_file in fold_label_files:
                base_name = os.path.basename(label_file)  # Keep .pt suffix
                audio_file = os.path.join(os.path.dirname(audio_path), base_name)
                
                # Check if original audio file exists
                if os.path.exists(audio_file):
                    # Store label file path
                    label_file_map[audio_file] = label_file
                    audio_files.append(audio_file)
                    
                    # If training mode, also add LRswap version
                    if self.mode == 'dev_train':
                        lrswap_file = audio_file.replace('.pt', '_LRswap.pt')
                        if os.path.exists(lrswap_file):
                            label_file_map[lrswap_file] = label_file  # Use same label file
                            audio_files.append(lrswap_file)

        # Get corresponding label files in audio file order
        label_files = [label_file_map[audio_file] for audio_file in audio_files]
        
        print(f"Mode: {self.mode}")
        print(f"Total files:")
        print(f"  - Unique label files: {len(set(label_files))}")
        print(f"  - Audio files: {len(audio_files)}")
        if self.mode == 'dev_train':
            print(f"  - Including augmented versions")
        else:
            print(f"  - Only original versions")
        
        return audio_files, [], label_files

    def get_folds(self):
        """
        Returns the folds for the given data split
        Returns:
            list: List of folds.
        """
        if self.mode == 'dev_train':
            return self.params['dev_train_folds']  # fold 1, fold 3
        elif self.mode == 'dev_test':
            return self.params['dev_test_folds']  # fold 4
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from ['dev_train', 'dev_test'].")

    def swap_label_channels(self, labels):
        """
        Transform labels for LRswap augmentation
        Args:
            labels: shape [50, 6, 5, 13]
        Returns:
            Transformed labels, same shape
        """
        swapped_labels = labels.clone()
        # When left and right channels are swapped, source positions should be symmetric about y-axis
        # This means y coordinates should be negated, x coordinates remain unchanged
        swapped_labels[:, :, 2, :] = -swapped_labels[:, :, 2, :]  # Negate y coordinates
        return swapped_labels

    def test_random_pair(self):
        print("\n=== Random Label Transformation Test ===")
        # Get indices of all LRswap samples
        lrswap_indices = [i for i, file in enumerate(self.audio_files) if '_LRswap.pt' in file]
        if not lrswap_indices:
            print("No LRswap samples found")
            return
        
        # Randomly select one LRswap sample
        random_idx = np.random.choice(lrswap_indices)
        # Find corresponding original sample
        original_idx = random_idx - 1
        
        if original_idx >= 0 and '_LRswap.pt' not in self.audio_files[original_idx]:
            print(f"\nRandomly selected sample pair:")
            print(f"Original audio: {self.audio_files[original_idx]}")
            print(f"LRswap audio: {self.audio_files[random_idx]}")
            print(f"Label file: {self.label_files[random_idx]}")
            
            # Load both samples
            original_audio, original_label = self[original_idx]
            lrswap_audio, lrswap_label = self[random_idx]
            
            # Check non-zero coordinates
            non_zero_x = torch.nonzero(original_label[:, :, 1, :])
            if len(non_zero_x) > 0:
                print("\nCoordinate comparison:")
                # Randomly select 3 non-zero coordinate points
                selected_indices = np.random.choice(len(non_zero_x), min(3, len(non_zero_x)), replace=False)
                for idx in selected_indices:
                    pos = non_zero_x[idx]
                    t, s, c = pos
                    print(f"\nTime {t}, Source {s}, Class {c}:")
                    print(f"Original sample:")
                    print(f"  x: {original_label[t, s, 1, c]:.4f}")
                    print(f"  y: {original_label[t, s, 2, c]:.4f}")
                    print(f" Azimuth: {torch.atan2(original_label[t, s, 2, c], original_label[t, s, 1, c]):.4f}")
                    
                    print(f"LRswap sample:")
                    print(f"  x: {lrswap_label[t, s, 1, c]:.4f}")
                    print(f"  y: {lrswap_label[t, s, 2, c]:.4f}")
                    print(f" Azimuth: {torch.atan2(lrswap_label[t, s, 2, c], lrswap_label[t, s, 1, c]):.4f}")
                    
                    # Verify if azimuth is negated
                    original_azimuth = torch.atan2(original_label[t, s, 2, c], original_label[t, s, 1, c])
                    swapped_azimuth = torch.atan2(lrswap_label[t, s, 2, c], lrswap_label[t, s, 1, c])
                    print(f"Azimuth difference: {abs(original_azimuth + swapped_azimuth):.4f} (should be close to 0)")
            else:
                print("No non-zero coordinates found")
        else:
            print("No corresponding original sample found")

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    try:
        from experiments.baseline_config import get_params
        test_params = get_params()
    except ImportError:
        print("Warning: experiments/baseline_config.py not found, please ensure configuration file exists")
        exit(1)
    # Test training set
    print("\n=== Testing Training Set ===")
    test_params['multiACCDOA'] = True
    dev_train_dataset = DataGenerator(params=test_params, mode='dev_train')
    
    # Test label transformation
    # print("\n=== Testing Label Transformation ===")
    # # Find a pair of original and LRswap samples
    # for i in range(len(dev_train_dataset.audio_files)):
    #     if '_LRswap.pt' in dev_train_dataset.audio_files[i]:
    #         # Find corresponding original sample
    #         original_idx = i - 1  # The previous one should be the original sample
    #         if original_idx >= 0 and '_LRswap.pt' not in dev_train_dataset.audio_files[original_idx]:
    #             print(f"\nFound a pair of samples:")
    #             print(f"Original audio: {dev_train_dataset.audio_files[original_idx]}")
    #             print(f"LRswap audio: {dev_train_dataset.audio_files[i]}")
    #             print(f"Label file: {dev_train_dataset.label_files[i]}")
                
    #             # Load both samples
    #             original_audio, original_label = dev_train_dataset[original_idx]
    #             lrswap_audio, lrswap_label = dev_train_dataset[i]
                
    #             # Check non-zero coordinates
    #             non_zero_x = torch.nonzero(original_label[:, :, 1, :])
    #             if len(non_zero_x) > 0:
    #                 print("\nCoordinate comparison:")
    #                 for pos in non_zero_x[:3]:  # Only print first 3
    #                     t, s, c = pos
    #                     print(f"\nTime {t}, Source {s}, Class {c}:")
    #                     print(f"Original sample:")
    #                     print(f"  x: {original_label[t, s, 1, c]:.4f}")
    #                     print(f"  y: {original_label[t, s, 2, c]:.4f}")
    #                     print(f" Azimuth: {torch.atan2(original_label[t, s, 2, c], original_label[t, s, 1, c]):.4f}")
                        
    #                     print(f"LRswap sample:")
    #                     print(f"  x: {lrswap_label[t, s, 1, c]:.4f}")
    #                     print(f"  y: {lrswap_label[t, s, 2, c]:.4f}")
    #                     print(f" Azimuth: {torch.atan2(lrswap_label[t, s, 2, c], lrswap_label[t, s, 1, c]):.4f}")
                        
    #                     # Verify if azimuth is negated
    #                     original_azimuth = torch.atan2(original_label[t, s, 2, c], original_label[t, s, 1, c])
    #                     swapped_azimuth = torch.atan2(lrswap_label[t, s, 2, c], lrswap_label[t, s, 1, c])
    #                     print(f"Azimuth difference: {abs(original_azimuth + swapped_azimuth):.4f} (should be close to 0)")
    #             else:
    #                 print("No non-zero coordinates found")
    #             break  # Only test first pair

    # Run test
    dev_train_dataset.test_random_pair()



