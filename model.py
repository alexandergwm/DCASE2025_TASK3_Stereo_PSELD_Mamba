"""
model.py

This module defines the architecture of SELD (Sound Event Localization and Detection) deep learning models.
It includes multiple model architectures with support for various encoders, decoders, and pretrained backbones.

Classes:
    ConvBlock: A convolutional block for feature extraction from audio input
    CRNN: Base CNN-RNN model with configurable decoder architectures
    ConvConformer: CNN-Conformer model for SELD tasks
    ConvConformer_Multi: Multi-track ConvConformer model with parameter-based initialization
    HTSAT: HTS-AT (Hierarchical Token-Semantic Audio Transformer) model
    HTSAT_multi: Multi-track HTSAT model with parameter-based initialization
    SELDModel: Traditional SELD model combining ConvBlock, recurrent, and attention layers

Key Features:
    - Supports multiple encoder architectures (CNN8, CNN12, HTSAT)
    - Configurable decoder types (conformer, bmamba, gru, transformer)
    - Multi-track ACCDOA (Active Class-aware Cartesian Direction of Arrival) output
    - Pretrained model loading from AudioSet and DataSynthSELD
    - Flexible parameter freezing strategies for transfer learning
    - Support for both single and multi-track sound event localization

Model Architectures:
    - ConvConformer: CNN encoder + Conformer decoder
    - HTSAT: Hierarchical Token-Semantic Audio Transformer
    - SELDModel: Traditional CNN + GRU + Self-attention architecture

Author: Gavin
Date: June 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pretrained_model.components.htsat import HTSAT_Swin_Transformer
from pretrained_model.components.utils import interpolate
from pretrained_model.components.conformer import ConformerBlocks
from pretrained_model.components.backbone import CNN8, CNN12
from pretrained_model.components.model_utilities import Decoder, get_conv2d_layer

class ConvBlock(nn.Module):
    """
    Convolutional block with Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout.
    Designed for feature extraction from audio input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=(5, 4), dropout=0.05):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x


### ---------------------------------------------------###
class CRNN(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=7,
        encoder='CNN12',
        pretrained_path=None,
        audioset_pretrain=False,
        num_features=[32, 64, 128, 256],
        n_mels=64,
        sample_rate=24000,
        hoplen=480,
        decoder_type='bmamba',
        num_decoder_layers=1,
        num_freeze_layers=0,
        decoder_kwargs=None
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.label_res = 0.1
        self.interpolate_time_ratio = 2 ** 3
        self.output_frames = None
        self.pred_res = int(sample_rate / hoplen * self.label_res)

        self.scalar = nn.ModuleList([nn.BatchNorm2d(n_mels) for _ in range(in_channels)])
        if encoder == 'CNN8':
            self.convs = CNN8(in_channels, num_features)
        elif encoder == 'CNN12':
            self.convs = CNN12(in_channels, num_features)
            if pretrained_path:
                print('Loading pretrained model from {}...'.format(pretrained_path))
                self.load_ckpts(pretrained_path, audioset_pretrain)
        else:
            raise NotImplementedError(f'encoder {encoder} is not implemented')

        self.num_features = num_features

        if decoder_kwargs is None:
            decoder_kwargs = {}
        self.decoder = Decoder(decoder_type, num_features[-1], num_layers=num_decoder_layers, **decoder_kwargs)
        self.fc = nn.Linear(num_features[-1], 3 * num_classes, bias=True)
        self.xy_act = nn.Tanh()
        self.distance_act = nn.ReLU()

        self.num_freeze_layers = num_freeze_layers
        if self.num_freeze_layers > 0:
            self.freeze_encoder_layers(self.num_freeze_layers)
        print(f'Freezing first {self.num_freeze_layers} encoder layers (0 means no freezing)')
        self.print_trainable_params()
    
    def load_ckpts(self, pretrained_path, audioset_pretrain=True):
        if audioset_pretrain:
            CNN14_ckpt = torch.load(pretrained_path, map_location='cpu')['model']
            CNN14_ckpt['conv_block1.conv1.weight'] = nn.Parameter(
                CNN14_ckpt['conv_block1.conv1.weight'].repeat(1, self.in_channels, 1, 1) / self.in_channels)
            missing_keys, unexpected_keys = self.convs.load_state_dict(CNN14_ckpt, strict=False)
            assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
            for ich in range(self.in_channels):
                self.scalar[ich].weight.data.copy_(CNN14_ckpt['bn0.weight'])
                self.scalar[ich].bias.data.copy_(CNN14_ckpt['bn0.bias'])
                self.scalar[ich].running_mean.copy_(CNN14_ckpt['bn0.running_mean'])
                self.scalar[ich].running_var.copy_(CNN14_ckpt['bn0.running_var'])
                self.scalar[ich].num_batches_tracked.copy_(CNN14_ckpt['bn0.num_batches_tracked'])
        else:
            ckpt = torch.load(pretrained_path, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            for key, value in self.state_dict().items():
                if key.startswith('fc.'): print(f'Skipping {key}...')
                else: value.data.copy_(ckpt[key])

    def freeze_encoder_layers(self, num_freeze=3):
        layers = list(self.convs.children())
        print(f"Encoder has {len(layers)} layers")
        for i, layer in enumerate(layers):
            if i < num_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"Freezing encoder layer {i+1}")
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        # Print parameter information
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = trainable_params + non_trainable_params
        print(f'Trainable parameters: {trainable_params:,}')
        print(f'Non-trainable parameters: {non_trainable_params:,}')
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameter ratio: {trainable_params/total_params*100:.2f}%')

    def print_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = trainable_params + non_trainable_params
        print(f'Trainable parameters: {trainable_params:,}')
        print(f'Non-trainable parameters: {non_trainable_params:,}')
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameter ratio: {trainable_params/total_params*100:.2f}%')

    def forward(self, x):
        """
        x: waveform, (batch_size, num_channels, time_frames, mel_bins)
        """
        N, _, T, _ = x.shape

        # 1. Compute scalar
        x = x.transpose(1, 3)
        for nch in range(x.shape[-1]):
            x[..., [nch]] = self.scalar[nch](x[..., [nch]])
        x = x.transpose(1, 3)

        # 2. encoder
        x = self.convs(x)
        x = x.mean(dim=3) # (N, C, T)

        # 3. decoder
        x = x.permute(0, 2, 1) # (N, T, C)
        x = self.decoder(x)     # (N, T, C)

        # 4. interpolate
        x = interpolate(x, ratio=self.interpolate_time_ratio) # (N, T', C)

        # 5. Dynamically crop T to ensure it's divisible by pred_res
        N, T, C = x.shape
        pred_res = self.pred_res
        valid_T = int(T // pred_res) * pred_res
        if T != valid_T:
            x = x[:, :valid_T, :]
            T = valid_T
        output_frames = T // pred_res

        # 6. Reshape and aggregate
        x = x.reshape(N, output_frames, pred_res, C).mean(dim=2)  # (N, output_frames, C)

        # 7. FC and activation
        x = self.fc(x)  # (N, output_frames, 3*3*num_classes)
        B, T, _ = x.shape
        x = x.reshape(B, T, 3, 3, self.num_classes)  # (B, T, 3, 3, num_classes)
        x[..., 0:2, :] = self.xy_act(x[..., 0:2, :])      # x, y
        x[..., 2, :] = self.distance_act(x[..., 2, :])    # distance
        x = x.reshape(B, T, -1)  # (B, T, 117)

        return {
            'accdoa': x,
        }


class ConvConformer(CRNN):
    def __init__(
        self,
        num_classes = 13,
        in_channels=7,
        encoder='CNN12',
        audioset_pretrain=False,
        num_features=[64, 128, 256, 512, 1024, 2048],
        n_mels=64,
        sample_rate=24000,
        hoplen=480,
        num_decoder_layers=2,
        pretrained_path=None,
        decoder_type='conformer',
        num_freeze_layers=0,
        decoder_kwargs=None
    ):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            encoder=encoder,
            pretrained_path=pretrained_path,
            audioset_pretrain=audioset_pretrain,
            num_features=num_features,
            n_mels=n_mels,
            sample_rate=sample_rate,
            hoplen=hoplen,
            decoder_type=decoder_type,
            num_decoder_layers=num_decoder_layers,
            num_freeze_layers=num_freeze_layers,
            decoder_kwargs=decoder_kwargs
        )

class ConvConformer_Multi(ConvConformer):
    def __init__(self, params=None, **kwargs):
        """
        Initialize ConvConformer_Multi model, style consistent with HTSAT_multi
        Args:
            params (dict): Parameter dictionary, must be provided
            **kwargs: Other parameters
        """
        if params is None:
            raise ValueError("params parameter must be provided")
        self.params = params

        # Extract necessary parameters from params
        num_classes = self.params.get('nb_classes', 13)
        in_channels = 7  # Fixed to 7 channels
        n_mels = self.params.get('nb_mels', 64)
        sample_rate = self.params.get('sampling_rate', 24000)
        hoplen = int(sample_rate * self.params.get('hop_length_s', 0.02))
        pretrained_path = self.params.get('CNN14_Conformer_pretrained_dir', '/root/CODE/DCASE_2025_task3/2025_my_pretrained/PSELD_pretrained_ckpts/mACCDOA-CNN14-Conformer-0.582.ckpt')
        # Handle string 'None' cases
        if pretrained_path == 'None' or pretrained_path == 'none':
            pretrained_path = None
        freeze_all_encoder = self.params.get('freeze_all_encoder', False)
        num_features = self.params.get('num_features', [64, 128, 256, 512, 1024, 2048])
        num_decoder_layers = self.params.get('num_decoder_layers', 2)
        num_freeze_layers = self.params.get('num_freeze_layers', 0)
        encoder = self.params.get('encoder', 'CNN12')
        decoder_type = self.params.get('decoder_type', 'bmamba')
        audioset_pretrain = self.params.get('audioset_pretrain', False)
        decoder_kwargs = self.params.get('decoder_kwargs', {})  # Extract decoder_kwargs

        # Call parent class initialization
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            encoder=encoder,
            pretrained_path=pretrained_path,
            audioset_pretrain=audioset_pretrain,
            num_features=num_features,
            n_mels=n_mels,
            sample_rate=sample_rate,
            hoplen=hoplen,
            decoder_type=decoder_type,
            num_decoder_layers=num_decoder_layers,
            num_freeze_layers=num_freeze_layers,
            decoder_kwargs=decoder_kwargs  # Pass decoder_kwargs
        )

        # Reset fc layer
        self.fc = nn.Linear(self.num_features[-1], 3 * 3 * self.num_classes, bias=True)

        # Freeze encoder layers
        if freeze_all_encoder:
            for param in self.convs.parameters():
                param.requires_grad = False
            print("Freezing all encoder layers")
        else:
            # First unfreeze all, then freeze first num_freeze_layers layers
            for param in self.convs.parameters():
                param.requires_grad = True
            if num_freeze_layers > 0:
                self.freeze_encoder_layers(num_freeze_layers)
                print(f"Only freezing first {num_freeze_layers} encoder layers")

        # Re-print parameter information (parameter count changed due to fc layer reset)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = trainable_params + non_trainable_params
        print(f'Final trainable parameters: {trainable_params:,}')
        print(f'Final non-trainable parameters: {non_trainable_params:,}')
        print(f'Final total parameters: {total_params:,}')
        print(f'Final trainable parameter ratio: {trainable_params/total_params*100:.2f}%')
    
    def forward(self, audio_features, video_features=None):
        """
        Args:
            audio_features: Audio features, shape (batch_size, num_channels, time_frames, mel_bins)
            video_features: Video features, not used here, keep as None
        """
        # Only use audio features
        return super().forward(audio_features)['accdoa']

### ---------------------------------------------------###
class HTSAT(nn.Module):
    def __init__(self, num_classes, in_channels=7, n_mels=64, sample_rate=24000, hop_length=480,
                 audioset_pretrain=True, pretrained_path='/root/CODE/DCASE_2025_task3/2025_my_pretrained/PSELD_pretrained_ckpts/mACCDOA-HTSAT-0.567.ckpt', 
                 freeze_all_encoder=False,**kwargs):
        super().__init__()
        
        self.label_res = 0.1
        self.num_classes = num_classes
        self.output_frames = None
        self.tgt_output_frames = int(5 / 0.1)  # 5-second clip input to the model
        self.pred_res = int(sample_rate / hop_length * self.label_res)
        self.in_channels = in_channels
        self.freeze_all_encoder = freeze_all_encoder
        # scalar
        self.scalar = nn.ModuleList([nn.BatchNorm2d(n_mels) for _ in range(in_channels)])
        
        # encoder
        self.encoder = HTSAT_Swin_Transformer(
            in_chans=in_channels,
            mel_bins=n_mels,
            **kwargs
        )
        
        # fc
        self.tscam_conv = nn.Conv2d(
            in_channels = self.encoder.num_features,
            out_channels = self.num_classes * 3 * 3,
            kernel_size = (self.encoder.SF,3),
            padding = (0,1))

        self.fc = nn.Identity()
        self.xy_act = nn.Tanh()
        self.distance_act = nn.ReLU()

        if pretrained_path:
            print('Loading pretrained model from {}...'.format(pretrained_path))
            self.load_ckpts(pretrained_path, audioset_pretrain)





    def load_ckpts(self, pretrained_path, audioset_pretrain=True):
        if audioset_pretrain:
            print('AudioSet-pretrained model...')
            htsat_ckpts = torch.load(pretrained_path, map_location='cpu')['state_dict']
            htsat_ckpts = {k.replace('sed_model.', ''): v for k, v in htsat_ckpts.items()}
            for key, value in self.encoder.state_dict().items():
                try:
                    if key == 'patch_embed.proj.weight':
                        if key in htsat_ckpts:
                            paras = htsat_ckpts[key].repeat(1, self.in_channels, 1, 1) / self.in_channels
                            value.data.copy_(paras)
                        else:
                            print(f"Warning: {key} not in checkpoint, skip patch_embed init.")
                    elif 'tscam_conv' not in key and 'head' not in key and 'adapter' not in key:
                        if key in htsat_ckpts:
                            value.data.copy_(htsat_ckpts[key])
                        else:
                            print(f"Warning: {key} not in checkpoint, skip.")
                    else: 
                        print(f'Skipping {key}...')
                except Exception as e:
                    print(f"Error loading {key}: {e}")
            for ich in range(self.in_channels):
                if 'bn0.weight' in htsat_ckpts:
                    self.scalar[ich].weight.data.copy_(htsat_ckpts['bn0.weight'])
                    self.scalar[ich].bias.data.copy_(htsat_ckpts['bn0.bias'])
                    self.scalar[ich].running_mean.copy_(htsat_ckpts['bn0.running_mean'])
                    self.scalar[ich].running_var.copy_(htsat_ckpts['bn0.running_var'])
                    self.scalar[ich].num_batches_tracked.copy_(htsat_ckpts['bn0.num_batches_tracked'])
        else:
            print('DataSynthSELD-pretrained model...')
            ckpt = torch.load(pretrained_path, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            for idx, (key, value) in enumerate(self.state_dict().items()):
                if key.startswith(('fc.', 'head.', 'tscam_conv.')) or 'lora' in key or 'adapter' in key:
                    print(f'{idx+1}/{len(self.state_dict())}: Skipping {key}...')
                else:
                    try: 
                        if key in ckpt:
                            value.data.copy_(ckpt[key])
                        else:
                            print(f'{idx+1}/{len(self.state_dict())}: {key} not in ckpt.dict, skipping...')
                    except Exception as e:
                        print(f"Error loading {key}: {e}")

    def freeze_layers(self):
        # Freeze first 3 stages
        for i, layer in enumerate(self.encoder.layers):
            if i < 0:  # Freeze first 3 layers
                for param in layer.parameters():
                    param.requires_grad = False
            else:  # Keep 4th layer trainable
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x):
        """
        x: waveform, (batch_size, num_channels, time_frames, mel_bins)
        """
        B, C, T, F = x.shape

        # Concatenate clips to a 10-second clip if necessary
        if self.output_frames is None:
            self.output_frames = int(T // self.pred_res)
        if self.output_frames < self.tgt_output_frames:
            assert self.output_frames == self.tgt_output_frames // 2, \
                'only support 5-second or 10-second clip to be input to the model'
            factor = 2
            assert B % factor == 0, 'batch size should be a factor of {}'.format(factor)
            x = torch.cat((x[:B//factor, :, :-1], x[B//factor:, :, :-1]), dim=2)
        elif self.output_frames > self.tgt_output_frames:
            raise NotImplementedError('output_frames > tgt_output_frames is not implemented')
            
        # Compute scalar
        x = x.transpose(1, 3)
        for nch in range(x.shape[-1]):
            x[..., [nch]] = self.scalar[nch](x[..., [nch]])
        x = x.transpose(1, 3)

        x = self.encoder(x)
        x = self.tscam_conv(x)
        x = torch.flatten(x, 2) # (B, C, T)
        x = x.permute(0,2,1).contiguous() # B, T, C
        x = self.fc(x)

        # Match the output shape
        x = interpolate(x, ratio=self.encoder.time_res, method='bilinear')
        x = x[:, :self.output_frames * self.pred_res]
        if self.output_frames < self.tgt_output_frames:
            x_output_frames = self.output_frames * self.pred_res
            x = torch.cat((x[:B//2, :x_output_frames], x[B//2:, :x_output_frames]), dim=0)
        x = x.reshape(B, self.output_frames, self.pred_res, -1).mean(dim=2)
        # Reshape output to separate x,y and distance
        x = x.reshape(B, self.output_frames, 3, 3, -1)  # 3 tracks, 3 values (x,y,distance), num_classes
        
        x[..., 0:2, :] = self.xy_act(x[..., 0:2, :])  # x,y coordinates
        x[..., 2, :] = self.distance_act(x[..., 2, :])  # distance
        # Reshape back to original shape
        x = x.reshape(B, self.output_frames, -1)


        return {
            'accdoa': x,
        }


class HTSAT_multi(HTSAT):
    def __init__(self, params=None, **kwargs):
        """
        Initialize HTSAT model
        
        Args:
            params (dict): Parameter dictionary, must be provided
            **kwargs: Other parameters
        """
        if params is None:
            raise ValueError("params parameter must be provided")
        
        self.params = params

        # Extract necessary parameters from params
        num_classes = self.params.get('nb_classes', 13)
        in_channels = 7  # Fixed to 7 channels
        n_mels = self.params.get('nb_mels', 64)
        sample_rate = self.params.get('sampling_rate', 24000)
        hop_length = int(sample_rate * self.params.get('hop_length_s', 0.02))
        pretrained_path = self.params.get('HTS_AT_pretrained_dir', '/root/CODE/DCASE_2025_task3/2025_my_pretrained/PSELD_pretrained_ckpts/mACCDOA-HTSAT-0.567.ckpt')
        # Handle string 'None' cases
        if pretrained_path == 'None' or pretrained_path == 'none':
            pretrained_path = None
        freeze_all_encoder = self.params.get('freeze_all_encoder', False)

        # Call parent class initialization
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            n_mels=n_mels,
            sample_rate=sample_rate,
            hop_length=hop_length,
            pretrained_path=pretrained_path,
            audioset_pretrain=False,
            freeze_all_encoder=freeze_all_encoder,
            **kwargs
        )

        # Freeze all encoder layers
        if self.freeze_all_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Freezing all encoder layers")
        else:
            # Ensure all parameters are set to trainable before calling freeze_layers
            for param in self.encoder.parameters():
                param.requires_grad = True
            # Then freeze first 3 layers
            self.freeze_layers()
            print("Only freezing 0 encoder stages")
        
        self.tscam_conv.requires_grad_(True)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f'Trainable parameters: {trainable_params:,}')
        print(f'Non-trainable parameters: {non_trainable_params:,}')
        print(f'Total parameters: {trainable_params + non_trainable_params:,}')
        print(f'Trainable parameter ratio: {trainable_params/(trainable_params + non_trainable_params)*100:.2f}%')
    
    def forward(self, audio_features, video_features=None):
        """
        Args:
            audio_features: Audio features, shape (batch_size, num_channels, time_frames, mel_bins)
            video_features: Video features, not used here, keep as None
        """
        # Only use audio features
        return super().forward(audio_features)['accdoa']

class SELDModel(nn.Module):
    """
    SELD (Sound Event Localization and Detection) model combining ConvBlock, recurrent, and attention-based layers.
    Supports audio-only and audio_visual input.
    """
    def __init__(self, params=None):
        """
        Initialize SELD model
        
        Args:
            params (dict): Parameter dictionary, must be provided
        """
        super().__init__()
        if params is None:
            raise ValueError("params parameter must be provided")
        
        self.params = params
        
        # Conv layers
        self.conv_blocks = nn.ModuleList()
        for conv_cnt in range(self.params['nb_conv_blocks']):
            self.conv_blocks.append(ConvBlock(in_channels=self.params['nb_conv_filters'] if conv_cnt else 4,  # stereo
                                              out_channels=self.params['nb_conv_filters'],
                                              pool_size=(self.params['t_pool_size'][conv_cnt], self.params['f_pool_size'][conv_cnt]),
                                              dropout=self.params['dropout']))

        # GRU layers
        self.gru_input_dim = self.params['nb_conv_filters'] * int(np.floor(self.params['nb_mels'] / np.prod(self.params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=self.params['rnn_size'], num_layers=self.params['nb_rnn_layers'],
                                batch_first=True, dropout=self.params['dropout'], bidirectional=True)

        # Self attention layers
        self.mhsa_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=self.params['nb_attn_heads'],
                                  dropout=self.params['dropout'], batch_first=True) for _ in range(self.params['nb_self_attn_layers'])])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.params['rnn_size']) for _ in range(self.params['nb_self_attn_layers'])])

        self.fnn_list = torch.nn.ModuleList()

        for fc_cnt in range(self.params['nb_fnn_layers']):
            self.fnn_list.append(nn.Linear(self.params['fnn_size'] if fc_cnt else self.params['rnn_size'], self.params['fnn_size'], bias=True))

        if self.params['multiACCDOA']:
            if self.params['modality'] == 'audio_visual':
                self.output_dim = self.params['max_polyphony'] * 4 * self.params['nb_classes']  # 4 => (x,y), distance, on/off
            else:
                self.output_dim = self.params['max_polyphony'] * 3 * self.params['nb_classes']  # 3 => (x,y), distance
        else:
            if self.params['modality'] == 'audio_visual':
                self.output_dim = 4 * self.params['nb_classes']  # 4 => (x,y), distance, on/off
            else:
                self.output_dim = 3 * self.params['nb_classes']  # 3 => (x,y), distance
        self.fnn_list.append(nn.Linear(self.params['fnn_size'] if self.params['nb_fnn_layers'] else self.params['rnn_size'], self.output_dim, bias=True))

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()
        if self.params['modality'] == 'audio_visual':
            self.onscreen_act = nn.Sigmoid()

    def forward(self, audio_feat, vid_feat=None):
        """
        Forward pass for the SELD model.
        audio_feat: Tensor of shape (batch_size, 2, 251, 64) (stereo spectrogram input).
        Returns:  Tensor of shape
                  (batch_size, 50, 117) - audio - multiACCDOA.
                  (batch_size, 50, 39)  - audio - singleACCDOA.

        """
        # audio feat - B x 2 x 251 x 64
        for conv_block in self.conv_blocks:
            audio_feat = conv_block(audio_feat)  # B x 64 x 50 x 2

        audio_feat = audio_feat.transpose(1, 2).contiguous()  # B x 50 x 64 x 2
        audio_feat = audio_feat.view(audio_feat.shape[0], audio_feat.shape[1], -1).contiguous()  # B x 50 x 128

        (audio_feat, _) = self.gru(audio_feat)
        audio_feat = torch.tanh(audio_feat)
        audio_feat = audio_feat[:, :, audio_feat.shape[-1] // 2:] * audio_feat[:, :, :audio_feat.shape[-1] // 2]

        for mhsa, ln in zip(self.mhsa_layers, self.layer_norms):
            audio_feat_in = audio_feat
            audio_feat, _ = mhsa(audio_feat_in, audio_feat_in, audio_feat_in)
            audio_feat = audio_feat + audio_feat_in  # Residual connection
            audio_feat = ln(audio_feat)

        if vid_feat is not None:
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)  # b x 50 x 49
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            fused_feat = self.transformer_decoder(audio_feat, vid_feat)
        else:
            fused_feat = audio_feat

        for fnn_cnt in range(len(self.fnn_list) - 1):
            fused_feat = self.fnn_list[fnn_cnt](fused_feat)
        pred = self.fnn_list[-1](fused_feat)

        if self.params['modality'] == 'audio':
            if self.params['multiACCDOA']:
                # pred shape is batch,50,117 - 117 is 3 tracks x 3 (doa-x, doa-y, dist) x 13 classes
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 3, 13)
                doa_pred = pred[:, :, :, 0:2, :]
                dist_pred = pred[:, :, :, 2:3, :]
                doa_pred = self.doa_act(doa_pred)
                dist_pred = self.dist_act(dist_pred)
                pred = torch.cat((doa_pred, dist_pred), dim=3)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)
            else:
                # pred shape is batch,50,39 - 39 is 3 (doa-x, doa-y, dist) x 13 classes
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 13)
                doa_pred = pred[:, :,  0:2, :]
                dist_pred = pred[:, :, 2:3, :]
                doa_pred = self.doa_act(doa_pred)
                dist_pred = self.dist_act(dist_pred)
                pred = torch.cat((doa_pred, dist_pred), dim=2)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)


        return pred


if __name__ == '__main__':
    # Test code
    test_params = {
        'multiACCDOA': True,
        'modality': 'audio',
        'nb_classes': 13,
        'nb_mels': 64,
        'sampling_rate': 24000,
        'hop_length_s': 0.02,
        'nb_conv_blocks': 3,
        'nb_conv_filters': 64,
        'f_pool_size': [4, 4, 2],
        't_pool_size': [5, 1, 1],
        'dropout': 0.05,
        'rnn_size': 128,
        'nb_rnn_layers': 2,
        'nb_self_attn_layers': 2,
        'nb_attn_heads': 8,
        'nb_fnn_layers': 1,
        'fnn_size': 128,
        'max_polyphony': 3
    }

    test_audio_feat = torch.rand([8, 4, 251, 64])
    test_model = SELDModel(test_params)
    print(test_model)

    # Count parameters
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    # ConvConformer_Multi
    cc_params = {
        'nb_classes': 13,
        'nb_mels': 64,
        'sampling_rate': 24000,
        'hop_length_s': 0.02,
        'CNN14_Conformer_pretrained_dir': '',
        'freeze_all_encoder': False,
        'num_features': [64, 128, 256, 512, 1024, 2048],
        'num_decoder_layers': 2,
        'num_freeze_layers': 0,
        'encoder': 'CNN12',
        'decoder_type': 'conformer',
        'audioset_pretrain': False
    }
    cc_model = ConvConformer_Multi(cc_params)
    cc_total, cc_trainable = count_parameters(cc_model)
    print(f"ConvConformer_Multi: total params = {cc_total:,}, trainable params = {cc_trainable:,}")

    # HTSAT_multi
    htsat_params = {
        'nb_classes': 13,
        'nb_mels': 64,
        'sampling_rate': 24000,
        'hop_length_s': 0.02,
        'HTS_AT_pretrained_dir': '',
        'freeze_all_encoder': False
    }
    htsat_model = HTSAT_multi(htsat_params)
    htsat_total, htsat_trainable = count_parameters(htsat_model)
    print(f"HTSAT_multi: total params = {htsat_total:,}, trainable params = {htsat_trainable:,}")

    # SELDModel
    seld_model = SELDModel(test_params)
    seld_total, seld_trainable = count_parameters(seld_model)
    print(f"SELDModel: total params = {seld_total:,}, trainable params = {seld_trainable:,}")



