#!/usr/bin/env python3
"""
parameters_macs_calculation_ptflops.py

Calculate MACs (Multiply-Accumulate Operations) for SELD models using ptflops library.
Automated calculation, more accurate and reliable.

This module provides functionality to:
- Calculate model parameters and MACs using ptflops
- Support multiple SELD model architectures
- Format and display calculation results
- Parse ptflops output in various formats

Author: Gavin
Date: June 2025
"""

import torch
import torch.nn as nn
import sys
import os
from typing import Dict, Tuple, List
import numpy as np

# Add project path
sys.path.append('/root/CODE/2025_my_pretrained')

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    print("Warning: ptflops library not installed, please install with 'pip install ptflops'")
    PTFLOPS_AVAILABLE = False

from model import HTSAT_multi, ConvConformer_Multi, SELDModel
from experiments.baseline_config import get_baseline_params
from experiments.EXP_CNN14_BiMamba import get_params as get_bmamba_params
from experiments.EXP_CNN14_BiMambaAC import get_params as get_bmamba2dac_params
from experiments.EXP_CNN14_Conformer import get_params as get_conformer_params
from experiments.EXP_HTSAT import get_params as get_htsat_params
from experiments.EXP_CNN14_ConBimamba import get_params as get_conbimamba_params
from experiments.EXP_CNN14_ConBiMambaAC import get_params as get_conbimambaac_params

def format_number(num):
    """Format number with thousands separator"""
    return f"{num:,}"

def format_flops(flops):
    """Format FLOPs number, convert to appropriate unit"""
    if flops == 0:
        return "N/A"
    elif flops < 1e6:
        return f"{flops/1e3:.2f}K"
    elif flops < 1e9:
        return f"{flops/1e6:.2f}M"
    elif flops < 1e12:
        return f"{flops/1e9:.2f}G"
    else:
        return f"{flops/1e12:.2f}T"

def calculate_model_macs_ptflops(model, input_shape, model_name):
    """Calculate model MACs and parameters using ptflops"""
    if not PTFLOPS_AVAILABLE:
        return 0, 0, 0, 0, {"error": "ptflops not available"}
    
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Prepare input shape (remove batch dimension)
        if len(input_shape) == 4:  # (B, C, H, W)
            input_size = input_shape[1:]  # (C, H, W)
        elif len(input_shape) == 3:  # (B, C, L)
            input_size = input_shape[1:]  # (C, L)
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        # Calculate using ptflops
        print(f"   Calculating {model_name} with ptflops, input size: {input_size}")
        
        # Use simpler way to call ptflops
        with torch.no_grad():
            try:
                # Try direct call to get_model_complexity_info
                flops, params_calc = get_model_complexity_info(
                    model, 
                    input_size,
                    print_per_layer_stat=False,
                    verbose=False,
                    as_strings=False  # Return numerical values instead of strings
                )
                
                # ptflops returns FLOPs, MACs = FLOPs / 2
                macs_value = flops // 2
                flops_value = flops
                
                print(f"   ptflops calculation successful: FLOPs={flops}, calculated MACs={macs_value}")
                
            except Exception as e1:
                print(f"   ptflops numerical mode failed: {e1}")
                # Try string mode
                try:
                    flops_str, params_calc = get_model_complexity_info(
                        model, 
                        input_size,
                        print_per_layer_stat=False,
                        verbose=False,
                        as_strings=True  # Return strings
                    )
                    
                    # Parse string
                    flops_value = parse_flops_string(flops_str)
                    macs_value = flops_value // 2 if flops_value > 0 else 0
                    
                    print(f"   ptflops string mode successful: {flops_str} -> FLOPs={flops_value}, MACs={macs_value}")
                    
                except Exception as e2:
                    print(f"   ptflops string mode also failed: {e2}")
                    # Finally try basic mode
                    try:
                        result = get_model_complexity_info(model, input_size)
                        if isinstance(result, tuple) and len(result) >= 2:
                            flops_raw, params_calc = result[0], result[1]
                            if isinstance(flops_raw, str):
                                flops_value = parse_flops_string(flops_raw)
                            else:
                                flops_value = int(flops_raw)
                            macs_value = flops_value // 2 if flops_value > 0 else 0
                            print(f"   ptflops basic mode successful: FLOPs={flops_value}, MACs={macs_value}")
                        else:
                            raise ValueError(f"ptflops return format anomaly: {result}")
                    except Exception as e3:
                        print(f"   All ptflops modes failed: {e3}")
                        flops_value = 0
                        macs_value = 0
                        params_calc = total_params
        
        # Verify parameter calculation consistency
        if abs(params_calc - total_params) > 100:  # Allow small errors
            print(f"   Warning: ptflops parameters ({params_calc}) don't match actual calculation ({total_params})")
        
        return total_params, trainable_params, macs_value, flops_value, {
            "ptflops_params": params_calc,
            "ptflops_flops": flops_value,
            "ptflops_macs": macs_value,
            "input_size": input_size
        }
        
    except Exception as e:
        print(f"   ptflops overall calculation failed: {e}")
        # Fallback to basic parameter calculation
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params, 0, 0, {"error": str(e)}

def parse_flops_string(flops_str):
    """Parse ptflops returned FLOPs string"""
    try:
        if isinstance(flops_str, (int, float)):
            return int(flops_str)
            
        # Remove spaces and convert to lowercase
        flops_str = str(flops_str).replace(" ", "").lower()
        
        # Extract numerical part
        import re
        match = re.match(r"([0-9.]+)([a-z]*)", flops_str)
        if not match:
            print(f"   Cannot parse FLOPs string format: '{flops_str}'")
            return 0
            
        value = float(match.group(1))
        unit = match.group(2)
        
        # Convert units to actual values
        multipliers = {
            'k': 1e3, 'kmac': 1e3, 'kflop': 1e3, 'kflops': 1e3,
            'm': 1e6, 'mmac': 1e6, 'mflop': 1e6, 'mflops': 1e6,
            'g': 1e9, 'gmac': 1e9, 'gflop': 1e9, 'gflops': 1e9,
            't': 1e12, 'tmac': 1e12, 'tflop': 1e12, 'tflops': 1e12,
            'flops': 1, 'flop': 1, 'mac': 1, 'macs': 1,
            '': 1  # No unit
        }
        
        multiplier = multipliers.get(unit, 1)
        result = int(value * multiplier)
        
        print(f"   Parse successful: '{flops_str}' -> {result}")
        return result
        
    except Exception as e:
        print(f"   Failed to parse FLOPs string '{flops_str}': {e}")
        return 0

def analyze_model_with_ptflops(model_name, model_creator, input_shape):
    """Analyze single model using ptflops"""
    print(f"\nAnalyzing {model_name} model - input shape: {input_shape}")
    
    try:
        # Create model
        model = model_creator()
        
        # Verify forward pass
        print(f"   Model created, verifying forward pass...")
        
        # Create test input
        if len(input_shape) == 4:
            test_input = torch.randn(*input_shape)
        else:
            test_input = torch.randn(*input_shape)
        
        with torch.no_grad():
            output = model(test_input)
            if isinstance(output, dict):
                print(f"   Output shape verification: {output['accdoa'].shape}")
            elif hasattr(output, 'shape'):
                print(f"   Output shape verification: {output.shape}")
            else:
                print(f"   Output type: {type(output)}")
    except Exception as e:
        print(f"   Forward pass verification failed: {e}")
        return None
    
    # Move model back to CPU for MACs calculation (ptflops is more stable on CPU)
    model = model.cpu()
    
    # Calculate using ptflops
    total_params, trainable_params, macs_value, flops_value, info = calculate_model_macs_ptflops(
        model, input_shape, model_name
    )
    
    print(f"   Total parameters: {format_number(total_params)}")
    print(f"   Trainable parameters: {format_number(trainable_params)}")
    print(f"   MACs: {format_flops(macs_value)}")
    print(f"   FLOPs: {format_flops(flops_value)}")
    
    if 'ptflops_macs' in info and info['ptflops_macs'] > 0:
        print(f"   ptflops raw result: {info['ptflops_macs']}")
    
    return {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'macs': macs_value,
        'flops': flops_value,
        'info': info
    }

def main():
    """Main function"""
    try:
        print("Calculating SELD model MACs using ptflops")
        print("=" * 60)
        
        if not PTFLOPS_AVAILABLE:
            print("Error: ptflops library required")
            print("Please run: pip install ptflops")
            return
        
        # Detect device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        print("=" * 60)
        
        models_config = [
            ("HTSAT", 
             lambda: HTSAT_multi(params=get_htsat_params()), 
             (1, 7, 251, 64)),
            
            ("CNN14-Conformer", 
             lambda: ConvConformer_Multi(params=get_conformer_params()), 
             (1, 7, 251, 64)),
            
            ("CNN14-BiMamba", 
             lambda: ConvConformer_Multi(params=get_bmamba_params()), 
             (1, 7, 251, 64)),
            
            ("CNN14-BiMamba2DAC", 
             lambda: ConvConformer_Multi(params=get_bmamba2dac_params()), 
             (1, 7, 251, 64)),
            
            ("CNN14-ConBiMamba", 
             lambda: ConvConformer_Multi(params=get_conbimamba_params()), 
             (1, 7, 251, 64)),
            
            ("CNN14-ConBiMamba-AC", 
             lambda: ConvConformer_Multi(params=get_conbimambaac_params()), 
             (1, 7, 251, 64)),
            
            ("SELDModel", 
             lambda: SELDModel(params=get_baseline_params()), 
             (1, 4, 251, 64))
        ]
        
        results = []
        
        for model_name, model_creator, input_shape in models_config:
            print(f"\n{'='*50}")
            
            result = analyze_model_with_ptflops(model_name, model_creator, input_shape)
            if result:
                results.append(result)
                print(f"✅ {model_name} analysis completed")
            else:
                print(f"❌ {model_name} analysis failed")
        
        # Summary table
        print("\n" + "=" * 80)
        print("MACs Calculation Results Summary (using ptflops)")
        print("=" * 80)
        print(f"{'Model Name':<20} {'Total Params':<15} {'Trainable Params':<15} {'MACs':<15} {'FLOPs':<15} {'Input Shape':<20}")
        print("-" * 80)
        
        valid_results = [r for r in results if r]
        for result in valid_results:
            print(f"{result['model_name']:<20} {format_number(result['total_params']):<15} {format_number(result['trainable_params']):<15} {format_flops(result['macs']):<15} {format_flops(result['flops']):<15} {str(input_shape):<20}")
        
        # Sorting analysis
        print(f"\nSorted by MACs (smallest to largest):")
        valid_results.sort(key=lambda x: x['macs'])
        for i, result in enumerate(valid_results, 1):
            if result['macs'] > 0:
                print(f"{i}. {result['model_name']}: {format_flops(result['macs'])} MACs")
        
        print(f"\nSorted by parameters (smallest to largest):")
        valid_results.sort(key=lambda x: x['trainable_params'])
        for i, result in enumerate(valid_results, 1):
            print(f"{i}. {result['model_name']}: {format_number(result['trainable_params'])} parameters")
        
        print("\nNote: Using ptflops for automatic calculation, results are more accurate and reliable")
        print("=" * 80)
        
    except Exception as e:
        print(f"Main function failed: {e}")

if __name__ == "__main__":
    main()
