#!/usr/bin/env python3
"""
parameters_inference_time_calculation.py

Calculate parameter counts (trainable parameters) for different SELD models.
Supported models: SELDModel, HTSAT, CNN14-Conformer, CNN14-BiMamba, CNN14-BiMamba2DAC

This module performs comprehensive analysis of SELD models including:
- Parameter count calculation
- FLOPs estimation
- Inference time measurement
- Memory usage analysis

Author: Gavin
Date: June 2025
"""

import torch
import sys
import os
import time
import numpy as np

# Add project path
sys.path.append('/root/CODE/DCASE_2025_task3/2025_my_pretrained')

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    print("Warning: thop library not installed, skipping FLOPs calculation")
    THOP_AVAILABLE = False

from model import CRNN, HTSAT_multi, ConvConformer_Multi, SELDModel
from experiments.baseline_config import get_baseline_params
from experiments.CNN14_BMAMBA_PFOA_YSWAP import get_params as get_bmamba_params
from experiments.CNN14_BMAMBA2DAC_PFOA_YSWAP import get_params as get_bmamba2dac_params
from experiments.CNN14_Conformer_PFOA_YSWAP import get_params as get_conformer_params
from experiments.HTSAT_PFOA_YSWAP import get_params as get_htsat_params
from experiments.CNN14_ConBimamba_PFOA_YSWAP import get_params as get_conbimamba_params
from experiments.CNN14_ConBimamba_AC_PFOA_YSWAP import get_params as get_conbimambaac_params

def measure_inference_time(model, input_tensor, num_runs=100, warmup_runs=10):
    """
    Measure model inference time
    
    Args:
        model: Model to test
        input_tensor: Input tensor
        num_runs: Number of test runs
        warmup_runs: Number of warmup runs
        
    Returns:
        tuple: (Average inference time (ms), Standard deviation (ms))
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup GPU
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Synchronize GPU operations
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Start timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                _ = model(input_tensor)
                end_event.record()
                
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)  # ms
                times.append(elapsed_time)
            else:
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # Convert to ms
                times.append(elapsed_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time

def count_parameters_and_flops(model, input_tensor):
    """
    Calculate model parameters and FLOPs
    
    Args:
        model: Model to analyze
        input_tensor: Input tensor
        
    Returns:
        tuple: (Total parameters, Trainable parameters, MACs, FLOPs, Inference time (ms), Time std (ms), Memory usage (bytes))
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    macs, flops = 0, 0
    if THOP_AVAILABLE:
        try:
            # Use thop to calculate FLOPs (Note: thop.profile returns FLOPs not MACs)
            model_copy = model
            flops, _ = profile(model_copy, inputs=(input_tensor,), verbose=False)
            # MACs = FLOPs / 2 (one MAC contains one multiplication and one addition)
            macs = flops // 2
            print(f"   thop calculation successful: FLOPs={flops}, calculated MACs={macs}")
        except Exception as e:
            print(f"   FLOPs calculation failed: {e}")
            macs, flops = 0, 0
    
    # Measure inference time
    try:
        avg_time, std_time = measure_inference_time(model, input_tensor)
    except Exception as e:
        print(f"   Inference time measurement failed: {e}")
        avg_time, std_time = 0, 0
    
    # Measure memory usage
    try:
        memory_usage = measure_memory_usage(model, input_tensor)
    except Exception as e:
        print(f"   Memory measurement failed: {e}")
        memory_usage = 0
    
    return total_params, trainable_params, macs, flops, avg_time, std_time, memory_usage

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

def format_time(time_ms):
    """Format inference time"""
    if time_ms == 0:
        return "N/A"
    elif time_ms < 1:
        return f"{time_ms:.3f}ms"
    elif time_ms < 1000:
        return f"{time_ms:.2f}ms"
    else:
        return f"{time_ms/1000:.2f}s"

def format_memory(bytes):
    """Format memory size"""
    if bytes == 0:
        return "N/A"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.2f}TB"

def measure_memory_usage(model, input_tensor):
    """Measure peak memory usage"""
    device = next(model.parameters()).device
    if device.type != 'cuda':
        return 0
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(input_tensor)
    
    return torch.cuda.max_memory_allocated()

def calculate_model_parameters():
    """Calculate parameters for all models"""
    print("=" * 80)
    print("SELD Model Long Sequence Inference Time Comparison")
    print("=" * 80)
    
    # Detect computing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    print(f"Input shape: [batch_size, 7, time_frames, 64]")
    print(f"Inference time measurement: Average of 100 runs, 10 warmup runs")
    print("=" * 80)
    
    # Define long sequence test lengths - adjusted to multiples of 64
    time_lengths = [256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 5120, 7680, 10240, 15360, 20480]
    print(f"Test time lengths: {time_lengths}")
    print("=" * 80)
    
    models_info = []
    
    try:
        # 1. HTSAT model
        print("\n1. Testing HTSAT model...")
        htsat_params = get_htsat_params()
        htsat_model = HTSAT_multi(params=htsat_params).to(device)
        
        # Calculate parameters
        test_input_standard = torch.randn(1, 7, 256, 64).to(device)
        htsat_total, htsat_trainable, htsat_macs, htsat_flops, _, _, _ = count_parameters_and_flops(htsat_model, test_input_standard)
        
        # Test inference time for different time lengths
        htsat_times = {}
        for time_len in time_lengths:
            test_input = torch.randn(1, 7, time_len, 64).to(device)
            try:
                avg_time, std_time = measure_inference_time(htsat_model, test_input)
                htsat_times[time_len] = (avg_time, std_time)
                print(f"   Time length {time_len}: {format_time(avg_time)} ± {format_time(std_time)}")
            except Exception as e:
                print(f"   Time length {time_len} test failed: {e}")
                htsat_times[time_len] = (0, 0)
        
        models_info.append(("HTSAT", htsat_total, htsat_trainable, htsat_macs, htsat_flops, htsat_times))
        
    except Exception as e:
        print(f"   HTSAT model test failed: {e}")
    
    try:
        # 2. CNN14-Conformer model
        print("\n2. Testing CNN14-Conformer model...")
        conformer_params = get_conformer_params()
        conformer_model = ConvConformer_Multi(params=conformer_params).to(device)
        
        # Calculate parameters
        test_input_standard = torch.randn(1, 7, 256, 64).to(device)
        conformer_total, conformer_trainable, conformer_macs, conformer_flops, _, _, _ = count_parameters_and_flops(conformer_model, test_input_standard)
        
        # Test inference time for different time lengths
        conformer_times = {}
        for time_len in time_lengths:
            test_input = torch.randn(1, 7, time_len, 64).to(device)
            try:
                avg_time, std_time = measure_inference_time(conformer_model, test_input)
                conformer_times[time_len] = (avg_time, std_time)
                print(f"   Time length {time_len}: {format_time(avg_time)} ± {format_time(std_time)}")
            except Exception as e:
                print(f"   Time length {time_len} test failed: {e}")
                conformer_times[time_len] = (0, 0)
        
        models_info.append(("CNN14-Conformer", conformer_total, conformer_trainable, conformer_macs, conformer_flops, conformer_times))
        
    except Exception as e:
        print(f"   CNN14-Conformer model test failed: {e}")
    
    try:
        # 3. CNN14-BiMamba model
        print("\n3. Testing CNN14-BiMamba model...")
        bmamba_params = get_bmamba_params()
        bmamba_model = ConvConformer_Multi(params=bmamba_params).to(device)
        
        # Calculate parameters
        test_input_standard = torch.randn(1, 7, 256, 64).to(device)
        bmamba_total, bmamba_trainable, bmamba_macs, bmamba_flops, _, _, _ = count_parameters_and_flops(bmamba_model, test_input_standard)
        
        # Test inference time for different time lengths
        bmamba_times = {}
        for time_len in time_lengths:
            test_input = torch.randn(1, 7, time_len, 64).to(device)
            try:
                avg_time, std_time = measure_inference_time(bmamba_model, test_input)
                bmamba_times[time_len] = (avg_time, std_time)
                print(f"   Time length {time_len}: {format_time(avg_time)} ± {format_time(std_time)}")
            except Exception as e:
                print(f"   Time length {time_len} test failed: {e}")
                bmamba_times[time_len] = (0, 0)
        
        models_info.append(("CNN14-BiMamba", bmamba_total, bmamba_trainable, bmamba_macs, bmamba_flops, bmamba_times))
        
    except Exception as e:
        print(f"   CNN14-BiMamba model test failed: {e}")
    
    try:
        # 4. CNN14-BiMamba2DAC model
        print("\n4. Testing CNN14-BiMamba2DAC model...")
        bmamba2dac_params = get_bmamba2dac_params()
        bmamba2dac_model = ConvConformer_Multi(params=bmamba2dac_params).to(device)
        
        # Calculate parameters
        test_input_standard = torch.randn(1, 7, 256, 64).to(device)
        bmamba2dac_total, bmamba2dac_trainable, bmamba2dac_macs, bmamba2dac_flops, _, _, _ = count_parameters_and_flops(bmamba2dac_model, test_input_standard)
        
        # Test inference time for different time lengths
        bmamba2dac_times = {}
        for time_len in time_lengths:
            test_input = torch.randn(1, 7, time_len, 64).to(device)
            try:
                avg_time, std_time = measure_inference_time(bmamba2dac_model, test_input)
                bmamba2dac_times[time_len] = (avg_time, std_time)
                print(f"   Time length {time_len}: {format_time(avg_time)} ± {format_time(std_time)}")
            except Exception as e:
                print(f"   Time length {time_len} test failed: {e}")
                bmamba2dac_times[time_len] = (0, 0)
        
        models_info.append(("CNN14-BiMamba2DAC", bmamba2dac_total, bmamba2dac_trainable, bmamba2dac_macs, bmamba2dac_flops, bmamba2dac_times))
        
    except Exception as e:
        print(f"   CNN14-BiMamba2DAC model test failed: {e}")
    
    try:
        # 5. CNN14-ConBiMamba model
        print("\n5. Testing CNN14-ConBiMamba model...")
        conbimamba_params = get_conbimamba_params()
        conbimamba_model = ConvConformer_Multi(params=conbimamba_params).to(device)
        
        # Calculate parameters
        test_input_standard = torch.randn(1, 7, 256, 64).to(device)
        conbimamba_total, conbimamba_trainable, conbimamba_macs, conbimamba_flops, _, _, _ = count_parameters_and_flops(conbimamba_model, test_input_standard)
        
        # Test inference time for different time lengths
        conbimamba_times = {}
        for time_len in time_lengths:
            test_input = torch.randn(1, 7, time_len, 64).to(device)
            try:
                avg_time, std_time = measure_inference_time(conbimamba_model, test_input)
                conbimamba_times[time_len] = (avg_time, std_time)
                print(f"   Time length {time_len}: {format_time(avg_time)} ± {format_time(std_time)}")
            except Exception as e:
                print(f"   Time length {time_len} test failed: {e}")
                conbimamba_times[time_len] = (0, 0)
        
        models_info.append(("CNN14-ConBiMamba", conbimamba_total, conbimamba_trainable, conbimamba_macs, conbimamba_flops, conbimamba_times))
        
    except Exception as e:
        print(f"   CNN14-ConBiMamba model test failed: {e}")
    
    try:
        # 6. CNN14-ConBiMamba-AC model
        print("\n6. Testing CNN14-ConBiMamba-AC model...")
        conbimambaac_params = get_conbimambaac_params()
        conbimambaac_model = ConvConformer_Multi(params=conbimambaac_params).to(device)
        
        # Calculate parameters
        test_input_standard = torch.randn(1, 7, 256, 64).to(device)
        conbimambaac_total, conbimambaac_trainable, conbimambaac_macs, conbimambaac_flops, _, _, _ = count_parameters_and_flops(conbimambaac_model, test_input_standard)
        
        # Test inference time for different time lengths
        conbimambaac_times = {}
        for time_len in time_lengths:
            test_input = torch.randn(1, 7, time_len, 64).to(device)
            try:
                avg_time, std_time = measure_inference_time(conbimambaac_model, test_input)
                conbimambaac_times[time_len] = (avg_time, std_time)
                print(f"   Time length {time_len}: {format_time(avg_time)} ± {format_time(std_time)}")
            except Exception as e:
                print(f"   Time length {time_len} test failed: {e}")
                conbimambaac_times[time_len] = (0, 0)
        
        models_info.append(("CNN14-ConBiMamba-AC", conbimambaac_total, conbimambaac_trainable, conbimambaac_macs, conbimambaac_flops, conbimambaac_times))
        
    except Exception as e:
        print(f"   CNN14-ConBiMamba-AC model test failed: {e}")
    
    try:
        # 7. SELDModel baseline model
        print("\n7. Testing SELDModel baseline model...")
        base_params = get_baseline_params()
        seld_model = SELDModel(params=base_params).to(device)
        
        # Calculate parameters
        test_input_standard = torch.randn(1, 4, 256, 64).to(device)
        seld_total, seld_trainable, seld_macs, seld_flops, _, _, _ = count_parameters_and_flops(seld_model, test_input_standard)
        
        # Test inference time for different time lengths (SELDModel uses 4 channels)
        seld_times = {}
        for time_len in time_lengths:
            test_input = torch.randn(1, 4, time_len, 64).to(device)
            try:
                avg_time, std_time = measure_inference_time(seld_model, test_input)
                seld_times[time_len] = (avg_time, std_time)
                print(f"   Time length {time_len}: {format_time(avg_time)} ± {format_time(std_time)}")
            except Exception as e:
                print(f"   Time length {time_len} test failed: {e}")
                seld_times[time_len] = (0, 0)
        
        models_info.append(("SELDModel", seld_total, seld_trainable, seld_macs, seld_flops, seld_times))
        
    except Exception as e:
        print(f"   SELDModel model test failed: {e}")
        
    # Summary table - basic information
    print("\n" + "=" * 80)
    print("Model Basic Information Summary Table")
    print("=" * 80)
    print(f"{'Model Name':<20} {'Total Params':<15} {'Trainable Params':<15} {'Train Ratio':<10} {'Calc MACs':<10} {'Calc FLOPs':<10}")
    print("-" * 80)
    
    for model_name, total, trainable, macs, flops, times_dict in models_info:
        ratio = trainable / total * 100 if total > 0 else 0
        print(f"{model_name:<20} {format_number(total):<15} {format_number(trainable):<15} {ratio:.2f}% {format_flops(macs):<10} {format_flops(flops):<10}")
    
    print("=" * 80)
    
    # Inference time detailed table
    print("\n" + "=" * 150)
    print("Long Sequence Inference Time Comparison Table")
    print("=" * 150)
    
    # Table header
    header = f"{'Model Name':<20}"
    for time_len in time_lengths:
        header += f" {'T='+str(time_len):<12}"
    print(header)
    print("-" * 150)
    
    # Data rows
    for model_name, total, trainable, macs, flops, times_dict in models_info:
        row = f"{model_name:<20}"
        for time_len in time_lengths:
            if time_len in times_dict:
                avg_time, std_time = times_dict[time_len]
                if avg_time > 0:
                    row += f" {format_time(avg_time):<12}"
                else:
                    row += f" {'N/A':<12}"
            else:
                row += f" {'N/A':<12}"
        print(row)
    
    print("=" * 150)
    
    # Simple time growth trend analysis
    print("\n" + "=" * 80)
    print("Time Growth Trend Analysis")
    print("=" * 80)
    
    for model_name, total, trainable, macs, flops, times_dict in models_info:
        print(f"\n{model_name}:")
        valid_times = [(time_len, avg_time) for time_len, (avg_time, _) in times_dict.items() if avg_time > 0]
        if len(valid_times) >= 2:
            valid_times.sort(key=lambda x: x[0])  # Sort by time length
            print(f"  Time lengths: {[t[0] for t in valid_times]}")
            print(f"  Inference times: {[format_time(t[1]) for t in valid_times]}")
            
            # Calculate time growth ratio
            if len(valid_times) >= 2:
                first_time = valid_times[0][1]
                last_time = valid_times[-1][1]
                time_ratio = last_time / first_time if first_time > 0 else 0
                length_ratio = valid_times[-1][0] / valid_times[0][0]
                print(f"  Time growth ratio: {time_ratio:.2f}x (length growth {length_ratio:.2f}x)")
                
                # Calculate average time per frame
                avg_time_per_frame = sum(t[1] for t in valid_times) / sum(t[0] for t in valid_times)
                print(f"  Average time per frame: {format_time(avg_time_per_frame)}")
        else:
            print(f"  Insufficient data for trend analysis")

def main():
    """Main function"""
    print("Starting SELD model parameter calculation...")
    calculate_model_parameters()
    print("\nCalculation completed!")

if __name__ == "__main__":
    main()
