# run_experiment.py

"""
Experiment runner script for DCASE 2025 Task 3 SELD models.

This script provides functionality to:
- Run experiments with different configurations
- Resume training from checkpoints
- List all experiments
- Manage experiment directories and configurations

Usage:
    python run_experiment.py --exp <experiment_name>
    python run_experiment.py --list
    python run_experiment.py --exp <experiment_name> --resume --resume_id <experiment_id>

Author: Gavin
Date: June 2025
"""

import argparse
import importlib
import os
import json
import shutil
from datetime import datetime
from main import main

# Try to import yaml
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

def setup_experiment_dir(exp_id):
    """
    Setup experiment directory structure
    """
    # Create main experiment directory
    exp_dir = os.path.join('experiments', exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'outputs', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    return exp_dir

def save_experiment_config(exp_dir, exp_record):
    """
    Save experiment configuration
    """
    # Save JSON format
    config_file = os.path.join(exp_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(exp_record, f, indent=4)
    
    # Save YAML format
    if HAS_YAML:
        config_yaml = os.path.join(exp_dir, 'config.yaml')
        try:
            with open(config_yaml, 'w', encoding='utf-8') as f:
                yaml.dump(exp_record, f, default_flow_style=False, allow_unicode=True, indent=2)
        except Exception as e:
            print(f"Warning: Unable to save YAML configuration: {e}")

def run_experiment(exp_name, resume=False, resume_id=None):
    try:
        # Import experiment configuration
        try:
            exp_module = importlib.import_module(f'experiments.{exp_name}')
            exp_params = exp_module.get_params()
        except ImportError as e:
            print(f"Unable to import experiment configuration {exp_name}: {str(e)}")
            print("Please ensure the corresponding configuration file exists in the experiments directory")
            raise e
        except AttributeError as e:
            print(f"Configuration file {exp_name} is missing the get_params() function")
            raise e
        
        # Generate experiment ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id = f"{exp_name}_{timestamp}"
        
        if resume and resume_id:
            exp_id = resume_id
            print(f"Resuming training from experiment {resume_id}...")
        
        # Create experiment record
        exp_record = {
            'exp_id': exp_id,
            'exp_name': exp_name,
            'timestamp': timestamp,
            'params': exp_params,
            'status': 'running',
            'start_time': datetime.now().isoformat()
        }
        
        # Setup experiment directory
        exp_dir = setup_experiment_dir(exp_id)
        
        # Save experiment configuration
        save_experiment_config(exp_dir, exp_record)
        
        # Update parameters
        # Note: No need to import from parameters, use exp_params directly
        params = exp_params
        
        # Update output paths
        params['checkpoints_dir'] = os.path.join(exp_dir, 'checkpoints')
        params['log_dir'] = os.path.join(exp_dir, 'logs')
        params['output_dir'] = os.path.join(exp_dir, 'outputs')
        
        # Run experiment
        try:
            # Pass resume training parameters
            main(params=params,  # Pass parameters directly
                 restore_from_checkpoint=resume, 
                 initial_checkpoint_path=os.path.join('experiments', resume_id) if resume_id else None)
            exp_record['status'] = 'completed'
        except Exception as e:
            exp_record['status'] = 'failed'
            exp_record['error'] = str(e)
            raise e
        finally:
            exp_record['end_time'] = datetime.now().isoformat()
            save_experiment_config(exp_dir, exp_record)
            
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        raise e

        
def list_experiments():
    """
    List all experiments
    """
    exp_dir = 'experiments'
    if not os.path.exists(exp_dir):
        print("No experiment directory found")
        return
    
    experiments = []
    for exp_id in os.listdir(exp_dir):
        config_file = os.path.join(exp_dir, exp_id, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                exp_data = json.load(f)
                experiments.append(exp_data)
    
    # Sort by timestamp
    experiments.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Print experiment list
    print("\nExperiment List:")
    print("-" * 80)
    for exp in experiments:
        print(f"ID: {exp['exp_id']}")
        print(f"Name: {exp['exp_name']}")
        print(f"Status: {exp['status']}")
        print(f"Time: {exp['timestamp']}")
        print("-" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment runner script')
    parser.add_argument('--exp', type=str, help='Experiment configuration file name')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--resume_id', type=str, help='Experiment ID to resume from')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
    elif args.exp:
        run_experiment(args.exp, args.resume, args.resume_id)
    else:
        parser.print_help()