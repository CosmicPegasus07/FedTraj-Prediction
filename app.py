from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import os
import json
import threading
import time
from datetime import datetime
import numpy as np
from pathlib import Path
import glob
import sys
import logging

app = Flask(__name__, static_folder='web')
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the base directory of your project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Global variable to track running processes
running_processes = {}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/run_script', methods=['POST'])
def run_script():
    data = request.json
    mode = data.get('mode')
    train_dir = data.get('train_dir')
    val_dir = data.get('val_dir')
    test_dir = data.get('test_dir')
    batch_size = data.get('batch_size', 32)
    num_scenarios = data.get('num_scenarios', -1)
    epochs = data.get('epochs', 5)
    lr = data.get('lr', 1e-3)
    role = data.get('role')
    model_name = data.get('model_name', 'GATv2')

    # Create unique process ID
    process_id = f"{mode}_{model_name}_{int(time.time())}"

    # Check if similar process is already running
    for pid, proc_info in running_processes.items():
        if proc_info['mode'] == mode and proc_info['model_name'] == model_name and proc_info['status'] == 'running':
            return jsonify({"status": "error", "message": f"A {mode} process for {model_name} is already running. Please wait for it to complete."})

    if mode == 'centralized_train':
        from train import centralized_train
        def run_training():
            try:
                running_processes[process_id]['status'] = 'running'
                running_processes[process_id]['start_time'] = time.time()
                running_processes[process_id]['output'] = []

                # Redirect stdout and stderr to capture all output
                import sys
                from io import StringIO
                import builtins

                # Create a custom output handler
                class OutputCapture:
                    def __init__(self, process_id):
                        self.process_id = process_id
                        self.original_stdout = sys.stdout
                        self.original_stderr = sys.stderr
                        self.original_print = builtins.print

                    def write(self, text):
                        # Write to original stdout
                        self.original_stdout.write(text)
                        self.original_stdout.flush()

                        # Also capture for web interface
                        if text.strip() and self.process_id in running_processes:
                            running_processes[self.process_id]['output'].append(text.strip())
                            # Keep only last 1000 lines
                            if len(running_processes[self.process_id]['output']) > 1000:
                                running_processes[self.process_id]['output'] = running_processes[self.process_id]['output'][-1000:]

                    def flush(self):
                        self.original_stdout.flush()

                output_capture = OutputCapture(process_id)
                sys.stdout = output_capture
                sys.stderr = output_capture

                try:
                    running_processes[process_id]['output'].append(f"[INFO] Starting centralized training for {model_name}")
                    running_processes[process_id]['output'].append(f"[INFO] Parameters: scenarios={num_scenarios}, batch_size={batch_size}, epochs={epochs}")

                    centralized_train(
                        train_dir=train_dir,
                        val_dir=val_dir,
                        batch_size=batch_size,
                        num_scenarios=num_scenarios,
                        epochs=epochs,
                        lr=lr,
                        model_name=model_name,
                        seq_len=data.get('seq_len', 30)  # Use configurable seq_len
                    )

                    running_processes[process_id]['output'].append(f"[INFO] Centralized training completed successfully!")

                except Exception as e:
                    error_msg = f"[ERROR] Training failed: {str(e)}"
                    running_processes[process_id]['output'].append(error_msg)
                    logger.error(error_msg)
                    raise e
                finally:
                    # Restore original stdout/stderr
                    sys.stdout = output_capture.original_stdout
                    sys.stderr = output_capture.original_stderr

                running_processes[process_id]['status'] = 'completed'
                running_processes[process_id]['end_time'] = time.time()

            except Exception as e:
                running_processes[process_id]['status'] = 'error'
                running_processes[process_id]['error'] = str(e)
                print(f"Error in centralized training: {e}")

        running_processes[process_id] = {
            'mode': mode,
            'model_name': model_name,
            'status': 'starting',
            'start_time': None,
            'end_time': None,
            'output': []
        }

        thread = threading.Thread(target=run_training)
        thread.start()

        return jsonify({"status": "started", "message": f"Centralized training started for {model_name}", "process_id": process_id})

    elif mode == 'centralized_test' or mode == 'federated_test':
        from train import centralized_test, federated_test

        # Get visualization limit from request
        visualize_limit = data.get('visualize_limit', 10)

        def run_testing():
            try:
                running_processes[process_id]['status'] = 'running'
                running_processes[process_id]['start_time'] = time.time()
                running_processes[process_id]['output'] = []

                # Use the same output capture system
                class OutputCapture:
                    def __init__(self, process_id):
                        self.process_id = process_id
                        self.original_stdout = sys.stdout
                        self.original_stderr = sys.stderr

                    def write(self, text):
                        # Write to original stdout
                        self.original_stdout.write(text)
                        self.original_stdout.flush()

                        # Also capture for web interface
                        if text.strip() and self.process_id in running_processes:
                            running_processes[self.process_id]['output'].append(text.strip())
                            # Keep only last 1000 lines
                            if len(running_processes[self.process_id]['output']) > 1000:
                                running_processes[self.process_id]['output'] = running_processes[self.process_id]['output'][-1000:]

                    def flush(self):
                        self.original_stdout.flush()

                output_capture = OutputCapture(process_id)
                sys.stdout = output_capture
                sys.stderr = output_capture

                try:
                    test_type = "centralized" if mode == 'centralized_test' else "federated"
                    running_processes[process_id]['output'].append(f"[INFO] Starting {test_type} testing for {model_name}")
                    running_processes[process_id]['output'].append(f"[INFO] Parameters: scenarios={num_scenarios}, visualizations={visualize_limit}")

                    if mode == 'centralized_test':
                        centralized_test(
                            test_dir=test_dir,
                            batch_size=batch_size,
                            num_scenarios=num_scenarios,
                            visualize_limit=visualize_limit,
                            model_name=model_name,
                            seq_len=data.get('seq_len', 30)  # Use configurable seq_len
                        )
                    else:  # federated_test
                        federated_test(
                            test_dir=test_dir,
                            batch_size=batch_size,
                            num_scenarios=num_scenarios,
                            visualize_limit=visualize_limit,
                            model_name=model_name,
                            seq_len=data.get('seq_len', 30)  # Use configurable seq_len
                        )

                    running_processes[process_id]['output'].append(f"[INFO] {test_type.capitalize()} testing completed successfully!")

                except Exception as e:
                    error_msg = f"[ERROR] Testing failed: {str(e)}"
                    running_processes[process_id]['output'].append(error_msg)
                    logger.error(error_msg)
                    raise e
                finally:
                    # Restore original stdout/stderr
                    sys.stdout = output_capture.original_stdout
                    sys.stderr = output_capture.original_stderr

                # Consolidate test results (this is already done in the test functions now)

                running_processes[process_id]['status'] = 'completed'
                running_processes[process_id]['end_time'] = time.time()

            except Exception as e:
                running_processes[process_id]['status'] = 'error'
                running_processes[process_id]['error'] = str(e)
                running_processes[process_id]['output'].append(f"Error: {str(e)}")
                print(f"Error in testing: {e}")

        running_processes[process_id] = {
            'mode': mode,
            'model_name': model_name,
            'status': 'starting',
            'start_time': None,
            'end_time': None,
            'output': []
        }

        thread = threading.Thread(target=run_testing)
        thread.start()

        return jsonify({"status": "started", "message": f"Testing started for {model_name} with {visualize_limit} visualizations", "process_id": process_id})

    else:
        return jsonify({"status": "error", "message": "Invalid mode specified"})

def consolidate_test_results(training_type, model_name):
    """Consolidate individual test prediction JSONs into a single comprehensive file"""
    try:
        predictions_dir = os.path.join(RESULTS_DIR, 'test_predictions', training_type, model_name)
        if not os.path.exists(predictions_dir):
            return

        # Find all individual prediction JSON files
        json_files = glob.glob(os.path.join(predictions_dir, 'test_sample_*_predictions.json'))

        consolidated_data = {
            'model_name': model_name,
            'training_type': training_type,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_samples': len(json_files),
            'predictions': [],
            'summary_metrics': {
                'avg_prediction_confidence': 0.0,
                'samples_processed': len(json_files)
            }
        }

        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    sample_data = json.load(f)
                    consolidated_data['predictions'].append(sample_data)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue

        # Save consolidated results
        consolidated_path = os.path.join(predictions_dir, 'consolidated_predictions.json')
        with open(consolidated_path, 'w') as f:
            json.dump(consolidated_data, f, indent=2)

        # Clean up individual files
        for json_file in json_files:
            try:
                os.remove(json_file)
            except:
                pass

        print(f"Consolidated {len(json_files)} prediction files into {consolidated_path}")

    except Exception as e:
        print(f"Error consolidating test results: {e}")

@app.route('/get_process_status/<process_id>', methods=['GET'])
def get_process_status(process_id):
    """Get the status of a running process"""
    if process_id in running_processes:
        process_info = running_processes[process_id].copy()
        if process_info.get('start_time') and process_info.get('end_time'):
            process_info['duration'] = process_info['end_time'] - process_info['start_time']
        elif process_info.get('start_time'):
            process_info['duration'] = time.time() - process_info['start_time']
        return jsonify({"status": "success", "process": process_info})
    else:
        return jsonify({"status": "error", "message": "Process not found"})

@app.route('/get_process_output/<process_id>', methods=['GET'])
def get_process_output(process_id):
    """Get the console output of a running process"""
    if process_id in running_processes:
        output = running_processes[process_id].get('output', [])
        return jsonify({"status": "success", "output": output})
    else:
        return jsonify({"status": "error", "message": "Process not found"})

@app.route('/get_history/<history_type>', methods=['GET'])
def get_history(history_type):
    if history_type == 'centralized':
        history_path = os.path.join(RESULTS_DIR, 'centralized_training_history.json')
    elif history_type == 'federated':
        history_path = os.path.join(RESULTS_DIR, 'federated_training_history.json')
    else:
        return jsonify({"status": "error", "message": "Invalid history type"}), 400

    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)

        # Enhance history with summary statistics
        enhanced_history = []
        for run in history:
            enhanced_run = run.copy()
            if 'history' in run and run['history']:
                # Calculate training summary
                train_losses = [epoch['train_loss'] for epoch in run['history']]
                val_losses = [epoch['val_loss'] for epoch in run['history'] if 'val_loss' in epoch]

                enhanced_run['summary'] = {
                    'initial_train_loss': train_losses[0] if train_losses else 0,
                    'final_train_loss': train_losses[-1] if train_losses else 0,
                    'best_val_loss': min(val_losses) if val_losses else 0,
                    'loss_improvement': (train_losses[0] - train_losses[-1]) / train_losses[0] * 100 if train_losses and train_losses[0] > 0 else 0,
                    'convergence_epoch': len(train_losses),
                    'avg_epoch_time': enhanced_run.get('total_training_time', 0) / len(train_losses) if train_losses else 0
                }
            enhanced_history.append(enhanced_run)

        return jsonify({"status": "success", "history": enhanced_history})
    else:
        return jsonify({"status": "error", "message": f"History file not found: {history_path}"}), 404

@app.route('/get_comparison_data', methods=['GET'])
def get_comparison_data():
    """Get comprehensive comparison data between centralized and federated approaches"""
    try:
        comparison_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'centralized': {},
            'federated': {},
            'comparison_metrics': {}
        }

        print(f"[DEBUG] Loading comparison data...")

        # Load centralized data
        centralized_history_path = os.path.join(RESULTS_DIR, 'centralized_training_history.json')
        centralized_test_path = os.path.join(RESULTS_DIR, 'centralized_test_metrics.json')

        print(f"[DEBUG] Checking centralized history: {centralized_history_path}")
        print(f"[DEBUG] Exists: {os.path.exists(centralized_history_path)}")

        if os.path.exists(centralized_history_path):
            with open(centralized_history_path, 'r') as f:
                centralized_history = json.load(f)
                if centralized_history:
                    latest_run = centralized_history[-1]
                    comparison_data['centralized']['training'] = latest_run
                    print(f"[DEBUG] Loaded centralized training data: {latest_run.get('model_name', 'Unknown')}")

        print(f"[DEBUG] Checking centralized test: {centralized_test_path}")
        print(f"[DEBUG] Exists: {os.path.exists(centralized_test_path)}")

        if os.path.exists(centralized_test_path):
            with open(centralized_test_path, 'r') as f:
                centralized_test = json.load(f)
                if centralized_test:
                    latest_test = centralized_test[-1] if isinstance(centralized_test, list) else centralized_test
                    comparison_data['centralized']['testing'] = latest_test
                    print(f"[DEBUG] Loaded centralized test data: {latest_test.get('model_name', 'Unknown')}")

        # Load federated data
        federated_history_path = os.path.join(RESULTS_DIR, 'federated_training_history.json')
        federated_test_path = os.path.join(RESULTS_DIR, 'federated_test_metrics.json')

        print(f"[DEBUG] Checking federated history: {federated_history_path}")
        print(f"[DEBUG] Exists: {os.path.exists(federated_history_path)}")

        if os.path.exists(federated_history_path):
            with open(federated_history_path, 'r') as f:
                federated_history = json.load(f)
                if federated_history:
                    latest_run = federated_history[-1]
                    comparison_data['federated']['training'] = latest_run
                    print(f"[DEBUG] Loaded federated training data: {latest_run.get('model_name', 'Unknown')}")

        print(f"[DEBUG] Checking federated test: {federated_test_path}")
        print(f"[DEBUG] Exists: {os.path.exists(federated_test_path)}")

        if os.path.exists(federated_test_path):
            with open(federated_test_path, 'r') as f:
                federated_test = json.load(f)
                if federated_test:
                    latest_test = federated_test[-1] if isinstance(federated_test, list) else federated_test
                    comparison_data['federated']['testing'] = latest_test
                    print(f"[DEBUG] Loaded federated test data: {latest_test.get('model_name', 'Unknown')}")

        print(f"[DEBUG] Final comparison data structure: {list(comparison_data.keys())}")
        print(f"[DEBUG] Centralized keys: {list(comparison_data['centralized'].keys())}")
        print(f"[DEBUG] Federated keys: {list(comparison_data['federated'].keys())}")

        # Calculate comparison metrics
        if comparison_data['centralized'].get('testing') and comparison_data['federated'].get('testing'):
            cent_test = comparison_data['centralized']['testing']
            fed_test = comparison_data['federated']['testing']

            comparison_data['comparison_metrics'] = {
                'performance_gap': {
                    'test_loss_diff': fed_test.get('test_loss', 0) - cent_test.get('test_loss', 0),
                    'ade_diff': fed_test.get('test_min_ade_k1', 0) - cent_test.get('test_min_ade_k1', 0),
                    'fde_diff': fed_test.get('test_min_fde_k1', 0) - cent_test.get('test_min_fde_k1', 0),
                    'mr_diff': fed_test.get('test_mr_2m', 0) - cent_test.get('test_mr_2m', 0)
                },
                'privacy_benefit': 'High - No raw data sharing in federated approach',
                'communication_overhead': 'Low - Only model parameters shared',
                'scalability': 'High - Can add more clients without data centralization'
            }

        return jsonify({"status": "success", "data": comparison_data})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_visualizations/<training_type>/<model_name>', methods=['GET'])
def get_visualizations(training_type, model_name):
    if training_type == 'centralized':
        viz_dir = os.path.join(RESULTS_DIR, 'test_predictions', 'centralized', model_name)
    elif training_type == 'federated':
        viz_dir = os.path.join(RESULTS_DIR, 'test_predictions', 'federated', model_name)
    else:
        return jsonify({"status": "error", "message": "Invalid training type"}), 400

    # Create directory if it doesn't exist
    os.makedirs(viz_dir, exist_ok=True)

    if os.path.exists(viz_dir):
        # Get all visualization files
        all_files = os.listdir(viz_dir)

        visualizations = {
            'static_images': [],
            'animations': [],
            'consolidated_data': None
        }

        for file in all_files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join('results', 'test_predictions', training_type, model_name, file).replace('\\', '/')

                # Extract sample number for sorting
                sample_num = 0
                try:
                    # Try different patterns to extract sample number
                    import re
                    # Pattern for test_sample_X_prediction.png or sample_X.png
                    match = re.search(r'(?:test_)?sample_(\d+)', file)
                    if match:
                        sample_num = int(match.group(1))
                    else:
                        # Try to find any number in filename
                        numbers = re.findall(r'\d+', file)
                        if numbers:
                            sample_num = int(numbers[0])
                except:
                    pass

                visualizations['static_images'].append({
                    'path': file_path,
                    'filename': file,
                    'sample_id': sample_num,
                    'type': 'prediction' if 'prediction' in file else 'trajectory'
                })

            elif file.endswith('.gif'):
                file_path = os.path.join('results', 'test_predictions', training_type, model_name, file).replace('\\', '/')

                # Extract sample number for animations too
                sample_num = 0
                try:
                    import re
                    match = re.search(r'(?:test_)?sample_(\d+)', file)
                    if match:
                        sample_num = int(match.group(1))
                except:
                    pass

                visualizations['animations'].append({
                    'path': file_path,
                    'filename': file,
                    'sample_id': sample_num,
                    'type': 'animation'
                })

            elif file == 'consolidated_predictions.json':
                consolidated_path = os.path.join(viz_dir, file)
                try:
                    with open(consolidated_path, 'r') as f:
                        visualizations['consolidated_data'] = json.load(f)
                except Exception as e:
                    print(f"Error loading consolidated data: {e}")
                    pass

        # Sort by sample ID
        visualizations['static_images'].sort(key=lambda x: x['sample_id'])
        visualizations['animations'].sort(key=lambda x: x['sample_id'])

        print(f"[INFO] Found {len(visualizations['static_images'])} images and {len(visualizations['animations'])} animations for {training_type} {model_name}")
        print(f"[DEBUG] Static images: {[img['filename'] for img in visualizations['static_images']]}")
        print(f"[DEBUG] Animations: {[anim['filename'] for anim in visualizations['animations']]}")
        print(f"[DEBUG] Consolidated data available: {visualizations['consolidated_data'] is not None}")

        return jsonify({"status": "success", "visualizations": visualizations})
    else:
        return jsonify({"status": "error", "message": f"Visualization directory not found: {viz_dir}"}), 404

@app.route('/get_training_progress', methods=['GET'])
def get_training_progress():
    """Get real-time training progress for all running processes"""
    progress_data = {
        'running_processes': [],
        'completed_processes': [],
        'system_status': 'idle'
    }

    current_time = time.time()

    for process_id, process_info in running_processes.items():
        process_data = process_info.copy()
        process_data['process_id'] = process_id

        if process_info['status'] == 'running':
            if process_info.get('start_time'):
                process_data['elapsed_time'] = current_time - process_info['start_time']
            progress_data['running_processes'].append(process_data)
            progress_data['system_status'] = 'busy'

        elif process_info['status'] in ['completed', 'error']:
            if process_info.get('start_time') and process_info.get('end_time'):
                process_data['total_time'] = process_info['end_time'] - process_info['start_time']
            progress_data['completed_processes'].append(process_data)

    return jsonify({"status": "success", "progress": progress_data})

@app.route('/start_federated_training', methods=['POST'])
def start_federated_training():
    """Start federated training with server and clients"""
    data = request.json
    model_name = data.get('model_name', 'GATv2')
    num_clients = data.get('num_clients', 2)
    num_rounds = data.get('num_rounds', 3)
    train_dir = data.get('train_dir', 'dataset/train_small')
    val_dir = data.get('val_dir', 'dataset/val_small')
    num_scenarios = data.get('num_scenarios', 100)
    batch_size = data.get('batch_size', 32)
    epochs = data.get('epochs', 5)
    lr = data.get('lr', 1e-3)

    process_id = f"federated_{model_name}_{int(time.time())}"

    def run_federated_training():
        try:
            running_processes[process_id]['status'] = 'running'
            running_processes[process_id]['start_time'] = time.time()
            running_processes[process_id]['output'].append("[INFO] Starting federated training...")
            running_processes[process_id]['output'].append(f"[INFO] Model: {model_name}, Clients: {num_clients}, Rounds: {num_rounds}")

            # Use the same approach as run_demo.py - run in same process
            from train import federated_train_complete

            # Create output capture
            class OutputCapture:
                def __init__(self, process_id):
                    self.process_id = process_id
                    self.original_stdout = sys.stdout
                    self.original_stderr = sys.stderr

                def write(self, text):
                    # Write to original stdout
                    self.original_stdout.write(text)
                    self.original_stdout.flush()

                    # Also capture for web interface
                    if text.strip() and self.process_id in running_processes:
                        running_processes[self.process_id]['output'].append(text.strip())
                        # Keep only last 1000 lines
                        if len(running_processes[self.process_id]['output']) > 1000:
                            running_processes[self.process_id]['output'] = running_processes[self.process_id]['output'][-1000:]

                def flush(self):
                    self.original_stdout.flush()

            output_capture = OutputCapture(process_id)
            sys.stdout = output_capture
            sys.stderr = output_capture

            try:
                # Run federated training in the same process
                federated_train_complete(
                    model_name=model_name,
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    train_dir=train_dir,
                    val_dir=val_dir,
                    num_scenarios=num_scenarios,
                    batch_size=batch_size,
                    epochs=epochs,
                    lr=lr,
                    seq_len=data.get('seq_len', 30)  # Use configurable seq_len
                )

                running_processes[process_id]['output'].append("[INFO] Federated training completed successfully!")

            except Exception as e:
                error_msg = f"[ERROR] Federated training failed: {str(e)}"
                running_processes[process_id]['output'].append(error_msg)
                logger.error(error_msg)
                raise e
            finally:
                # Restore original stdout/stderr
                sys.stdout = output_capture.original_stdout
                sys.stderr = output_capture.original_stderr

            running_processes[process_id]['status'] = 'completed'
            running_processes[process_id]['end_time'] = time.time()

        except Exception as e:
            running_processes[process_id]['status'] = 'error'
            running_processes[process_id]['error'] = str(e)
            running_processes[process_id]['output'].append(f"[ERROR] {str(e)}")
            logger.error(f"Error in federated training: {e}")

    running_processes[process_id] = {
        'mode': 'federated_train',
        'model_name': model_name,
        'status': 'starting',
        'start_time': None,
        'end_time': None,
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'output': []
    }

    thread = threading.Thread(target=run_federated_training)
    thread.start()

    return jsonify({
        "status": "started",
        "message": f"Federated training started for {model_name} with {num_clients} clients",
        "process_id": process_id
    })

@app.route('/get_model_comparison/<model_name>', methods=['GET'])
def get_model_comparison(model_name):
    """Get detailed comparison for a specific model"""
    try:
        comparison = {
            'model_name': model_name,
            'centralized': {'available': False},
            'federated': {'available': False},
            'performance_comparison': {}
        }

        # Check centralized results
        cent_test_path = os.path.join(RESULTS_DIR, 'centralized_test_metrics.json')
        if os.path.exists(cent_test_path):
            with open(cent_test_path, 'r') as f:
                cent_data = json.load(f)
                for entry in (cent_data if isinstance(cent_data, list) else [cent_data]):
                    if entry.get('model_name') == model_name:
                        comparison['centralized'] = {
                            'available': True,
                            'metrics': entry,
                            'visualizations_count': len(glob.glob(os.path.join(RESULTS_DIR, 'test_predictions', 'centralized', model_name, '*.png')))
                        }
                        break

        # Check federated results
        fed_test_path = os.path.join(RESULTS_DIR, 'federated_test_metrics.json')
        if os.path.exists(fed_test_path):
            with open(fed_test_path, 'r') as f:
                fed_data = json.load(f)
                for entry in (fed_data if isinstance(fed_data, list) else [fed_data]):
                    if entry.get('model_name') == model_name:
                        comparison['federated'] = {
                            'available': True,
                            'metrics': entry,
                            'visualizations_count': len(glob.glob(os.path.join(RESULTS_DIR, 'test_predictions', 'federated', model_name, '*.png')))
                        }
                        break

        # Calculate performance comparison
        if comparison['centralized']['available'] and comparison['federated']['available']:
            cent_metrics = comparison['centralized']['metrics']
            fed_metrics = comparison['federated']['metrics']

            comparison['performance_comparison'] = {
                'loss_ratio': fed_metrics.get('test_loss', 0) / cent_metrics.get('test_loss', 1),
                'ade_ratio': fed_metrics.get('test_min_ade_k1', 0) / cent_metrics.get('test_min_ade_k1', 1),
                'fde_ratio': fed_metrics.get('test_min_fde_k1', 0) / cent_metrics.get('test_min_fde_k1', 1),
                'performance_gap_percentage': ((fed_metrics.get('test_loss', 0) - cent_metrics.get('test_loss', 0)) / cent_metrics.get('test_loss', 1)) * 100
            }

        return jsonify({"status": "success", "comparison": comparison})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve result files (images, JSONs, etc.)"""
    try:
        return send_from_directory(RESULTS_DIR, filename)
    except Exception as e:
        print(f"[ERROR] Could not serve file {filename}: {e}")
        return jsonify({"status": "error", "message": f"File not found: {filename}"}), 404

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Create model-specific directories
    for model in ['GATv2', 'VectorNet']:
        os.makedirs(os.path.join(RESULTS_DIR, 'test_predictions', 'centralized', model), exist_ok=True)
        os.makedirs(os.path.join(RESULTS_DIR, 'test_predictions', 'federated', model), exist_ok=True)

    print("🚀 Enhanced Federated Trajectory Prediction Web Interface")
    print("📊 Features: Interactive training, real-time monitoring, comprehensive comparisons")
    print("🌐 Access at: http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)