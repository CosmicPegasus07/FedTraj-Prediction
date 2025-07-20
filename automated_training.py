#!/usr/bin/env python3
"""
Automated Training and Testing Script
Runs complete training and testing pipeline for both GATv2 and VectorNet models
in both centralized and federated modes without user intervention.
"""

import subprocess
import time
import os
import sys
from datetime import datetime
import json

def wait_for_completion(completion_file, timeout=10800, check_interval=10):
    """
    Wait for a completion file to be created with 'completed' content
    
    Args:
        completion_file: Path to the completion file to watch
        timeout: Maximum time to wait in seconds (default: 3 hours)
        check_interval: How often to check for the file in seconds (default: 10 seconds)
    
    Returns:
        bool: True if completed successfully, False if timed out
    """
    start_time = time.time()
    print(f"⏳ Waiting for completion marker: {completion_file}")
    
    while time.time() - start_time < timeout:
        if os.path.exists(completion_file):
            try:
                with open(completion_file, 'r') as f:
                    content = f.read().strip()
                if content == "completed":
                    print(f"✅ Completion marker found: {completion_file}")
                    # Clean up the completion file
                    os.remove(completion_file)
                    return True
            except Exception as e:
                print(f"⚠️ Error reading completion file: {e}")
        
        elapsed = int(time.time() - start_time)
        print(f"⏳ Waiting... ({elapsed}s elapsed)")
        time.sleep(check_interval)
    
    print(f"⚠️ Timeout waiting for completion marker after {timeout} seconds")
    return False

class AutomatedTraining:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models = ["GATv2", "VectorNet"]
        self.config = {}
        self.results_log = []
        
    def get_user_config(self):
        """Get all configuration from user upfront"""
        print("🚀 Automated Training and Testing Configuration")
        print("=" * 60)
        
        # Dataset paths
        print("\n📁 Dataset Configuration:")
        self.config['train_dir'] = input("Train directory [dataset/train_small]: ").strip() or "dataset/train_small"
        self.config['val_dir'] = input("Validation directory [dataset/val_small]: ").strip() or "dataset/val_small"
        self.config['test_dir'] = input("Test directory [dataset/test_small]: ").strip() or "dataset/test_small"
        
        # Training parameters
        print("\n⚙️ Training Parameters:")
        self.config['num_scenarios'] = int(input("Number of scenarios [100]: ").strip() or "100")
        self.config['batch_size'] = int(input("Batch size [16]: ").strip() or "16")
        self.config['seq_len'] = int(input("Sequence length [30]: ").strip() or "30")
        self.config['epochs'] = int(input("Training epochs [5]: ").strip() or "5")
        self.config['lr'] = float(input("Learning rate [1e-3]: ").strip() or "1e-3")
        
        # Testing parameters
        print("\n🧪 Testing Parameters:")
        self.config['test_scenarios'] = int(input("Test scenarios [10]: ").strip() or "10")
        self.config['visualize_limit'] = int(input("Visualizations to generate [5]: ").strip() or "5")
        
        # Federated parameters
        print("\n🌐 Federated Learning Parameters:")
        self.config['num_clients'] = int(input("Number of clients [2]: ").strip() or "2")
        self.config['num_rounds'] = int(input("Federated rounds [3]: ").strip() or "3")
        self.config['client_epochs'] = int(input("Client epochs per round [1]: ").strip() or "1")
        
        # Execution options
        print("\n🎯 Execution Options:")
        print("1. Full pipeline (Centralized train/test + Federated train/test)")
        print("2. Centralized only (train + test)")
        print("3. Federated only (train + test)")
        print("4. Testing only (both centralized and federated)")
        
        choice = input("Select option [1]: ").strip() or "1"
        self.config['execution_mode'] = int(choice)
        
        # Confirmation
        print("\n📋 Configuration Summary:")
        print(f"Models: {', '.join(self.models)}")
        print(f"Train dir: {self.config['train_dir']}")
        print(f"Test dir: {self.config['test_dir']}")
        print(f"Scenarios: {self.config['num_scenarios']} (train), {self.config['test_scenarios']} (test)")
        print(f"Training: {self.config['epochs']} epochs, batch size {self.config['batch_size']}, seq_len {self.config['seq_len']}")
        print(f"Federated: {self.config['num_clients']} clients, {self.config['num_rounds']} rounds")
        print(f"Visualizations: {self.config['visualize_limit']} per test")
        
        confirm = input("\nProceed with this configuration? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("❌ Configuration cancelled.")
            sys.exit(0)
            
        # Save configuration
        config_path = os.path.join(self.base_dir, 'automated_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✅ Configuration saved to: {config_path}")
        
    def log_result(self, stage, model, mode, status, duration=None, error=None):
        """Log training/testing results"""
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'stage': stage,
            'model': model,
            'mode': mode,
            'status': status,
            'duration': duration,
            'error': error
        }
        self.results_log.append(result)
        
        # Also print to console
        status_icon = "✅" if status == "success" else "❌"
        print(f"{status_icon} {stage} - {model} ({mode}): {status}")
        if duration:
            print(f"   Duration: {duration:.1f} seconds")
        if error:
            print(f"   Error: {error}")
            
    def run_centralized_training(self, model_name):
        """Run centralized training for a model"""
        print(f"\n🏋️ Starting Centralized Training - {model_name}")
        print("-" * 50)
        
        start_time = time.time()
        try:
            # Import and run centralized training
            from train import centralized_train
            
            centralized_train(
                train_dir=self.config['train_dir'],
                val_dir=self.config['val_dir'],
                batch_size=self.config['batch_size'],
                num_scenarios=self.config['num_scenarios'],
                epochs=self.config['epochs'],
                lr=self.config['lr'],
                model_name=model_name,
                seq_len=self.config['seq_len']
            )
            
            duration = time.time() - start_time
            self.log_result("Training", model_name, "Centralized", "success", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Training", model_name, "Centralized", "failed", duration, str(e))
            print(f"❌ Centralized training failed for {model_name}: {e}")
            return False
            
    def run_centralized_testing(self, model_name):
        """Run centralized testing for a model"""
        print(f"\n🧪 Starting Centralized Testing - {model_name}")
        print("-" * 50)
        
        start_time = time.time()
        try:
            # Import and run centralized testing
            from train import centralized_test
            
            centralized_test(
                test_dir=self.config['test_dir'],
                batch_size=self.config['batch_size'],
                num_scenarios=self.config['test_scenarios'],
                visualize_limit=self.config['visualize_limit'],
                model_name=model_name,
                seq_len=self.config['seq_len']
            )
            
            duration = time.time() - start_time
            self.log_result("Testing", model_name, "Centralized", "success", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Testing", model_name, "Centralized", "failed", duration, str(e))
            print(f"❌ Centralized testing failed for {model_name}: {e}")
            return False
            
    def run_federated_training(self, model_name):
        """Run federated training for a model using completion file tracking"""
        print(f"\n🌐 Starting Federated Training - {model_name}")
        print("-" * 50)
        
        start_time = time.time()
        try:
            # Create completion file path
            completion_file = os.path.join("temp", "federated_training_completed.txt")
            
            # Remove any existing completion file
            if os.path.exists(completion_file):
                os.remove(completion_file)
            
            # Start server process in new window
            print(f"🖥️ Starting federated server for {model_name}...")
            server_cmd = f"python -m federated.server --rounds {self.config['num_rounds']} --client_epochs {self.config['client_epochs']} --model_name {model_name} --seq_len {self.config['seq_len']}"
            server_process = subprocess.Popen(f'start cmd /k {server_cmd}', shell=True)
            
            # Wait for server to initialize
            print("⏳ Waiting for server to initialize...")
            time.sleep(15)
            
            # Start client processes in new windows
            client_processes = []
            for client_id in range(self.config['num_clients']):
                print(f"👤 Starting client {client_id + 1}/{self.config['num_clients']}...")
                
                client_cmd = (
                    f"python -m federated.client --train_dir {self.config['train_dir']} "
                    f"--val_dir {self.config['val_dir']} --num_scenarios {self.config['num_scenarios']} "
                    f"--batch_size {self.config['batch_size']} --seq_len {self.config['seq_len']} "
                    f"--client_id {client_id} --num_clients {self.config['num_clients']} "
                    f"--model_name {model_name}"
                )
                
                client_process = subprocess.Popen(f'start cmd /k {client_cmd}', shell=True)
                client_processes.append(client_process)
                
                # Stagger client starts
                time.sleep(5)
            
            print(f"🔄 Federated training in progress for {model_name}...")
            print("   Server and clients are running in new terminal windows...")
            
            # Wait for completion marker instead of managing processes
            print("⏳ Waiting for training completion...")
            if wait_for_completion(completion_file, timeout=10800):  # 3 hours timeout
                duration = time.time() - start_time
                self.log_result("Training", model_name, "Federated", "success", duration)
                print(f"✅ Federated training completed successfully for {model_name}")
                return True
            else:
                duration = time.time() - start_time
                self.log_result("Training", model_name, "Federated", "failed", duration, "Timeout waiting for completion")
                print(f"❌ Federated training failed for {model_name} (timeout)")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Training", model_name, "Federated", "failed", duration, str(e))
            print(f"❌ Federated training failed for {model_name}: {e}")
            
            # Clean up processes
            try:
                server_process.terminate()
                for client_process in client_processes:
                    client_process.terminate()
            except:
                pass
                
            return False
            
    def run_federated_testing(self, model_name):
        """Run federated testing for a model"""
        print(f"\n🧪 Starting Federated Testing - {model_name}")
        print("-" * 50)
        
        start_time = time.time()
        try:
            # Import and run federated testing
            from train import federated_test
            
            federated_test(
                test_dir=self.config['test_dir'],
                batch_size=self.config['batch_size'],
                num_scenarios=self.config['test_scenarios'],
                visualize_limit=self.config['visualize_limit'],
                model_name=model_name,
                seq_len=30
            )
            
            duration = time.time() - start_time
            self.log_result("Testing", model_name, "Federated", "success", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Testing", model_name, "Federated", "failed", duration, str(e))
            print(f"❌ Federated testing failed for {model_name}: {e}")
            return False

    def run_full_pipeline(self):
        """Run the complete training and testing pipeline"""
        print("\n🚀 Starting Automated Training Pipeline")
        print("=" * 60)

        pipeline_start = time.time()
        total_success = 0
        total_tasks = 0

        execution_mode = self.config['execution_mode']

        # Phase 1: Centralized Training and Testing
        if execution_mode in [1, 2]:
            print("\n" + "="*60)
            print("🏢 PHASE 1: CENTRALIZED TRAINING AND TESTING")
            print("="*60)

            for model in self.models:
                print(f"\n🎯 Processing Model: {model}")
                print("-" * 40)

                # Centralized Training
                print(f"Step 1/2: Training {model} (Centralized)")
                success = self.run_centralized_training(model)
                total_tasks += 1
                if success:
                    total_success += 1

                    # Only test if training succeeded
                    print(f"Step 2/2: Testing {model} (Centralized)")
                    success = self.run_centralized_testing(model)
                    total_tasks += 1
                    if success:
                        total_success += 1
                else:
                    print(f"⚠️ Skipping testing for {model} due to training failure")
                    total_tasks += 1  # Count the skipped test

                # Cool down between models
                if model != self.models[-1]:
                    print("😴 Cooling down for 30 seconds...")
                    time.sleep(30)

        # Phase 2: Federated Training and Testing
        if execution_mode in [1, 3]:
            print("\n" + "="*60)
            print("🌐 PHASE 2: FEDERATED TRAINING AND TESTING")
            print("="*60)

            for model in self.models:
                print(f"\n🎯 Processing Model: {model}")
                print("-" * 40)

                # Federated Training
                print(f"Step 1/2: Training {model} (Federated)")
                success = self.run_federated_training(model)
                total_tasks += 1
                if success:
                    total_success += 1

                    # Cool down after federated training
                    print("😴 Cooling down for 30 seconds after federated training...")
                    time.sleep(30)

                    # Only test if training succeeded
                    print(f"Step 2/2: Testing {model} (Federated)")
                    success = self.run_federated_testing(model)
                    total_tasks += 1
                    if success:
                        total_success += 1
                else:
                    print(f"⚠️ Skipping testing for {model} due to training failure")
                    total_tasks += 1  # Count the skipped test

                # Cool down between models
                if model != self.models[-1]:
                    print("😴 Cooling down for 30 seconds...")
                    time.sleep(30)

        # Phase 3: Testing Only
        if execution_mode == 4:
            print("\n" + "="*60)
            print("🧪 PHASE: TESTING ONLY")
            print("="*60)

            for model in self.models:
                print(f"\n🎯 Testing Model: {model}")
                print("-" * 40)

                # Centralized Testing
                print(f"Step 1/2: Testing {model} (Centralized)")
                success = self.run_centralized_testing(model)
                total_tasks += 1
                if success:
                    total_success += 1

                # Federated Testing
                print(f"Step 2/2: Testing {model} (Federated)")
                success = self.run_federated_testing(model)
                total_tasks += 1
                if success:
                    total_success += 1

                # Cool down between models
                if model != self.models[-1]:
                    print("😴 Cooling down for 15 seconds...")
                    time.sleep(15)

        # Final Summary
        pipeline_duration = time.time() - pipeline_start
        self.print_final_summary(total_success, total_tasks, pipeline_duration)

    def print_final_summary(self, total_success, total_tasks, pipeline_duration):
        """Print comprehensive final summary"""
        print("\n" + "="*80)
        print("🎉 AUTOMATED PIPELINE COMPLETED")
        print("="*80)

        print(f"\n📊 Overall Statistics:")
        print(f"   Total Tasks: {total_tasks}")
        print(f"   Successful: {total_success}")
        print(f"   Failed: {total_tasks - total_success}")
        print(f"   Success Rate: {(total_success/total_tasks)*100:.1f}%")
        print(f"   Total Duration: {pipeline_duration/3600:.1f} hours ({pipeline_duration/60:.1f} minutes)")

        print(f"\n📋 Detailed Results:")
        print("-" * 80)
        print(f"{'Stage':<12} {'Model':<10} {'Mode':<12} {'Status':<10} {'Duration':<12} {'Error'}")
        print("-" * 80)

        for result in self.results_log:
            duration_str = f"{result['duration']:.1f}s" if result['duration'] else "N/A"
            error_str = result['error'][:30] + "..." if result['error'] and len(result['error']) > 30 else (result['error'] or "")
            print(f"{result['stage']:<12} {result['model']:<10} {result['mode']:<12} {result['status']:<10} {duration_str:<12} {error_str}")

        # Save detailed log
        log_path = os.path.join(self.base_dir, f"automated_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_path, 'w') as f:
            json.dump({
                'config': self.config,
                'results': self.results_log,
                'summary': {
                    'total_tasks': total_tasks,
                    'successful': total_success,
                    'failed': total_tasks - total_success,
                    'success_rate': (total_success/total_tasks)*100,
                    'duration_hours': pipeline_duration/3600
                }
            }, f, indent=2)

        print(f"\n💾 Detailed log saved to: {log_path}")

        # Final recommendations
        print(f"\n🎯 Next Steps:")
        if total_success == total_tasks:
            print("   ✅ All tasks completed successfully!")
            print("   🌐 Open the web interface to view results: python app.py")
            print("   📊 Check performance comparisons and visualizations")
        else:
            print("   ⚠️ Some tasks failed. Check the detailed log above.")
            print("   🔧 Review error messages and fix issues before retrying")
            print("   🌐 You can still view successful results in the web interface")

        print(f"\n🏁 Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main execution function"""
    try:
        trainer = AutomatedTraining()
        trainer.get_user_config()

        print(f"\n⏰ Starting pipeline at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🚨 WARNING: This will run for several hours. Do not close this terminal!")
        print("💻 You can safely leave your laptop - the script will run automatically.")

        input("\nPress Enter to start the automated pipeline...")

        trainer.run_full_pipeline()

    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user!")
        print("🛑 Stopping all processes...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
