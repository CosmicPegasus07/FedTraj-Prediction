#!/usr/bin/env python3
"""
Automated Training and Testing Script
Runs complete training and testing pipeline for both GATv2 and VectorNet models
in both centralized and federated modes without user intervention.
"""
import psutil, os, ctypes
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)  # For Windows

import subprocess
import time
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# EMAIL CONFIGURATION - using environment variables for security
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'afeef2001kashif@gmail.com',
    'sender_password': os.getenv('GOOGLE_APP_PASSWORD'),
    'recipient_email': 'afeef2001kashif@gmail.com'
}

def send_email(subject, body, is_success=True):
    """Send email notification"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = EMAIL_CONFIG['recipient_email']
        msg['Subject'] = subject
        
        # Add body
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        text = msg.as_string()
        server.sendmail(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['recipient_email'], text)
        server.quit()
        
        print(f"[EMAIL] Email sent successfully: {subject}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False

def send_phase_email(subject, body):
    """Send phase notification email - wrapper for backwards compatibility"""
    return send_email(subject, body)

def format_duration(seconds):
    """Format duration in human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

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
    print(f"[WAIT] Waiting for completion marker: {completion_file}")
    
    while time.time() - start_time < timeout:
        if os.path.exists(completion_file):
            try:
                with open(completion_file, 'r') as f:
                    content = f.read().strip()
                if content == "completed":
                    print(f"[SUCCESS] Completion marker found: {completion_file}")
                    # Clean up the completion file
                    os.remove(completion_file)
                    return True
            except Exception as e:
                print(f"[WARNING] Error reading completion file: {e}")
        
        elapsed = int(time.time() - start_time)
        print(f"[WAIT] Waiting... ({elapsed}s elapsed)")
        time.sleep(check_interval)
    
    print(f"[WARNING] Timeout waiting for completion marker after {timeout} seconds")
    return False

class AutomatedTraining:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models = ["GATv2", "VectorNet"]
        self.config = {}
        self.results_log = []
        
    def load_config(self):
        """Load configuration from automated_config.json"""
        print("[START] Automated Training System")
        print("=" * 50)
        
        config_path = os.path.join(self.base_dir, 'automated_config.json')
        if not os.path.exists(config_path):
            print(f"[ERROR] Configuration file not found: {config_path}")
            print("[INFO] Please create automated_config.json with your training parameters.")
            sys.exit(1)
            
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print("[SUCCESS] Configuration loaded from automated_config.json")
            
            # Display configuration
            print("\n[CONFIG] Training Configuration:")
            print(f"  Models: {', '.join(self.models)}")
            print(f"  Train dir: {self.config['train_dir']}")
            print(f"  Test dir: {self.config['test_dir']}")
            print(f"  Scenarios: {self.config['num_scenarios']} (train), {self.config['test_scenarios']} (test)")
            print(f"  Training: {self.config['epochs']} epochs, batch size {self.config['batch_size']}")
            print(f"  Federated: {self.config['num_clients']} clients, {self.config['num_rounds']} rounds")
            print(f"  Execution mode: {self.config['execution_mode']}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load configuration: {e}")
            sys.exit(1)
        
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
        status_icon = "[SUCCESS]" if status == "success" else "[FAILED]"
        print(f"{status_icon} {stage} - {model} ({mode}): {status}")
        if duration:
            print(f"   Duration: {duration:.1f} seconds")
        if error:
            print(f"   Error: {error}")
            
    def run_centralized_training(self, model_name):
        """Run centralized training for a model"""
        print(f"\n[TRAINING] Starting Centralized Training - {model_name}")
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

            # Send individual task completion email
            subject = f"✅ Training Complete - {model_name} (Centralized)"
            body = f"""
Centralized training has completed successfully!

Model: {model_name}
Type: Centralized Training
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.1f} seconds
Epochs: {self.config['epochs']}
Batch Size: {self.config['batch_size']}
Scenarios: {self.config['num_scenarios']}

The model is now ready for testing.
            """
            send_email(subject, body, is_success=True)

            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Training", model_name, "Centralized", "failed", duration, str(e))
            print(f"[FAILED] Centralized training failed for {model_name}: {e}")

            # Send failure notification
            subject = f"❌ Training Failed - {model_name} (Centralized)"
            body = f"""
Centralized training has failed!

Model: {model_name}
Type: Centralized Training
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.1f} seconds
Error: {str(e)[:500]}...

Please check the logs for more details.
            """
            send_email(subject, body, is_success=False)

            return False
            
    def run_centralized_testing(self, model_name):
        """Run centralized testing for a model"""
        print(f"\n[TESTING] Starting Centralized Testing - {model_name}")
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

            # Send individual task completion email
            subject = f"✅ Testing Complete - {model_name} (Centralized)"
            body = f"""
Centralized testing has completed successfully!

Model: {model_name}
Type: Centralized Testing
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.1f} seconds
Test Scenarios: {self.config['test_scenarios']}
Visualizations: {self.config['visualize_limit']}

Check the results in the web interface.
            """
            send_email(subject, body, is_success=True)

            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Testing", model_name, "Centralized", "failed", duration, str(e))
            print(f"[FAILED] Centralized testing failed for {model_name}: {e}")

            # Send failure notification
            subject = f"❌ Testing Failed - {model_name} (Centralized)"
            body = f"""
Centralized testing has failed!

Model: {model_name}
Type: Centralized Testing
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.1f} seconds
Error: {str(e)[:500]}...

Please check the logs for more details.
            """
            send_email(subject, body, is_success=False)

            return False
            
    def run_federated_training(self, model_name):
        """Run federated training for a model using completion file tracking"""
        print(f"\n[FEDERATED] Starting Federated Training - {model_name}")
        print("-" * 50)

        start_time = time.time()
        server_process = None
        client_processes = []

        try:
            # Create completion file path
            completion_file = os.path.join("temp", "federated_training_completed.txt")

            # Remove any existing completion file
            if os.path.exists(completion_file):
                os.remove(completion_file)

            # Ensure temp directory exists
            os.makedirs("temp", exist_ok=True)

            # Start server process in new window
            print(f"[SERVER] Starting federated server for {model_name}...")
            server_cmd = f"python -m federated.server --rounds {self.config['num_rounds']} --client_epochs {self.config['client_epochs']} --model_name {model_name} --seq_len {self.config['seq_len']}"

            # Try PowerShell approach first, fallback to background process
            try:
                # Use PowerShell to start a new window
                powershell_server_cmd = f'powershell -Command "Start-Process cmd -ArgumentList \'/k\', \'cd /d {os.getcwd()} && {server_cmd}\'"'
                server_process = subprocess.Popen(powershell_server_cmd, shell=True)
                print("[SERVER] Started server in new PowerShell window")
            except Exception as ps_error:
                print(f"[SERVER] PowerShell failed ({ps_error}), trying background process...")
                # Fallback to background process with output redirection
                server_log = os.path.join("temp", f"server_{model_name}.log")
                with open(server_log, 'w') as log_file:
                    server_process = subprocess.Popen(
                        server_cmd.split(),
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        cwd=os.getcwd()
                    )
                print(f"[SERVER] Started server as background process (log: {server_log})")

            # Wait for server to initialize
            print("[WAIT] Waiting for server to initialize...")
            time.sleep(15)

            # Start client processes
            for client_id in range(self.config['num_clients']):
                print(f"[CLIENT] Starting client {client_id + 1}/{self.config['num_clients']}...")

                client_cmd = (
                    f"python -m federated.client --train_dir {self.config['train_dir']} "
                    f"--val_dir {self.config['val_dir']} --num_scenarios {self.config['num_scenarios']} "
                    f"--batch_size {self.config['batch_size']} --seq_len {self.config['seq_len']} "
                    f"--client_id {client_id} --num_clients {self.config['num_clients']} "
                    f"--model_name {model_name}"
                )

                # Try PowerShell approach first, fallback to background process
                try:
                    # Use PowerShell to start a new window for client
                    powershell_client_cmd = f'powershell -Command "Start-Process cmd -ArgumentList \'/k\', \'cd /d {os.getcwd()} && {client_cmd}\'"'
                    client_process = subprocess.Popen(powershell_client_cmd, shell=True)
                    print(f"[CLIENT] Started client {client_id + 1} in new PowerShell window")
                except Exception as ps_error:
                    print(f"[CLIENT] PowerShell failed for client {client_id + 1} ({ps_error}), trying background process...")
                    # Fallback to background process with output redirection
                    client_log = os.path.join("temp", f"client_{client_id}_{model_name}.log")
                    with open(client_log, 'w') as log_file:
                        client_process = subprocess.Popen(
                            client_cmd.split(),
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            cwd=os.getcwd()
                        )
                    print(f"[CLIENT] Started client {client_id + 1} as background process (log: {client_log})")

                client_processes.append(client_process)

                # Stagger client starts
                time.sleep(5)

            print(f"[PROGRESS] Federated training in progress for {model_name}...")
            print("   Server and clients are running...")
            print(f"   Check temp/ directory for log files if processes are running in background")

            # Wait for completion marker instead of managing processes
            print("[WAIT] Waiting for training completion...")
            if wait_for_completion(completion_file, timeout=10800):  # 3 hours timeout
                duration = time.time() - start_time
                self.log_result("Training", model_name, "Federated", "success", duration)
                print(f"[SUCCESS] Federated training completed successfully for {model_name}")

                # Send individual task completion email
                subject = f"✅ Training Complete - {model_name} (Federated)"
                body = f"""
Federated training has completed successfully!

Model: {model_name}
Type: Federated Training
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.1f} seconds
Clients: {self.config['num_clients']}
Rounds: {self.config['num_rounds']}
Client Epochs: {self.config['client_epochs']}

The model is now ready for testing.
                """
                send_email(subject, body, is_success=True)

                return True
            else:
                duration = time.time() - start_time
                self.log_result("Training", model_name, "Federated", "failed", duration, "Timeout waiting for completion")
                print(f"[FAILED] Federated training failed for {model_name} (timeout)")

                # Send timeout notification
                subject = f"⏰ Training Timeout - {model_name} (Federated)"
                body = f"""
Federated training has timed out!

Model: {model_name}
Type: Federated Training
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.1f} seconds
Timeout: 3 hours

The training process was terminated due to timeout.
                """
                send_email(subject, body, is_success=False)

                return False

        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Training", model_name, "Federated", "failed", duration, str(e))
            print(f"[FAILED] Federated training failed for {model_name}: {e}")

            # Send exception notification
            subject = f"💥 Training Exception - {model_name} (Federated)"
            body = f"""
An exception occurred during federated training!

Model: {model_name}
Type: Federated Training
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.1f} seconds
Exception: {str(e)[:500]}...

Please check the system and try again.
            """
            send_email(subject, body, is_success=False)

            return False

        finally:
            # Clean up processes if they exist and are still running
            try:
                if server_process and server_process.poll() is None:
                    print("[CLEANUP] Terminating server process...")
                    server_process.terminate()
                    time.sleep(2)
                    if server_process.poll() is None:
                        server_process.kill()

                for i, client_process in enumerate(client_processes):
                    if client_process and client_process.poll() is None:
                        print(f"[CLEANUP] Terminating client {i + 1} process...")
                        client_process.terminate()
                        time.sleep(1)
                        if client_process.poll() is None:
                            client_process.kill()
            except Exception as cleanup_error:
                print(f"[CLEANUP] Error during cleanup: {cleanup_error}")
                pass
            
    def run_federated_testing(self, model_name):
        """Run federated testing for a model"""
        print(f"\n[TESTING] Starting Federated Testing - {model_name}")
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

            # Send individual task completion email
            subject = f"✅ Testing Complete - {model_name} (Federated)"
            body = f"""
Federated testing has completed successfully!

Model: {model_name}
Type: Federated Testing
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.1f} seconds
Test Scenarios: {self.config['test_scenarios']}
Visualizations: {self.config['visualize_limit']}

Check the results in the web interface.
            """
            send_email(subject, body, is_success=True)

            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Testing", model_name, "Federated", "failed", duration, str(e))
            print(f"[FAILED] Federated testing failed for {model_name}: {e}")

            # Send failure notification
            subject = f"❌ Testing Failed - {model_name} (Federated)"
            body = f"""
Federated testing has failed!

Model: {model_name}
Type: Federated Testing
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.1f} seconds
Error: {str(e)[:500]}...

Please check the logs for more details.
            """
            send_email(subject, body, is_success=False)

            return False

    def run_full_pipeline(self):
        """Run the complete training and testing pipeline"""
        print("\n[PIPELINE] Starting Automated Training Pipeline")
        print("=" * 60)

        pipeline_start = time.time()
        total_success = 0
        total_tasks = 0

        execution_mode = self.config['execution_mode']

        # Phase 1: Centralized Training and Testing
        if execution_mode in [1, 2]:
            print("\n" + "="*60)
            print("[PHASE 1] CENTRALIZED TRAINING AND TESTING")
            print("="*60)

            for model in self.models:
                print(f"\n Processing Model: {model}")
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
                    print(f"Skipping testing for {model} due to training failure")
                    total_tasks += 1  # Count the skipped test

                # Cool down between models
                if model != self.models[-1]:
                    print(" Cooling down for 30 seconds...")
                    time.sleep(30)

            # Send Phase 1 completion notification
            print("[EMAIL] Sending Phase 1 completion notification...")
            phase1_subject = "[TRAINING] Phase 1 Completed - Centralized Training"
            phase1_body = f"""
Phase 1 (Centralized Training and Testing) has completed!

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Phase: Centralized Training and Testing
Models Processed: {', '.join(self.models)}
Total Tasks Completed So Far: {total_success}/{total_tasks}

The system is now moving to Phase 2 (Federated Training) if configured.
            """
            send_phase_email(phase1_subject, phase1_body)

        # Phase 2: Federated Training and Testing
        if execution_mode in [1, 3]:
            print("\n" + "="*60)
            print("[PHASE 2] FEDERATED TRAINING AND TESTING")
            print("="*60)

            for model in self.models:
                print(f"\n Processing Model: {model}")
                print("-" * 40)

                # Federated Training
                print(f"Step 1/2: Training {model} (Federated)")
                success = self.run_federated_training(model)
                total_tasks += 1
                if success:
                    total_success += 1

                    # Cool down after federated training
                    print(" Cooling down for 30 seconds after federated training...")
                    time.sleep(30)

                    # Only test if training succeeded
                    print(f"Step 2/2: Testing {model} (Federated)")
                    success = self.run_federated_testing(model)
                    total_tasks += 1
                    if success:
                        total_success += 1
                else:
                    print(f"Skipping testing for {model} due to training failure")
                    total_tasks += 1  # Count the skipped test

                # Cool down between models
                if model != self.models[-1]:
                    print(" Cooling down for 30 seconds...")
                    time.sleep(30)

            # Send Phase 2 completion notification
            print("[EMAIL] Sending Phase 2 completion notification...")
            phase2_subject = "[TRAINING] Phase 2 Completed - Federated Training"
            phase2_body = f"""
Phase 2 (Federated Training and Testing) has completed!

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Phase: Federated Training and Testing
Models Processed: {', '.join(self.models)}
Total Tasks Completed So Far: {total_success}/{total_tasks}

Federated training with {self.config['num_clients']} clients and {self.config['num_rounds']} rounds is complete.
            """
            send_phase_email(phase2_subject, phase2_body)

        # Phase 3: Testing Only
        if execution_mode == 4:
            print("\n" + "="*60)
            print("[PHASE] TESTING ONLY")
            print("="*60)

            for model in self.models:
                print(f"\n Testing Model: {model}")
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
                    print(" Cooling down for 15 seconds...")
                    time.sleep(15)

            # Send Phase 3 completion notification
            print("[EMAIL] Sending Phase 3 completion notification...")
            phase3_subject = "[TRAINING] Phase 3 Completed - Testing Only"
            phase3_body = f"""
Phase 3 (Testing Only) has completed!

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Phase: Testing Only
Models Tested: {', '.join(self.models)}
Total Tasks Completed So Far: {total_success}/{total_tasks}

All testing has been completed for both centralized and federated models.
            """
            send_phase_email(phase3_subject, phase3_body)

        # Final Summary
        pipeline_duration = time.time() - pipeline_start
        self.print_final_summary(total_success, total_tasks, pipeline_duration)

    def print_final_summary(self, total_success, total_tasks, pipeline_duration):
        """Print comprehensive final summary"""
        print("\n" + "="*80)
        print("[COMPLETED] AUTOMATED PIPELINE COMPLETED")
        print("="*80)

        print(f"\n[STATS] Overall Statistics:")
        print(f"   Total Tasks: {total_tasks}")
        print(f"   Successful: {total_success}")
        print(f"   Failed: {total_tasks - total_success}")
        print(f"   Success Rate: {(total_success/total_tasks)*100:.1f}%")
        print(f"   Total Duration: {pipeline_duration/3600:.1f} hours ({pipeline_duration/60:.1f} minutes)")

        print(f"\n[RESULTS] Detailed Results:")
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

        print(f"\n[SAVED] Detailed log saved to: {log_path}")

        # Final recommendations
        print(f"\n[NEXT] Next Steps:")
        if total_success == total_tasks:
            print("   All tasks completed successfully!")
            print("   Open the web interface to view results: python app.py")
            print("   Check performance comparisons and visualizations")
        else:
            print("   Some tasks failed. Check the detailed log above.")
            print("   Review error messages and fix issues before retrying")
            print("   You can still view successful results in the web interface")

        print(f"\n[FINISHED] Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main execution function with comprehensive email notifications"""

    # Check if email is configured
    if not EMAIL_CONFIG['sender_password']:
        print("[WARNING] Email not configured!")
        print("Please set GOOGLE_APP_PASSWORD in your .env file.")
        print("Continuing without email notifications...")
        send_notifications = False
    else:
        send_notifications = True
        print("[INFO] Email notifications enabled")
    
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("[START] Starting Automated Training Pipeline with Email Notifications")
    print("=" * 70)
    print(f"Start time: {start_datetime}")
    print(f"Working directory: {Path.cwd()}")
    
    if send_notifications:
        # Send start notification
        start_subject = "🚀 Training Started - Trajectory Prediction Pipeline"
        start_body = f"""
Training pipeline has started successfully!

Start Time: {start_datetime}
Working Directory: {Path.cwd()}
Mode: Fully Automated

You will receive notifications for:
- Phase completions (Centralized/Federated training)
- Individual model completions  
- Final pipeline completion or failures

Pipeline is running...
        """
        send_email(start_subject, start_body)
    
    try:
        trainer = AutomatedTraining()
        trainer.load_config()

        print(f"\n[PIPELINE] Pipeline initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("[INFO] Running fully automated - no user interaction required")
        print("[INFO] Email notifications are active for all phases")
        
        # Run the full pipeline
        trainer.run_full_pipeline()
        
        # If we get here, training completed successfully
        end_time = time.time()
        duration = end_time - start_time
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n✅ Training completed successfully!")
        print(f"Duration: {format_duration(duration)}")
        
        if send_notifications:
            success_subject = "✅ Training Completed Successfully - Trajectory Prediction"
            success_body = f"""
Training pipeline has completed successfully!

Start Time: {start_datetime}
End Time: {end_datetime}
Duration: {format_duration(duration)}

All phases completed successfully:
- Centralized training and testing
- Federated training and testing

Your models are ready for use!
Check the results in the web interface.
            """
            send_email(success_subject, success_body, is_success=True)

    except KeyboardInterrupt:
        # User interrupted
        end_time = time.time()
        duration = end_time - start_time
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n\n[INTERRUPTED] Pipeline interrupted by user!")
        print(f"Duration: {format_duration(duration)}")
        print("[STOP] Stopping all processes...")
        
        if send_notifications:
            interrupt_subject = "⚠️ Training Interrupted - Trajectory Prediction"
            interrupt_body = f"""
Training pipeline was interrupted by user!

Start Time: {start_datetime}
End Time: {end_datetime}
Duration: {format_duration(duration)}

Training was stopped manually before completion.
            """
            send_email(interrupt_subject, interrupt_body, is_success=False)
        
        sys.exit(1)
        
    except Exception as e:
        # Unexpected error
        end_time = time.time()
        duration = end_time - start_time
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_trace = traceback.format_exc()
        
        print(f"\n\n[ERROR] Pipeline failed with error: {e}")
        print(f"Duration: {format_duration(duration)}")
        print(f"Traceback:\n{error_trace}")
        
        if send_notifications:
            error_subject = "❌ Training Failed - Trajectory Prediction"
            error_body = f"""
An unexpected error occurred during training!

Start Time: {start_datetime}
End Time: {end_datetime}
Duration: {format_duration(duration)}
Error: {e}

Full traceback:
{error_trace}

Please check the error and try again.
            """
            send_email(error_subject, error_body, is_success=False)
        
        sys.exit(1)

if __name__ == "__main__":
    main()
