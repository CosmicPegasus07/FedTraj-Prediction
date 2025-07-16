from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
import json
import threading

app = Flask(__name__, static_folder='web')

# Define the base directory of your project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

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

    cmd = ["python", os.path.join(BASE_DIR, "train.py"), "--mode", mode]

    if mode == 'centralized':
        if train_dir:
            cmd.extend(["--train_dir", train_dir])
        if val_dir:
            cmd.extend(["--val_dir", val_dir])
        cmd.extend(["--batch_size", str(batch_size), "--num_scenarios", str(num_scenarios), "--epochs", str(epochs), "--lr", str(lr)])
    elif mode == 'test':
        if test_dir:
            cmd.extend(["--test_dir", test_dir])
        cmd.extend(["--batch_size", str(batch_size), "--num_scenarios", str(num_scenarios)])
    elif mode == 'federated':
        if role:
            cmd.extend(["--role", role])
        # Federated mode might need more specific args depending on your client/server scripts
        # For now, assuming client/server scripts handle their own args or use defaults

    print(f"Running command: {' '.join(cmd)}")

    def run_process(command):
        try:
            process = subprocess.Popen(command, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            print(f"STDOUT:\n{stdout}")
            print(f"STDERR:\n{stderr}")
            if process.returncode != 0:
                return {"status": "error", "message": stderr, "command": " ".join(command)}
            return {"status": "success", "message": stdout, "command": " ".join(command)}
        except Exception as e:
            return {"status": "error", "message": str(e), "command": " ".join(command)}

    # Run in a separate thread to avoid blocking the Flask app
    # In a real application, you might use a task queue like Celery
    result = {}
    thread = threading.Thread(target=lambda: result.update(run_process(cmd)))
    thread.start()
    
    return jsonify({"status": "started", "message": "Script execution started in background. Check console for output."})

@app.route('/get_history/<history_type>', methods=['GET'])
def get_history(history_type):
    if history_type == 'centralized':
        history_path = os.path.join(RESULTS_DIR, 'centralized_training_history.json')
    elif history_type == 'federated':
        # Assuming federated history is also saved in results/federated_training_history.json
        history_path = os.path.join(RESULTS_DIR, 'federated_training_history.json')
    else:
        return jsonify({"status": "error", "message": "Invalid history type"}), 400

    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        return jsonify({"status": "success", "history": history})
    else:
        return jsonify({"status": "error", "message": f"History file not found: {history_path}"}), 404

@app.route('/get_visualizations/<model_type>', methods=['GET'])
def get_visualizations(model_type):
    if model_type == 'centralized':
        viz_dir = os.path.join(RESULTS_DIR, 'test_predictions', 'centralized')
    elif model_type == 'federated':
        viz_dir = os.path.join(RESULTS_DIR, 'test_predictions', 'federated')
    else:
        return jsonify({"status": "error", "message": "Invalid model type"}), 400

    if os.path.exists(viz_dir):
        images = [f for f in os.listdir(viz_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        # Return relative paths for web access
        image_paths = [os.path.join('results', 'test_predictions', model_type, img).replace('\\', '/') for img in images]
        return jsonify({"status": "success", "images": image_paths})
    else:
        return jsonify({"status": "error", "message": f"Visualization directory not found: {viz_dir}"}), 404

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'test_predictions', 'centralized'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'test_predictions', 'federated'), exist_ok=True)
    app.run(debug=True)
