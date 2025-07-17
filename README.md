# Federated Learning for Multi-Vehicle Trajectory Forecasting using Graph Neural Networks

This project implements and compares centralized and federated learning approaches for multi-vehicle trajectory forecasting using Graph Neural Networks (GNNs). The goal is to predict the future trajectory of a target vehicle based on its past behavior and the behavior of surrounding vehicles.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [Command Line](#command-line)
- [Models](#models)
  - [Centralized Model](#centralized-model)
  - [Federated Model](#federated-model)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project explores two machine learning paradigms for vehicle trajectory prediction:

1.  **Centralized Learning:** A single GNN model is trained on a centralized dataset containing data from all vehicles.
2.  **Federated Learning:** Multiple GNN models are trained locally on individual client datasets, and their model updates are aggregated on a central server to create a global model. This approach preserves data privacy by keeping the raw data on the client devices.

The project uses the Argoverse 2 dataset and a Graph Attention Network (GAT) to model the interactions between vehicles.

## Features

-   **Centralized and Federated Learning:** Implements both training and testing for both approaches.
-   **Graph Neural Network Model:** Uses a GAT to capture the complex interactions between vehicles.
-   **Web Interface:** A Flask-based web application to run experiments and visualize results.
-   **Data Visualization:** Visualizes the predicted trajectories on the Argoverse 2 maps.
-   **Command-Line Interface:** A `run_demo.py` script to run the experiments from the command line.

## Project Structure

```
.
├── app.py                  # Flask web application
├── README.md               # This file
├── requirements.txt        # Python package dependencies
├── run_demo.py             # Command-line demo script
├── train.py                # Centralized training and testing
├── dataset/                # Argoverse 2 dataset
├── federated/
│   ├── client.py           # Federated learning client
│   └── server.py           # Federated learning server
├── models/
│   └── gnn.py              # GNN model definition
├── results/
│   ├── centralized_training_history.json
│   ├── federated_training_history.json
│   └── test_predictions/   # Saved prediction visualizations
├── saved_models/
│   ├── *.pt              # Saved model files
├── utils/
│   ├── data_utils.py       # Data loading and preprocessing
│   └── viz_utils.py        # Visualization utilities
└── web/
    └── index.html          # Web interface
```

## Getting Started

### Prerequisites

-   Python 3.8+
-   PyTorch
-   PyTorch Geometric
-   Flower
-   Flask
-   Pandas
-   Matplotlib
-   Argoverse API

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/CosmicPegasus07/FedTraj-Prediction.git
    cd federated-gnn-trajectory
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download the Argoverse 2 dataset and place it in the `dataset/` directory.

## Usage

You can run the project using either the web interface or the command line.

### Web Interface

1.  Start the Flask web server:
    ```bash
    python app.py
    ```
2.  Open your web browser and go to `http://127.0.0.1:5000`.
3.  From the web interface, you can:
    -   Select the training mode (centralized or federated).
    -   Specify the dataset directories.
    -   Set the training parameters (batch size, epochs, learning rate).
    -   Run the training and testing scripts.
    -   View the training history and prediction visualizations.

### Command Line

The `run_demo.py` script provides a command-line interface to run the experiments.

**Example: Run Federated Learning Training**

```bash
python run_demo.py
```

The script will prompt you to select the mode (federated, centralized, or both) and the action (train, test, or both).

## Models

### Centralized Model

The centralized model is a `GATTrajectoryPredictor` defined in `models/gnn.py`. It is trained on the entire dataset using the `train.py` script.

### Federated Model

The federated model is also a `GATTrajectoryPredictor`. It is trained using the Flower framework. The `federated/` directory contains the client and server implementations.

## Dataset

This project uses the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html). The dataset should be downloaded and placed in the `dataset/` directory.

## Results

The training history and prediction visualizations are saved in the `results/` directory. The web interface provides a convenient way to view these results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.