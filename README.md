# AutoML Project

An intelligent AutoML (Automated Machine Learning) system that automatically analyzes datasets, preprocesses data, and trains optimized machine learning models based on user constraints. Built with Python, Streamlit, and TensorFlow.

## What is AutoML?

AutoML (Automated Machine Learning) is a system that automates the process of applying machine learning to real-world problems. This project aims to make machine learning accessible to users without deep technical expertise by automating the following tasks:

1. Data analysis and preprocessing
2. Feature engineering
3. Model selection and architecture design
4. Hyperparameter optimization
5. Model training and evaluation

## How It Works

The system follows a systematic approach to automate the machine learning pipeline:

1. **Data Upload and Analysis**
   - Users upload their dataset in CSV format
   - System automatically analyzes the data structure
   - Identifies data types, missing values, and potential issues
   - Determines if the problem is regression or classification

2. **Data Preprocessing**
   - Handles missing values using appropriate strategies
   - Applies feature scaling based on data distribution
   - Converts categorical variables to numerical representations
   - Splits data into training and validation sets

3. **Model Configuration**
   - Users specify constraints:
     - Maximum training time (1-30 minutes)
     - Target accuracy (0-100%)
     - Maximum model size (0.1-10 MB)
     - Target device (Jetson Nano/Google Coral Dev Board/Raspberry Pi 4)
   - System automatically adjusts model architecture based on constraints

4. **Model Training and Optimization**
   - Creates an optimized neural network architecture
   - Implements early stopping to prevent overfitting
   - Adjusts learning rate and batch size
   - Monitors training progress and validation metrics

5. **Results and Deployment**
   - Provides model performance metrics
   - Generates a reproducible training script
   - Exports the trained model in HDF5 format
   - Offers insights into model architecture and training process

## Key Features

- **Smart Data Analysis**: Automatically detects data types, handles missing values, and identifies target variables
- **Adaptive Preprocessing**: 
  - Handles both numerical and categorical data
  - Applies appropriate scaling (Z-score or MinMax) based on data distribution
  - Converts categorical variables to dummy variables when needed
- **Constraint-Based Model Training**:
  - Adjusts model architecture based on training time constraints
  - Optimizes for target accuracy
  - Considers model size limitations
  - Supports different deployment devices (Jetson Nano/Google Coral Dev Board/Raspberry Pi 4)
- **Interactive UI**: User-friendly Streamlit interface for easy data upload and model configuration
- **Code Generation**: Automatically generates training scripts for reproducibility
- **Model Export**: Saves trained models in HDF5 format for easy deployment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rk0802p/Minor-Project.git
cd Minor-Project
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Upload your dataset (CSV format) through the web interface

3. Configure model constraints:
   - Training time (1-30 minutes)
   - Target accuracy (0-100%)
   - Maximum model size (0.1-10 MB)
   - Target device (Jetson Nano/Google Coral Dev Board/Raspberry Pi 4)

4. The system will:
   - Analyze your data
   - Preprocess it automatically
   - Train an optimized model
   - Generate a training script
   - Provide model performance metrics

## Project Structure

```
Minor-Project/
├── app.py                 # Streamlit frontend application
├── automl_backend.py      # Core AutoML logic and model training
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Core Components

### Data Analysis (`analyze` function)
- Detects data types and distributions
- Identifies missing values
- Determines appropriate preprocessing steps
- Classifies problem type (regression/classification)

### Preprocessing (`preprocess_data` function)
- Handles missing values
- Applies appropriate scaling
- Converts categorical variables
- Prepares features and target variables

### Model Training (`train_model` function)
- Creates optimized neural network architecture
- Implements early stopping
- Supports different problem types
- Optimizes for specified constraints

### Code Generation (`generate_training_script` function)
- Generates reproducible Python scripts
- Includes all preprocessing steps
- Contains model architecture and training code

## Supported Problem Types

1. **Regression**
   - Continuous target variables
   - Uses mean squared error loss
   - Single output neuron
   - Example: Predicting house prices, stock prices

2. **Binary Classification**
   - Two-class problems
   - Uses binary cross-entropy loss
   - Sigmoid activation
   - Example: Spam detection, disease diagnosis

3. **Multi-class Classification**
   - Multiple classes
   - Uses categorical cross-entropy loss
   - Softmax activation
   - Example: Image classification, text categorization

## Model Architecture

The system automatically determines the optimal architecture based on:
- Input feature dimensions
- Target accuracy requirements
- Model size constraints
- Available training time

Typical architecture includes:
- Input layer (size based on features)
- Hidden layers (1-2 layers, size optimized)
- Dropout layers for regularization
- Output layer (size based on problem type)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SciPy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- Streamlit team for the web application framework
- Scikit-learn team for preprocessing utilities 