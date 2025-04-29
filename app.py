import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from automl_backend import (
    analyze, preprocess_data, adjust_model_for_constraints, train_model, generate_training_script
)

def reset_session_state():
    st.session_state.clear()
    st.session_state.analysis_complete = False
    st.session_state.metadata = None
    st.session_state.model_trained = False
    st.session_state.current_step = 1

def visualize_model_architecture(metadata):
    st.subheader("Model Architecture")
    st.write("Model Structure:")
    st.write("```")
    st.write(f"Input Layer ({metadata['input_dim']} features)")
    st.write(f"└─ Dense Layer ({metadata['hidden_units'][0]} units, ReLU)")
    st.write(f"   └─ Dropout ({metadata['dropout_rate']})")
    st.write(f"      └─ Dense Layer ({metadata['hidden_units'][1]} units, ReLU)")
    st.write(f"         └─ Dropout ({metadata['dropout_rate']})")
    if metadata["type"] == "regression":
        st.write("            └─ Output Layer (1 unit, Linear)")
    elif metadata["type"] == "binary-classification":
        st.write("            └─ Output Layer (1 unit, Sigmoid)")
    else:
        st.write(f"            └─ Output Layer ({metadata['output_units']} units, Softmax)")
    st.write("```")
    st.subheader("How Constraints Affect the Model")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model Size Impact:**")
        if metadata["hidden_units"][0] == 16:
            st.write("- Small model (0.1-0.4MB)")
            st.write("- Fewer parameters for edge devices")
            st.write("- Faster inference but lower capacity")
        elif metadata["hidden_units"][0] == 32:
            st.write("- Medium model (0.4-2MB)")
            st.write("- Balanced capacity and size")
            st.write("- Good for most applications")
        else:
            st.write("- Large model (2-10MB)")
            st.write("- Higher capacity for complex patterns")
            st.write("- Better accuracy but slower inference")
        st.write("**Training Time Impact:**")
        st.write(f"- Maximum {metadata['epochs']} epochs")
        st.write(f"- Early stopping after {metadata['patience']} epochs without improvement")
        st.write(f"- Batch size {metadata['batch_size']} for optimal training speed")
    with col2:
        st.write("**Accuracy Impact:**")
        if metadata["learning_rate"] == 0.0001:
            st.write("- High accuracy target (>90%)")
            st.write("- Slower learning for better precision")
            st.write("- More epochs for convergence")
        elif metadata["learning_rate"] == 0.001:
            st.write("- Medium accuracy target (80-90%)")
            st.write("- Balanced learning speed and precision")
            st.write("- Standard convergence time")
        else:
            st.write("- Lower accuracy target (<80%)")
            st.write("- Faster learning for quick results")
            st.write("- Fewer epochs needed")
        st.write("**Device Optimization:**")
        if metadata["batch_size"] == 32:
            st.write("- Optimized for Jetson Nano GPU")
            st.write("- Parallel processing enabled")
        elif metadata["batch_size"] == 64:
            st.write("- Optimized for Google Coral TPU")
            st.write("- Matrix operations acceleration")
        else:
            st.write("- Optimized for Raspberry Pi CPU")
            st.write("- Memory-efficient processing")

def main():
    st.set_page_config(layout="wide")
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        reset_session_state()

    # Sidebar with step headers only
    with st.sidebar:
        st.title("AutoML Steps")
        steps = [
            "Step 1: Data Upload",
            "Step 2: Problem Type",
            "Step 3: Data Analysis",
            "Step 4: Model Constraints",
            "Step 5: Download Results"
        ]
        
        for i, step in enumerate(steps, 1):
            if i == st.session_state.current_step:
                st.markdown(f"**→ {step}**")
            else:
                st.markdown(step)
        
        # Reset button at bottom of sidebar
        if st.button("Reset All", key="reset_button"):
            reset_session_state()
            st.rerun()

    # Main content area
    st.title("AutoML Process")
    
    # Step 1: Data Upload
    if st.session_state.current_step == 1:
        st.header("Step 1: Data Upload")
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="file_uploader")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, na_values=['NA', '?'])
            st.session_state.df = df
            st.write("Dataset Preview:")
            st.write(df.head())
            st.session_state.current_step = 2
            st.rerun()
    
    # Step 2: Problem Type
    elif st.session_state.current_step == 2:
        st.header("Step 2: Problem Type")
        is_regression = st.radio(
            "Is this a regression problem?",
            ["Yes", "No"],
            key="problem_type"
        )
        if st.button("Continue"):
            st.session_state.is_regression = is_regression == "Yes"
            st.session_state.current_step = 3
            st.rerun()
        
        # Show dataset preview
        if 'df' in st.session_state:
            st.write("Dataset Preview:")
            st.write(st.session_state.df.head())
    
    # Step 3: Data Analysis
    elif st.session_state.current_step == 3:
        st.header("Step 3: Data Analysis")
        if st.button("Analyze Data"):
            target = st.session_state.df.columns[-1]
            st.session_state.metadata = analyze(st.session_state.df, target, st.session_state.is_regression)
            st.session_state.analysis_complete = True
            st.session_state.current_step = 4
            st.rerun()
        
        # Show dataset preview and problem type
        st.write("Dataset Preview:")
        st.write(st.session_state.df.head())
        st.write(f"Problem Type: {'Regression' if st.session_state.is_regression else 'Classification'}")
    
    # Step 4: Model Constraints
    elif st.session_state.current_step == 4:
        st.header("Step 4: Model Constraints")
        col1, col2 = st.columns(2)
        
        with col1:
            training_time_minutes = st.slider(
                "Maximum Training Time (minutes)",
                min_value=1,
                max_value=30,
                value=10
            )
            target_accuracy = st.slider(
                "Target Accuracy (%)",
                min_value=0,
                max_value=100,
                value=80
            )
        
        with col2:
            max_model_size_mb = st.slider(
                "Maximum Model Size (MB)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1
            )
            device = st.selectbox(
                "Training Device",
                ["Jetson Nano", "Google Coral Dev Board", "Raspberry Pi 4"]
            )
        
        if st.button("Train Model"):
            st.session_state.training_time = training_time_minutes
            st.session_state.target_accuracy = target_accuracy
            st.session_state.max_model_size = max_model_size_mb
            st.session_state.device = device
            
            # Train model
            adjusted_metadata = adjust_model_for_constraints(
                st.session_state.metadata.copy(),
                training_time_minutes,
                target_accuracy,
                max_model_size_mb,
                device
            )
            
            visualize_model_architecture(adjusted_metadata)
            
            with st.spinner('Preprocessing data and training model...'):
                x, y, x_fields = preprocess_data(st.session_state.df, adjusted_metadata)
                model, history = train_model(x, y, adjusted_metadata)
                st.session_state.model = model
                
                # Generate and store training script
                training_script = generate_training_script(st.session_state.df, adjusted_metadata, x_fields)
                st.session_state.training_script = training_script
                
                st.success("Model Training Complete!")
                
                if adjusted_metadata["type"] == "regression":
                    pred = model.predict(x)
                    score = np.sqrt(np.mean((pred - y) ** 2))
                    st.write(f"Root Mean Square Error (RMSE): {score:.4f}")
                else:
                    pred = model.predict(x)
                    if adjusted_metadata["type"] == "binary-classification":
                        pred_classes = (pred > 0.5).astype(int)
                        accuracy = np.mean(pred_classes == y)
                        st.write(f"Accuracy: {accuracy:.4f}")
                    else:
                        pred_classes = np.argmax(pred, axis=1)
                        y_classes = np.argmax(y, axis=1)
                        accuracy = np.mean(pred_classes == y_classes)
                        st.write(f"Accuracy: {accuracy:.4f}")
                
                st.session_state.current_step = 5
                st.rerun()
        
        # Show analysis results
        if st.session_state.analysis_complete:
            st.subheader("Analysis Results")
            st.json(st.session_state.metadata)
    
    # Step 5: Download Results
    elif st.session_state.current_step == 5:
        st.header("Step 5: Download Results")
        
        # Show training results
        if 'model' in st.session_state:
            st.success("Model Training Complete!")
            st.write(f"Model optimized for: {st.session_state.device}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                    st.session_state.model.save(tmp_file.name)
                    with open(tmp_file.name, 'rb') as f:
                        st.download_button(
                            label="Download Model",
                            data=f,
                            file_name="automl_model.h5",
                            mime="application/octet-stream"
                        )
                os.unlink(tmp_file.name)
            
            with col2:
                st.download_button(
                    label="Download Training Script",
                    data=st.session_state.training_script,
                    file_name="training_script.py",
                    mime="text/plain"
                )
            
            # Show training script
            st.subheader("Training Script")
            st.code(st.session_state.training_script, language='python')

if __name__ == "__main__":
    main() 