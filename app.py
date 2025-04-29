import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from automl_backend import (
    analyze, preprocess_data, adjust_model_for_constraints, train_model, generate_training_script
)

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
    st.title("AutoML with Streamlit")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, na_values=['NA', '?'])
        st.write("Dataset Preview:")
        st.write(df.head())
        is_regression = st.radio("Is this a regression problem?", ["Yes", "No"]) == "Yes"
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'metadata' not in st.session_state:
            st.session_state.metadata = None
        if st.button("Analyze Data"):
            target = df.columns[-1]
            st.session_state.metadata = analyze(df, target, is_regression)
            st.session_state.analysis_complete = True
            st.write("Analysis Results:")
            st.json(st.session_state.metadata)
        if st.session_state.analysis_complete and st.session_state.metadata is not None:
            st.subheader("Model Constraints")
            col1, col2 = st.columns(2)
            with col1:
                training_time_minutes = st.slider(
                    "Maximum Training Time (minutes)",
                    min_value=1,
                    max_value=30,
                    value=10,
                    help="Maximum time allowed for model training"
                )
                target_accuracy = st.slider(
                    "Target Accuracy (%)",
                    min_value=0,
                    max_value=100,
                    value=80,
                    help="Desired accuracy for the model"
                )
            with col2:
                max_model_size_mb = st.slider(
                    "Maximum Model Size (MB)",
                    min_value=0.1,
                    max_value=10.0,
                    value=2.0,
                    step=0.1,
                    help="Maximum size of the trained model"
                )
                device = st.selectbox(
                    "Training Device",
                    ["Jetson Nano", "Google Coral Dev Board", "Raspberry Pi 4"],
                    help="Select the target device for model deployment"
                )
            if st.button("Train Model with Constraints"):
                adjusted_metadata = adjust_model_for_constraints(
                    st.session_state.metadata.copy(), 
                    training_time_minutes,
                    target_accuracy,
                    max_model_size_mb,
                    device
                )
                st.write("Model Configuration:")
                visualize_model_architecture(adjusted_metadata)
                with st.spinner('Preprocessing data and training model...'):
                    x, y, x_fields = preprocess_data(df, adjusted_metadata)
                    model, history = train_model(x, y, adjusted_metadata)
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
                    st.write(f"Model optimized for: {device}")
                    if device == "Jetson Nano":
                        st.write("Note: Model will be optimized for NVIDIA Jetson Nano's GPU acceleration")
                    elif device == "Google Coral Dev Board":
                        st.write("Note: Model will be optimized for Google Coral's TPU acceleration")
                    else:
                        st.write("Note: Model will be optimized for Raspberry Pi's CPU capabilities")
                    st.write("Training Script:")
                    training_script = generate_training_script(df, adjusted_metadata, x_fields)
                    st.code(training_script, language='python')
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                        model.save(tmp_file.name)
                        with open(tmp_file.name, 'rb') as f:
                            st.download_button(
                                label="Download Model",
                                data=f,
                                file_name="automl_model.h5",
                                mime="application/octet-stream"
                            )
                    os.unlink(tmp_file.name)
                    st.download_button(
                        label="Download Training Script",
                        data=training_script,
                        file_name="training_script.py",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main() 