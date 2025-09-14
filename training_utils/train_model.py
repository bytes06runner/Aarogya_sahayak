# train_model.py (FINAL, ROBUST VERSION)
# This script reads data, trains a model, and saves the output
# to a 'models' folder in the main project directory.
# It includes robust data splitting and error handling.

import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate, TextVectorization
from keras.optimizers import Adam
import math

print("--- Step 1: Initializing Training Script ---")
print(f"TensorFlow Version: {tf.__version__}")

# --- Configuration ---
# Get the absolute path of the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the main project directory (one level up)
MAIN_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATASET_PATH = os.path.join(SCRIPT_DIR, 'health_data.csv')
MODEL_EXPORT_DIR = os.path.join(MAIN_PROJECT_DIR, 'models')
print(f"Model will be saved to: {MODEL_EXPORT_DIR}\n")

try:
    # --- Step 2: Load Dataset ---
    print(f"--- Step 2: Loading Dataset from {DATASET_PATH} ---")
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully. Shape:", df.shape)
    print("Columns:", df.columns.tolist(), "\n")

    # --- Step 3: Preprocess Data ---
    print("--- Step 3: Preprocessing Data ---")
    # Drop any rows with missing values to ensure data quality
    df.dropna(inplace=True)
    
    X_text = df['symptom_text'].astype(str)
    X_num = df[['aqi', 'temperature']].astype(float)
    y_labels = df['condition']

    scaler = MinMaxScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    
    # --- Robust Data Splitting Logic ---
    n_classes = len(np.unique(y))
    test_size_ratio = 0.2
    
    # Check if any class has fewer than 2 samples, which makes stratification impossible
    class_counts = df['condition'].value_counts()
    if (class_counts < 2).any():
        print("--- FATAL ERROR: At least one class has only 1 member. Cannot stratify. ---")
        problematic_classes = class_counts[class_counts < 2].index.tolist()
        print(f"Please ensure these conditions in health_data.csv have at least two rows: {problematic_classes}")
        exit()

    # Calculate the required size of the test set to include at least one of each class
    required_test_size = n_classes
    # Calculate the actual number of samples that will be in the test set
    actual_test_samples = math.floor(len(df) * test_size_ratio)

    if actual_test_samples < required_test_size:
        print(f"--- WARNING: Dataset is too small for a {int(test_size_ratio*100)}% test split. ---")
        print(f"Test set would have {actual_test_samples} samples, but we need at least {required_test_size} to cover all {n_classes} classes.")
        # We must disable stratification if we can't guarantee all classes in the test set
        print("Proceeding without stratification. A larger dataset is recommended.")
        stratify_option = None
    else:
        stratify_option = y

    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_num_scaled, y, test_size=test_size_ratio, random_state=42, stratify=stratify_option
    )
    print("Data successfully preprocessed and split.\n")

    # --- Step 4: Build the AI Model ---
    print("--- Step 4: Building the AI Model ---")
    # Define vocabulary size based on our larger dataset
    vocab_size = 2000
    sequence_length = 20
    
    text_vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)
    # Train the vectorizer on the training text data
    text_vectorizer.adapt(X_text_train.to_numpy())
    
    # Define the model architecture
    text_input = Input(shape=(1,), dtype=tf.string, name='text_input')
    vectorized_text = text_vectorizer(text_input)
    text_embedding = Embedding(input_dim=vocab_size, output_dim=16, name='embedding')(vectorized_text)
    text_flatten = Flatten()(text_embedding)

    numerical_input = Input(shape=(2,), name='numerical_input')
    concatenated = Concatenate()([text_flatten, numerical_input])

    dense1 = Dense(32, activation='relu')(concatenated)
    output = Dense(n_classes, activation='softmax')(dense1)

    model = Model(inputs=[text_input, numerical_input], outputs=output, name="functional")
    print("Model built successfully.")
    model.summary()

    # --- Step 5: Train the Model ---
    print("\n--- Step 5: Training the Model ---")
    
    # Explicitly convert pandas Series to NumPy arrays before fitting to prevent dtype errors
    X_text_train_np = X_text_train.to_numpy()
    X_text_test_np = X_text_test.to_numpy()

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit([X_text_train_np, X_num_train], y_train,
                        validation_data=([X_text_test_np, X_num_test], y_test),
                        epochs=100,
                        verbose=2)
    print("Training complete.\n")

    # --- Step 6: Convert to TensorFlow Lite ---
    print("--- Step 6: Converting Model to TensorFlow Lite ---")
    os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # This is the crucial part for Mac compatibility
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS 
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # --- Step 7: Save Final Artifacts ---
    print("--- Step 7: Saving Final Artifacts ---")
    tflite_model_path = os.path.join(MODEL_EXPORT_DIR, 'model.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved to: {tflite_model_path}")

    labels_path = os.path.join(MODEL_EXPORT_DIR, 'labels.json')
    labels_map = {str(i): label for i, label in enumerate(label_encoder.classes_)}
    with open(labels_path, 'w') as f:
        json.dump(labels_map, f, indent=4)
    print(f"Labels saved to: {labels_path}")
    
    print("\n\n--- SUCCESS! ---")
    print("You can now build the main chatbot application in the 'SIH-Project-Final' folder.")

except Exception as e:
    print("\n--- AN ERROR OCCURRED ---")
    print(f"Error details: {e}")
    print("\n--- TRACEBACK ---")
    import traceback
    traceback.print_exc()

