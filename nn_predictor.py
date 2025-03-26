import numpy as np
from trace_generation import *
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, ReLU, Input
from keras.callbacks import EarlyStopping
from keras.backend import clear_session

# ----------------------------
# Data Generation Functions
# ----------------------------

def create_samples_from_messages(messages, N_per_message=1000, sigma=0):
    samples = []
    labels  = []
    for message in messages:
        # Convert the 32-byte message into a 256-bit vector (LSB-first)
        message_bits = message_to_bits(message)
        for _ in range(N_per_message):
            # Generate a random mask for this trace
            mask = np.random.randint(low=0, high=256, size=32, dtype='int32')
            share0 = mask
            share1 = mask ^ message  # XOR to combine mask and message
            # Compute the corresponding coefficients (trace) for each share
            samples0 = compute_coeffs(share0, sigma=sigma, k=9)
            samples1 = compute_coeffs(share1, sigma=sigma, k=9)
            # Concatenate the two parts into one sample
            sample = np.concatenate((samples0, samples1))
            samples.append(sample)
            labels.append(message_bits)
    return np.array(samples), np.array(labels)

# ----------------------------
# Parameters (Adjust as Needed)
# ----------------------------

sigma = 0  # Noise level

# Setup: 
# - Training: ~30K traces (e.g., 30 messages with 1000 traces each)
# - Testing: ~10K traces (e.g., 10 messages with 1000 traces each)
num_messages_train = 80
N_per_message_train = 200
num_messages_test  = 20
N_per_message_test = 200

# Generate random 32-byte messages (each message is an array of 32 integers in [0,255])
messages_train = [np.random.randint(low=0, high=256, size=32, dtype='int32')
                  for _ in range(num_messages_train)]
messages_test = [np.random.randint(low=0, high=256, size=32, dtype='int32')
                 for _ in range(num_messages_test)]

# Create training and test datasets (raw traces are used directly)
X_train, y_train = create_samples_from_messages(messages_train, N_per_message_train, sigma=sigma)
X_test,  y_test  = create_samples_from_messages(messages_test, N_per_message_test, sigma=sigma)

#Check the shapes of the generated datasets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
# ----------------------------
# Deep Learning Model Definition
# ----------------------------

def create_model_single_bit(input_dim, omega=1):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dense(32 * (omega + 1), use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(2 ** (omega + 4), use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(2 ** (omega + 3), use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ----------------------------
# Training the Models for Bit Recovery
# ----------------------------

N_train, input_dim = X_train.shape
# y_train has shape (N_train, 256) where each row is the 256-bit message

bit_models = []
predictions_all_train = np.zeros_like(y_train)

# Train a separate model for each of the 256 bits
for bit_idx in range(256):
    # Extract the label for the current bit (shape: (N_train, 1))
    y_train_bit = y_train[:, bit_idx].reshape(-1, 1)
    assert y_train_bit.shape == (N_train, 1)
    # Create and train the model for this bit
    clear_session()
    model_bit = create_model_single_bit(input_dim=input_dim, omega=1)
    model_bit.fit(
        X_train, 
        y_train_bit, 
        epochs=10, 
        batch_size=32, 
        validation_split=0.2, 
        verbose=1,
        shuffle=True
    )
    
    # Predict on the training set (for evaluation)
    pred_bit = (model_bit.predict(X_train) > 0.5).astype(int).flatten()
    predictions_all_train[:, bit_idx] = pred_bit
    bit_models.append(model_bit)

# ----------------------------
# Combine Predictions for Final Message Reconstruction (Training)
# ----------------------------

# For demonstration, combine predictions of all training samples by averaging per bit.
predicted_bits_final_train = (predictions_all_train.mean(axis=0) > 0.5).astype(int)

# Compare against the first sample's label from the training set.
first_sample_original_bits = y_train[0]
correctness_mask = (predicted_bits_final_train == first_sample_original_bits)
num_correct_bits = correctness_mask.sum()
num_total_bits   = len(first_sample_original_bits)

print("[Final Single-Message Reconstruction on Training Data]")
print("Predicted bits: ", predicted_bits_final_train)
print("Original bits (first sample):", first_sample_original_bits)
print("Correctness per bit:", correctness_mask)
print(f"Correct bits: {num_correct_bits}/{num_total_bits}")
print(f"Bit error count: {num_total_bits - num_correct_bits}")

# ----------------------------
# Evaluation on Test Data
# ----------------------------

# For each bit, predict on the test set using the corresponding trained model.
predictions_all_test = np.zeros((X_test.shape[0], 256), dtype=int)
for bit_idx in range(256):
    y_test_pred = (bit_models[bit_idx].predict(X_test) > 0.5).astype(int).flatten()
    predictions_all_test[:, bit_idx] = y_test_pred

# The test set contains multiple traces per test message.
# Group the predictions by test message and perform a majority vote.
# Assuming the samples are ordered such that the first N_per_message_test samples
# correspond to the first test message, the next N_per_message_test samples to the second, etc.
grouped_predictions = predictions_all_test.reshape((num_messages_test, N_per_message_test, 256))
predicted_bits_final_test = (grouped_predictions.mean(axis=1) > 0.5).astype(int)

# Similarly, group the ground-truth labels from y_test (all traces from the same message have identical labels)
grouped_y_test = y_test.reshape((num_messages_test, N_per_message_test, 256))
true_message_bits = grouped_y_test[:, 0, :]  # Take the label from the first trace for each message

# Compute bit error counts for each test message.
bit_errors_per_message = (predicted_bits_final_test != true_message_bits).sum(axis=1)
average_bit_error = bit_errors_per_message.mean()

print("\n[Evaluation on Test Data]")
for i in range(num_messages_test):
    print(f"Test Message {i+1}: Bit errors = {bit_errors_per_message[i]} out of 256")
print(f"Average bit error per message: {average_bit_error} out of 256")