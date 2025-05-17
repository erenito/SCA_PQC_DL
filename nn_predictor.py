import numpy as np
from trace_generation import compute_coeffs, message_to_bits
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, ReLU, Input
from keras.metrics import BinaryAccuracy
from keras.backend import clear_session
from keras.callbacks import Callback

class StopOnThreshold(Callback):
    def __init__(self, monitor='val_accuracy', threshold=0.80):
        super().__init__()
        self.monitor   = monitor
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val = logs.get(self.monitor)
        if val is not None and val >= self.threshold:
            self.model.stop_training = True

def get_parameters_real():
    try:
        file_path    = (input("ETS file [kyber_poly_from_msg_d2_impREGULAR_t0.ets]: ")
                        or "kyber_poly_from_msg_d2_impREGULAR_t0.ets")
        k            = int(input("Samples per byte (k)  ➜  window radius [304]: ") or 304)
        test_ratio   = float(input("Test split ratio [0.2]: ") or 0.2)
        num_bytes = int(input("Number of bytes to train [default 16]: ") or 16)
        epochs       = int(input("Training epochs [700]: ") or 700)
        batch_size   = int(input("Batch size [1024]: ") or 1024)
        val_split    = float(input("Validation split (train) [0.2]: ") or 0.2)
        results_file = input("Results file [results.txt]: ") or "results.txt"
    except ValueError as e:
        print("Invalid input ->", e);  exit(1)

    return (file_path, k, test_ratio, num_bytes,
            epochs, batch_size, val_split, results_file)

def get_parameters():
    """Prompt user for configuration parameters with sensible defaults."""
    try:
        sigma = float(input("Noise level (sigma) [default 1.0]: ") or 1.0)
        k = int(input("Samples per bit (k) [default 20]: ") or 20)
        num_messages_train = int(input("Number of training messages [default 8000]: ") or 8000)
        N_per_message_train = int(input("Traces per training message [default 1]: ") or 1)
        num_messages_test = int(input("Number of test messages [default 2000]: ") or 2000)
        N_per_message_test = int(input("Traces per test message [default 1]: ") or 1)
        num_bytes = int(input("Number of bytes to train [default 16]: ") or 16)
        epochs = int(input("Training epochs [default 100]: ") or 100)
        batch_size = int(input("Batch size [default 4096]: ") or 4096)
        validation_split = float(input("Validation split ratio [default 0.2]: ") or 0.2)
        results_file = input("Results file path [default results.txt]: ") or "results.txt"
    except ValueError as e:
        print("Invalid input:", e)
        exit(1)
    return sigma, k, num_messages_train, N_per_message_train, num_messages_test, N_per_message_test, num_bytes, epochs, batch_size, validation_split, results_file


def generate_random_messages(count):
    return [np.random.randint(0, 256, size=32, dtype='int32') for _ in range(count)]


def create_samples_from_messages(messages, N_per_message, sigma, k):
    """Generate side-channel traces and labels from a list of messages."""
    samples, labels = [], []
    for message in messages:
        bits = message_to_bits(message)
        for _ in range(N_per_message):
            mask = np.random.randint(0, 256, size=32, dtype='int32')
            share0 = mask
            share1 = mask ^ message
            s0 = compute_coeffs(share0, sigma=sigma, k=k)
            s1 = compute_coeffs(share1, sigma=sigma, k=k)
            samples.append(np.concatenate((s0, s1)))
            labels.append(bits)
    return np.array(samples), np.array(labels)


def build_and_train_models(X, y, s0_k, s1_k, epochs, batch_size, validation_split, num_bytes, omega=6):
    models = []
    stopper = StopOnThreshold('val_accuracy', 1.0)
    for byte_idx in range(num_bytes):
        start0 = byte_idx * s0_k
        end0 = start0 + s0_k
        start1 = byte_idx * s1_k
        end1 = start1 + s1_k
        for bit_idx in range(8):
            print(f"Training model for bit {byte_idx*8+bit_idx}")
            x_bit = X[:, start1:end1]
            y_bit = y[:, byte_idx*8+bit_idx].reshape(-1, 1)
            clear_session()
            model = Sequential([
                Input(shape=(x_bit.shape[1],)),
                Dense(32*(omega+1), use_bias=False), BatchNormalization(), ReLU(), 
                Dense(2**(omega+4), use_bias=False), BatchNormalization(), ReLU(),
                Dense(2**(omega+3), use_bias=False), BatchNormalization(), ReLU(),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(
                x_bit, y_bit,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1,
                shuffle=True,
                callbacks=[stopper]
            )
            models.append((byte_idx, bit_idx, model))
    return models

def predict_models(models, X, s0_k, s1_k):
    predictions_dict = {}
    for byte_idx, bit_idx, model in models:
        start0 = byte_idx * s0_k
        end0 = start0 + s0_k
        start1 = byte_idx * s1_k
        end1 = start1 + s1_k
        x_bit = X[:, start1:end1]
        preds = (model.predict(x_bit, verbose=0) > 0.5).astype(int).ravel()
        predictions_dict[byte_idx*8+bit_idx] = preds
    return predictions_dict

def results(predictions_dict, y_true, results_file):
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("Real Data Results S1:\n")
        for bit_idx in sorted(predictions_dict.keys()):
            preds = predictions_dict[bit_idx]
            metric = BinaryAccuracy(threshold=0.5)
            metric.update_state(y_true[:, bit_idx], preds)
            acc = float(metric.result().numpy())
            acc_line = f"b{bit_idx} acc={acc:.4f}"
            f.write(acc_line + "\n")
        f.write("**********\n")

def append_results(predictions_dict, y_true, sigma, k, n_train, npm_train, n_test, npm_test, results_file):
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(
            f"sigma={sigma}\n"
            f"k={k}\n"
            f"train_msg={n_train}\tNpm_train={npm_train}\n"
            f"test_msg={n_test}\tNpm_test={npm_test}\n"
        )

        for bit_idx in sorted(predictions_dict.keys()):
            accs = []
            for L in sorted(predictions_dict[bit_idx].keys()):
                preds = predictions_dict[bit_idx][L]
                metric = BinaryAccuracy(threshold=0.5)
                metric.update_state(y_true[:, bit_idx], preds)
                acc = float(metric.result().numpy())
                accs.append((L, acc))
            acc_line = f"b{bit_idx} " + " ".join([f"(L={L})={acc:.4f}" for L, acc in accs])
            f.write(acc_line + "\n")

        f.write("**********\n")
    print(f"Results for sigma={sigma} appended.")

def main():
    sigma, k, num_messages_train, N_per_message_train, num_messages_test, N_per_message_test, num_bits, epochs, batch_size, validation_split, results_file = get_parameters()
    omega = 1
    msgs_train = generate_random_messages(num_messages_train)
    msgs_test = generate_random_messages(num_messages_test)
    
    X_train, y_train = create_samples_from_messages(msgs_train, N_per_message_train, sigma, k)
    X_test, y_test = create_samples_from_messages(msgs_test, N_per_message_test, sigma, k)
    print(f"Training set shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Test set shape: {X_test.shape}, Labels shape: {y_test.shape}")
    print(X_train)
    print(y_train)
if __name__ == '__main__':
    main()