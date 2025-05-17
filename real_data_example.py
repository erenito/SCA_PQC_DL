import estraces
import matplotlib.pyplot as plt
import numpy as np
from nn_predictor import *
from sklearn.model_selection import train_test_split

ths = estraces.read_ths_from_ets_file("kyber_poly_from_msg_d2_impREGULAR_t0.ets")

y = np.unpackbits(ths.s1, axis=1, bitorder="little")

X = ths.samples[:,5160:9960]

S1_k = 300
S0_k = 304

file_path, k, test_ratio, num_bytes, epochs, batch_size, val_split, results_file = get_parameters_real()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

models = build_and_train_models(X=X_train, y=y_train, s0_k=S0_k, s1_k=S1_k, epochs=epochs, batch_size=batch_size, validation_split=val_split, num_bytes=num_bytes)

predictions_dict = predict_models(models=models, X=X_test, s0_k=S0_k, s1_k=S1_k)

results(predictions_dict, y_test, results_file)