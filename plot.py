import estraces
import matplotlib.pyplot as plt
import numpy as np
from nn_predictor import *
from sklearn.model_selection import train_test_split

ths = estraces.read_ths_from_ets_file("kyber_poly_from_msg_d2_impREGULAR_t0.ets")

X = ths.samples[:]

mean = np.mean(X, axis=0)

# Plot the mean
plt.plot(mean)
plt.title("Mean of Samples")
plt.xlabel("Sample Index")
plt.ylabel("Mean Value")
plt.grid()
plt.show()