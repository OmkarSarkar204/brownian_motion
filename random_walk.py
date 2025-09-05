import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

pos = np.array([0.0,0.0])
num_of_steps = 10000
step_size = 1.0
path = [pos.copy()]

for i in range(num_of_steps):
  angle = np.random.uniform(0,2 * np.pi)
  disp = np.array([step_size * np.cos(angle), step_size * np.sin(angle)])
  pos += disp
  path.append(pos.copy())
path_arr = np.array(path)
print(f"Generated a path with shape: {path_arr.shape}")
print("Preparing the data")

window_size = 10
X_data = []
y_data = []

for i in range(len(path_arr) - window_size):
  window = path_arr[i:i+window_size]
  X_data.append(window)

  label = path_arr[i+window_size]
  y_data.append(label)

X = np.array(X_data)
y = np.array(y_data)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=73)

model = Sequential([
  LSTM(50, activation='relu', input_shape=(10, 2)),
  Dense(25, activation='relu'),
  Dense(2)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))


plt.figure(figsize=(10,10))
plt.plot(path_arr[:,0], path_arr[:,1], lw=0.5)
plt.title("Brownian Motion(Random Walk)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

num_points_to_plot = 100
path_segment_X = X_test[:num_points_to_plot]
true_path_y = y_test[:num_points_to_plot]

predicted_positions = model.predict(path_segment_X)

plt.figure(figsize=(12, 12))

plt.scatter(predicted_positions[:, 0], predicted_positions[:, 1], color='r', marker='x', label='Model Predictions', zorder=3)

plt.title("Model Predictions")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

num_points_to_plot = 100
path_segment_X = X_test[:num_points_to_plot]
true_path_y = y_test[:num_points_to_plot]

predicted_positions = model.predict(path_segment_X)

plt.figure(figsize=(12, 12))

true_path_start_points = path_segment_X[:, -1, :]
plt.plot(true_path_start_points[:, 0], true_path_start_points[:, 1], 'b-s', label='True Path')

plt.scatter(predicted_positions[:, 0], predicted_positions[:, 1], color='r', marker='x', label='Model Predictions', zorder=3)

plt.title("Model Predictions vs Original Path")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()





