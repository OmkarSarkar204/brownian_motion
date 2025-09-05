# Predicting Brownian Motion with an LSTM Neural Network
This project uses a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) units to predict the path of a particle undergoing Brownian motion. The model is built with TensorFlow and Keras and is trained on data generated from a 2D random walk simulation. ***The purpose is to see whether a machine learning model can replicate or predict random motion.***

## Background: The Physics of Brownian Motion
Brownian motion is the random movement of particles suspended in a fluid such as a liquid or a gas. It was first observed by ***Robert Brown*** in 1827 and later explained by ***Albert Einstein*** in 1905.
The particle moves because it is constantly hit by the much smaller molecules of the fluid. The collisions are not perfectly balanced at each moment, causing the particle to follow an erratic, unpredictable path. This project trains a model to learn the statistical pattern of this motion.

## Project Goal
The goal is to predict the next (x, y) position of a particle based on a sequence of its recent positions. This demonstrates time-series forecasting applied to a physics problem.

## Method
1. Data Generation: Simulate a 2D random walk using NumPy to create the particle's trajectory.
2. Data Preparation: Use a sliding window of 10 past positions to predict the next position. This generates thousands of training samples.
3. Model Architecture: Build an RNN with Keras:
  - LSTM layer to process sequences and learn temporal patterns
  - Dense layer for intermediate processing
  - Output Dense layer with 2 neurons for (x, y) prediction
4. Training and Evaluation: Train on 80% of the data and evaluate on 20%. Use Mean Squared Error as the loss function.

## Results
*The model predicts the next position with over 98% accuracy compared to the simulation's average step size.*

## Random Walk test(generated in python) - 1
<img width="841" height="855" alt="random_walk-1" src="https://github.com/user-attachments/assets/4bbf9cd6-aa1c-4103-95db-6f1c3289acfa" />

## Prediction for Image - 1
<img width="1005" height="1009" alt="model_pred-1" src="https://github.com/user-attachments/assets/80e338e9-7c9f-40a3-b092-69dd88af6144" />

## Prediction vs Original Motion - 1
<img width="1005" height="1009" alt="model_vs_pred-1" src="https://github.com/user-attachments/assets/80ff38ee-57fd-42c6-ae57-a4d4461cef3f" />

## Random Walk test(generated in python) - 2
<img width="853" height="855" alt="random_walk-2" src="https://github.com/user-attachments/assets/5d9c6d1e-dd84-427a-85a4-5327bf3e1408" />

## Prediction for Image - 2
<img width="1008" height="1009" alt="model_pred-2" src="https://github.com/user-attachments/assets/d7555f78-b59d-4bf5-8249-8546005b57ed" />

## Prediction vs Original Motion - 2
<img width="1008" height="1009" alt="model_vs_pred-2" src="https://github.com/user-attachments/assets/6295bdb7-c450-4249-bb95-c12a26c2bdc1" />

## How to Run

First install all the libraries from the ```requirement.txt```

## Installation

1. Clone the repository:
``` git clone https://github.com/OmkarSarkar204/brownian_motion.git ```

2. Go to the project folder:
  ```cd brownian_motion```

3. Install dependencies (use a virtual environment if possible):
  ```pip install -r requirements.txt```

Run the program.
