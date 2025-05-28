import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Dense (fully connected) layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with small random values and biases with zeros
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Calculate output values from inputs, weights, and biases
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradient on parameters and inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU activation function
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        # Apply ReLU (max(0, x))
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Backpropagation through ReLU
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# Softmax activation function
class Activation_Softmax:
    def forward(self, inputs):
        # Apply softmax with numerical stability
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probability = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = probability

# Base loss class
class Loss:
    def calculate(self, output, y):
        # Calculate the mean loss over all samples
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Categorical Crossentropy Loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # If labels are integers
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # If one-hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Negative log likelihood
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # If labels are sparse, convert to one-hot
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Gradient of loss
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

# Combined Softmax activation and Categorical Crossentropy loss
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # Perform softmax and return loss
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # If one-hot encoded, convert to class indices
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can modify safely
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

# Preprocess the medical no-show dataset
def preprocess_medical_data(df):
    data = df.copy()

    # Drop columns that aren't useful for prediction
    irrelevant_columns = ['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay']
    data.drop(columns=[col for col in irrelevant_columns if col in data.columns], inplace=True)

    # Remove rows with missing values
    data = data.dropna()

    label_encoders = {}

    # Encode Gender to numeric
    if 'Gender' in data.columns:
        le = LabelEncoder()
        data['Gender'] = le.fit_transform(data['Gender'])
        label_encoders['Gender'] = le

    # One-hot encode Neighbourhood
    if 'Neighbourhood' in data.columns:
        data = pd.get_dummies(data, columns=['Neighbourhood'], drop_first=True)

    # Convert 'No-show' to binary target
    if 'No-show' in data.columns:
        data['No-show'] = (data['No-show'] == 'Yes').astype(int)

    # Separate features and labels
    X = data.drop('No-show', axis=1).values
    y = data['No-show'].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoders

# Load and preprocess data
df = pd.read_csv('dataset.csv')
X, y, scaler, encoders = preprocess_medical_data(df)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create layers
dense1 = Layer_Dense(X_train.shape[1], 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 2)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Training loop
for epoch in range(1000):
    # Forward pass
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y_train)

    # Calculate predictions and accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y_train)

    # Print loss and accuracy every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y_train)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    lr = 0.01
    dense1.weights -= lr * dense1.dweights
    dense1.biases -= lr * dense1.dbiases
    dense2.weights -= lr * dense2.dweights
    dense2.biases -= lr * dense2.dbiases
