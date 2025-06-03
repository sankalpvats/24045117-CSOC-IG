#<----------------------------------------THIS IS NEURAL NETWORK FROM SCRATCH----------------------------------------------------------------->
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc, confusion_matrix

# Dense (fully connected) layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU activation function
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# Softmax activation function
class Activation_Softmax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probability = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = probability

# Base loss class
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Categorical Crossentropy Loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

# Combined Softmax activation and Categorical Crossentropy loss
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

# Preprocess the medical no-show dataset
def preprocess_medical_data(df):
    data = df.copy()
    irrelevant_columns = ['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay']
    data.drop(columns=[col for col in irrelevant_columns if col in data.columns], inplace=True)
    data = data.dropna()
    label_encoders = {}
    if 'Gender' in data.columns:
        le = LabelEncoder()
        data['Gender'] = le.fit_transform(data['Gender'])
        label_encoders['Gender'] = le
    if 'Neighbourhood' in data.columns:
        data = pd.get_dummies(data, columns=['Neighbourhood'], drop_first=True)
    if 'No-show' in data.columns:
        data['No-show'] = (data['No-show'] == 'Yes').astype(int)
    X = data.drop('No-show', axis=1).values
    y = data['No-show'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, label_encoders

def calculate_f1_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return f1

# Load and preprocess data
df = pd.read_csv('dataset.csv')
X, y, scaler, encoders = preprocess_medical_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create layers
dense1 = Layer_Dense(X_train.shape[1], 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 2)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Training loop
for epoch in range(1000):
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y_train)
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y_train)
    if epoch % 100 == 0:
        f1 = calculate_f1_score(y_train, predictions)
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    loss_activation.backward(loss_activation.output, y_train)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    lr = 0.01
    dense1.weights -= lr * dense1.dweights
    dense1.biases -= lr * dense1.dbiases
    dense2.weights -= lr * dense2.dweights
    dense2.biases -= lr * dense2.dbiases

# Final evaluation on test set
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss_activation.forward(dense2.output, y_test)
test_predictions = np.argmax(loss_activation.output, axis=1)
test_accuracy = accuracy_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions)
conf_matrix = confusion_matrix(y_test, test_predictions)
positive_probs = loss_activation.output[:, 1]
precision, recall, _ = precision_recall_curve(y_test, positive_probs)
pr_auc = auc(recall, precision)

print("\n--- Test Set Evaluation ---")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
