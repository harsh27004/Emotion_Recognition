import pickle
import numpy as np

# Load trained model
with open('./model', 'rb') as f:
    model = pickle.load(f)

# Load training data
data = np.loadtxt('data.txt')

# Split into X (features) and y (labels)
X, y = data[:, :-1], data[:, -1]

# Get accuracy per class
from sklearn.metrics import classification_report
y_pred = model.predict(X)
print(classification_report(y, y_pred, target_names=["Happy", "Sad", "Surprised"]))
