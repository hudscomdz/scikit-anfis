from skanfis import scikit_anfis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris data
iris = load_iris()
# Split the data using the test_size argument in training (80%) and test (20%) set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create a Scikit-ANFIS fuzzy system object
model = scikit_anfis(data=iris.data, description="Iris", epoch=100, hybrid=True, label="c")
model.fit(X_train, y_train) # Model training
y_pred = model.predict(X_test)  # Prediction
print("Model Accuracy: ", accuracy_score(y_pred, y_test))