from skanfis.fs import *
from skanfis import *
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load Tip data
tip_data = loadmat('data_Tip_441.mat')['Tip_data']
# Split the data using the test_size argument in training(50%) and test(50%) set
train_data, test_data = train_test_split(tip_data, test_size=0.5, random_state=42)
X_test = test_data[:,:-1]
y_test = test_data[:,-1]

# Create a TSK fuzzy system object by default
fs = FS()

# Define fuzzy sets and linguistic variables
S_1 = TriangleFuzzySet(a=0, b=0, c=5, term="poor")
S_2 = TriangleFuzzySet(a=0, b=5,c=10, term="good")
S_3 = TriangleFuzzySet(a=5, b=10, c=10, term="excellent")
fs.add_linguistic_variable("Service", LinguisticVariable([S_1, S_2, S_3], concept="Service quality"))
F_1 = TriangleFuzzySet(a=0, b=0, c=10, term="rancid")
F_2 = TriangleFuzzySet(a=0, b=10, c=10, term="delicious")
fs.add_linguistic_variable("Food",LinguisticVariable([F_1, F_2], concept="Food quality"))

# Define crisp outputs for small and average tip
fs.set_crisp_output_value("small", 5)
fs.set_crisp_output_value("average", 15)
# Define function for generous tip (2*food score + 3*service score + 5%)
fs.set_output_function("generous", "2*Food+3*Service+5")

# Define fuzzy rules
R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)"
R2 = "IF (Service IS good) THEN (Tip IS average)"
R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
fs.add_rules([R1, R2, R3])

# Create a Scikit-ANFIS object based on fs
model = scikit_anfis(fs, description="Tip", epoch=10, hybrid=True)
model.fit(train_data) # Model training
y_pred = model.predict(X_test) # Prediction
print("Test RMSE: ", mean_squared_error(y_pred, y_test))