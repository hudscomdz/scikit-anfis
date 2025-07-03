from skanfis.fs import *
from skanfis import *
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load PCD data
X = loadtxt('data_PCD_1000.txt', usecols=(0,1,2,3))
y = loadtxt('data_PCD_1000.txt', usecols=4)

# Split the data using the test_size argument in training (70%) and test (30%) set
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a TSK fuzzy system object by default
fs = FS()

# Define fuzzy sets and linguistic variables
S_1 = GaussianFuzzySet(mu=2, sigma=0.4270, term="mf0")
S_2 = GaussianFuzzySet(mu=2, sigma=1.3114, term="mf1")
fs.add_linguistic_variable("x0", LinguisticVariable([S_1, S_2]))
S_3 = GaussianFuzzySet(mu=2, sigma=0.4270, term="mf0")
S_4 = GaussianFuzzySet(mu=2, sigma=1.3114, term="mf1")
fs.add_linguistic_variable("x1", LinguisticVariable([S_3, S_4]))
F_1 = GaussianFuzzySet(mu=2, sigma=0.4270, term="mf0")
F_2 = GaussianFuzzySet(mu=2, sigma=1.3114, term="mf1")
fs.add_linguistic_variable("x2", LinguisticVariable([F_1, F_2]))
F_3 = GaussianFuzzySet(mu=2, sigma=0.4270, term="mf0")
F_4 = GaussianFuzzySet(mu=2, sigma=1.3114, term="mf1")
fs.add_linguistic_variable("x3", LinguisticVariable([F_3, F_4]))

# Define fuzzy rules
R1 = "IF (x0 IS mf0) AND (x1 IS mf0) AND (x2 IS mf0) AND (x3 IS mf0) THEN (y0 IS chaotic0)"
R2 = "IF (x0 IS mf0) AND (x1 IS mf0) AND (x2 IS mf0) AND (x3 IS mf1) THEN (y0 IS chaotic1)"
R3 = "IF (x0 IS mf0) AND (x1 IS mf0) AND (x2 IS mf1) AND (x3 IS mf0) THEN (y0 IS chaotic2)"
R4 = "IF (x0 IS mf0) AND (x1 IS mf0) AND (x2 IS mf1) AND (x3 IS mf1) THEN (y0 IS chaotic3)"
R5 = "IF (x0 IS mf0) AND (x1 IS mf1) AND (x2 IS mf0) AND (x3 IS mf0) THEN (y0 IS chaotic4)"
R6 = "IF (x0 IS mf0) AND (x1 IS mf1) AND (x2 IS mf0) AND (x3 IS mf1) THEN (y0 IS chaotic5)"
R7 = "IF (x0 IS mf0) AND (x1 IS mf1) AND (x2 IS mf1) AND (x3 IS mf0) THEN (y0 IS chaotic6)"
R8 = "IF (x0 IS mf0) AND (x1 IS mf1) AND (x2 IS mf1) AND (x3 IS mf1) THEN (y0 IS chaotic7)"
R9 = "IF (x0 IS mf1) AND (x1 IS mf0) AND (x2 IS mf0) AND (x3 IS mf0) THEN (y0 IS chaotic8)"
R10 = "IF (x0 IS mf1) AND (x1 IS mf0) AND (x2 IS mf0) AND (x3 IS mf1) THEN (y0 IS chaotic9)"
R11 = "IF (x0 IS mf1) AND (x1 IS mf0) AND (x2 IS mf1) AND (x3 IS mf0) THEN (y0 IS chaotic10)"
R12 = "IF (x0 IS mf1) AND (x1 IS mf0) AND (x2 IS mf1) AND (x3 IS mf1) THEN (y0 IS chaotic11)"
R13 = "IF (x0 IS mf1) AND (x1 IS mf1) AND (x2 IS mf0) AND (x3 IS mf0) THEN (y0 IS chaotic12)"
R14 = "IF (x0 IS mf1) AND (x1 IS mf1) AND (x2 IS mf0) AND (x3 IS mf1) THEN (y0 IS chaotic13)"
R15 = "IF (x0 IS mf1) AND (x1 IS mf1) AND (x2 IS mf1) AND (x3 IS mf0) THEN (y0 IS chaotic14)"
R16 = "IF (x0 IS mf1) AND (x1 IS mf1) AND (x2 IS mf1) THEN (y0 IS chaotic15)"
fs.add_rules([R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16])

# Define output functions
for i in range(15):
    fs.set_output_function("chaotic"+str(i), "2*x0+2*x1+2*x2+2*x3+1")
fs.set_output_function("chaotic15", "2*x0+2*x1+2*x2+0*x3+1")

# Create a Scikit-ANFIS fuzzy system object based on fs
model = scikit_anfis(fs, description="PCD", epoch=500, hybrid=True)
model.fit(X_train, y_train)  # Model training
y_pred = model.predict(X_test)  # Prediction
print("Test RMSE: ", mean_squared_error(y_pred, y_test))