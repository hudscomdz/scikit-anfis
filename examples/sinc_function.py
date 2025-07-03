from skanfis.fs import *
from skanfis import *
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load Sinc data
sinc_data = read_csv("data_sinc_121.csv")
# Split the data using the test_size argument in training(60%) and test(40%) set
train_data, test_data = train_test_split(sinc_data, test_size=0.4, random_state=42)
y_test = test_data.pop('z')

# Create a TSK fuzzy system object by default
fs =FS()

# Define fuzzy sets and linguistic variables
S_1 = BellFuzzySet(a=3.33330, b=2, c=-10, term="mf0")
S_2 = BellFuzzySet(a=3.33330, b=2, c=-3.3330, term="mf1")
S_3 = BellFuzzySet(a=3.33330, b=2, c=3.33330, term="mf2")
S_4 = BellFuzzySet(a=3.33330, b=2, c=10, term="mf3")
fs.add_linguistic_variable("x0", LinguisticVariable([S_1, S_2, S_3, S_4], concept="Input Variable 1"))
F_1 = BellFuzzySet(a=3.33330, b=2, c=-10, term="mf0")
F_2 = BellFuzzySet(a=3.33330, b=2, c=-3.33330, term="mf1")
F_3 = BellFuzzySet(a=3.33330, b=2, c=3.3330, term="mf2")
F_4 = BellFuzzySet(a=3.33330, b=2, c=10, term="mf3")
fs.add_linguistic_variable("x1", LinguisticVariable([F_1, F_2, F_3, F_4], concept="Input Variable 2"))

# Define output functions and crisp value
for i in range(15):
    fs.set_output_function("sinc_x_y"+str(i),"2*x0+2*x1+1")
fs.set_crisp_output_value("sinc_x_y15", 1)

# Define fuzzy rules
R1 = "IF (x0 IS mf0) AND (x1 IS mf0) THEN (y0 IS sinc_x_y0)"
R2 = "IF (x0 IS mf0) AND (x1 IS mf1) THEN (y0 IS sinc_x_y1)"
R3 = "IF (x0 IS mf0) AND (x1 IS mf2) THEN (y0 IS sinc_x_y2)"
R4 = "IF (x0 IS mf0) AND (x1 IS mf3) THEN (y0 IS sinc_x_y3)"
R5 = "IF (x0 IS mf1) AND (x1 IS mf0) THEN (y0 IS sinc_x_y4)"
R6 = "IF (x0 IS mf1) AND (x1 IS mf1) THEN (y0 IS sinc_x_y5)"
R7 = "IF (x0 IS mf1) AND (x1 IS mf2) THEN (y0 IS sinc_x_y6)"
R8 = "IF (x0 IS mf1) AND (x1 IS mf3) THEN (y0 IS sinc_x_y7)"
R9 = "IF (x0 IS mf2) AND (x1 IS mf0) THEN (y0 IS sinc_x_y8)"
R10 = "IF (x0 IS mf2) AND (x1 IS mf1) THEN (y0 IS sinc_x_y9)"
R11 = "IF (x0 IS mf2) AND (x1 IS mf2) THEN (y0 IS sinc_x_y10)"
R12 = "IF (x0 IS mf2) AND (x1 IS mf3) THEN (y0 IS sinc_x_y11)"
R13 = "IF (x0 IS mf3) AND (x1 IS mf0) THEN (y0 IS sinc_x_y12)"
R14 = "IF (x0 IS mf3) AND (x1 IS mf1) THEN (y0 IS sinc_x_y13)"
R15 = "IF (x0 IS mf3) AND (x1 IS mf2) THEN (y0 IS sinc_x_y14)"
R16 = "IF (x0 IS mf3) THEN (y0 IS sinc_x_y15)"
fs.add_rules([R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16])

# Create a Scikit-ANFIS object based on fs
model = scikit_anfis(fs, description="Sinc", epoch=250, hybrid=True)
model.fit(train_data) # Model training
y_pred = model.predict(test_data) # Predicticon
print("Test RMSE:", mean_squared_error(y_pred,y_test))