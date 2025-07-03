# Scikit-ANFIS: A Scikit-Learn Compatible Python Implementation for Adaptive Neuro-Fuzzy Inference System  

Scikit-ANFIS is a Python implementation of the Adaptive Neuro-Fuzzy Inference System (ANFIS) that is fully compatible with the Scikit-Learn interface, enabling seamless integration of ANFIS into existing machine learning workflows.  


## Background  

The Adaptive Neuro-Fuzzy Inference System (ANFIS) combines the self-learning capability of neural networks with the interpretability of fuzzy systems, demonstrating significant potential in control, prediction, and inference applications . While MATLAB has been the predominant platform for ANFIS, the growing popularity of Python in machine learning and deep learning has increased demand for Python-based ANFIS implementations.  

Existing Python-based ANFIS solutions lack compatibility with Scikit-Learn, one of the most widely used machine learning libraries. Scikit-ANFIS addresses this gap by providing uniform `fit()` and `predict()` interfaces, aligning with Scikit-Learn standards to facilitate integration with existing Python machine learning workflows .  


## Key Features  

- **Scikit-Learn Compatible Interface**: Uniform `fit()` and `predict()` methods for seamless collaboration with other Scikit-Learn models.  
- **Flexible Fuzzy System Construction**: Support for manual creation of general TSK fuzzy systems or automatic generation of ANFIS fuzzy systems .  
- **Diverse Membership Functions**: Implementation of 12 membership functions, including Gaussian, bell-shaped, triangular, etc.  
- **Two Training Strategies**: Hybrid learning and online learning algorithms for ANFIS training .  
- **Model Persistence**: Automatic saving and loading of trained ANFIS models .  
- **High-Performance Computing**: Powered by PyTorch and NumPy for efficient computations .  


## Installation  

### Install from Source  

```bash
# Clone the repository
git clone https://github.com/hudscomdz/scikit-anfis.git
cd scikit-anfis

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies and compile
pip install -r requirements.txt
pip install .
```  

### Dependencies  

- Python >= 3.8  
- PyTorch >= 1.8.0  
- NumPy >= 1.19.0  
- Scikit-Learn >= 0.24.0  
- SciPy >= 1.5.0  
- Pandas >= 1.2.0  


## Quick Start  

Here’s a simple example demonstrating how to use Scikit-ANFIS for the Iris classification problem:  

```python
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
```  


## Advanced Example: Manual Fuzzy System Construction  

```python
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
```  


## Project Structure  

```
scikit-anfis/  
├── src/
│   ├── __init__.py
│   └── skanfis/
│       ├── __init__.py
│       ├── scikit_anfis.py  # Core ANFIS implementation
│       ├── experimental.py  # Experimental infrastructure
│       ├── antecedent_parsing.py  # Parsing antecedent functions
│       ├── membership.py    # Membership function implementations
│       ├── utils.py         # Utility functions
│       └── fs               # Fuzzy system definitions
├── examples/               # Example data and scripts
│   ├── data_PCD_1000.txt   # PCD data
│   ├── data_sinc_121.csv   # Sinc data
│   ├── data_Tip_441.mat    # Tip data
│   ├── tipping_problem.py  # Restaurant tipping problem
│   ├── sinc_function.py    # Sinc function fitting
│   ├── chaotic_dynamics.py # Predict chaotic dynamics problem
│   └── iris.py             # Iris classification
├── README.md              # Readme
├── LICENSE                # License
├── requirements.txt       # Dependencies
├── setup.py               # Installation script
└── pyproject.toml         # Project configuration
```  


## Citation  

If you use Scikit-ANFIS in your research, please consider citing our paper:  

```bibtex
@article{zhang2024scikit,
  title={Scikit-ANFIS: A Scikit-Learn Compatible Python Implementation for Adaptive Neuro-Fuzzy Inference System},
  author={Zhang, Dongsong and Chen, Tianhua},
  journal={International Journal of Fuzzy Systems},  
  volume={26},
  pages={2039--2057},
  year={2024},
  publisher={Springer},
  doi={10.1007/s40815-024-01697-0}
}
```  


## Contribution Guidelines  

We welcome community contributions! If you encounter issues or have feature suggestions, please open an Issue. For code contributions:  

1. Fork the repository.  
2. Create a feature branch (`git checkout -b feature/amazing-feature`).  
3. Commit changes (`git commit -m 'Add some amazing feature'`).  
4. Push to the branch (`git push origin feature/amazing-feature`).  
5. Submit a Pull Request.  


## License  

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  


## Contact  

For questions or suggestions, contact:  
- Dongsong Zhang - dszhang@nudt.edu.cn  
- Tianhua Chen - T.Chen@hud.ac.uk  

Visit our [GitHub repository](https://github.com/hudscomdz/scikit-anfis) for more information and the latest code.
