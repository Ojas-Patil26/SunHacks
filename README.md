# 🏠 SunHacks: Housing Price Predictor

A hands-on machine learning project built for **SunHacks**, leveraging `scikit-learn`'s Linear Regression model to predict housing prices. This project emphasizes reproducibility, clean code, and efficient ML workflows.

---

## 🚀 Key Features

* Load and preprocess structured tabular data from `Housing.csv`
* Split dataset into training and test sets using `train_test_split`
* Train a `LinearRegression` model using `scikit-learn`
* Evaluate performance using:

  * Mean Absolute Error (MAE)
  * Mean Squared Error (MSE)
  * Root Mean Squared Error (RMSE)
* Visualize predicted vs actual results using `seaborn`

---

## 📁 Project Structure

```
SunHacks/
├── main.py             # End-to-end pipeline: load, train, evaluate, visualize
├── Housing.csv         # Dataset with features and target values
├── requirements.txt    # Environment dependencies
├── .gitignore          # Ignore virtualenv, pycache, etc.
└── README.md           # Project documentation
```

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repository
$ git clone https://github.com/Ojas-Patil26/SunHacks.git
$ cd SunHacks

# 2. Create and activate a virtual environment
$ python3 -m venv venv
$ source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install required dependencies
$ pip install -r requirements.txt

# 4. Run the project
$ python main.py
```

---

## 📊 Sample Output (After Running `main.py`)

Performance Metrics:

```
MAE:         ~82137.70
MSE:         ~1.01e+10
RMSE:        ~100318.48
```

Visual Output:

<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/ddb6f287-c93a-4019-bbc0-182fb762a183" />

---

## 📦 Dependencies

All required libraries are open-source and listed in `requirements.txt`:

* `pandas`, `numpy` → Data manipulation
* `scikit-learn` → Model building, training, metrics
* `matplotlib`, `seaborn` → Visualizations

---

## 👨‍💻 Author

**Ojas Patil** [GitHub Profile](https://github.com/Ojas-Patil26)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
