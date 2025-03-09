# Rainfall Prediction Using Ensemble SVR

## 📌 Overview
This project is part of the thesis titled **"Application of Ensemble Methods for Kernel-Based Multivariate Time Series in Rainfall Forecasting (Case Study: Perak I Meteorological Station)"**. This research utilizes the **Ensemble Support Vector Regression (SVR)** method to forecast rainfall based on normalized multivariate weather data.

## 🔍 Research Methodology
- **Data**: Weather data from (BMKG) Perak I Meteorological Station with features **Temperature, Humidity, Wind Speed, and Rainfall**.
- **Exploratory Data Analysis**: Exploring and analyzing the dataset using visualization and statistical techniques, such as:
  - **Descriptive Statistics**: Calculating summary statistics like mean, median, and standard deviation to analyze central tendencies and data distribution.
  - **Data Visualization**: Using various types of graphs, such as histograms, scatter plots, and box plots, to visualize data distribution and relationships between variables.
- **Preprocessing**:
  - **Interpolation** to fill missing values.
  - **Normalization** to standardize feature scales.
  - **Outlier Detection and Handling** using Z-score.
  - **Sliding Window for Input and Output Determination** using ACF and PACF.
- **Data Splitting**: The dataset is split with an 80:20 ratio, where 80% is used for training and the remaining 20% for testing.
- **Modeling**:
  - **Bootstrap Sampling** to create multiple training data subsets with 5, 10, and 20 estimators.
  - **Support Vector Regression (SVR) with Linear, RBF, and Polynomial kernels**.
  - **Hyperparameter Optimization using GridSearch**.
- **Evaluation**: Using **MAE and RMSE** metrics.

## 📂 Repository Structure
```
├── data/                      # Preprocessed rainfall dataset
├── models/                    # Trained models
├── notebooks/                 # Jupyter Notebooks for exploration and experiments
├── scripts/                   # Scripts for preprocessing, training, and evaluation
├── results/                   # Model predictions and evaluation results
├── README.md                  # Project documentation
```

## 🚀 Installation and Usage
### 1. Clone Repository
```bash
git clone https://github.com/rizkyjisantt-dev/Rainfall-Prediction-Using-Ensemble-SVR.git
cd Rainfall-Prediction-Using-Ensemble-SVR
```
### 2. Install Dependencies
Use Python 3.8+ and install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run Experiments
Use Jupyter Notebook or run Python scripts to execute experiments:
```bash
jupyter notebook
```
Or run the model directly:
```bash
python scripts/Prediksi.py
```

## 📊 Results and Analysis
- The best model is selected based on **MAE and RMSE** values.
- A comparison is made between **Standard SVR and Ensemble SVR** to assess performance improvements.
- Autocorrelation analysis is conducted to understand rainfall relationships.

## 🏆 Contributions and License
This project is open-source. If you would like to contribute or discuss, feel free to submit a **pull request** or reach out via **[GitHub Issues](https://github.com/rizkyjisantt-dev/rainfall-prediction-using-ensemblesvr/issues)**.

---
✉️ **Developed by [Rizky Jisantt](https://github.com/rizkyjisantt-dev/)**



