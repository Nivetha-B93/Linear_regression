# # Mobile Price Prediction using Linear Regression

A simple linear regression project that predicts mobile phone prices based on hardware specifications such as RAM, display resolution (PPI), internal memory, and rear camera quality.

## Project Overview

This project explores a mobile phone specifications dataset, identifies which features correlate most strongly with price, and builds a linear regression model to predict price from those features. It walks through the full, standard data science workflow: data exploration, correlation analysis, feature selection, model training, evaluation, and interpretation of results.

## Dataset

- **Source:** [mobile_price.csv](https://raw.githubusercontent.com/ArchanaInsights/Datasets/main/mobile_price.csv)
- **Size:** 161 rows × 14 columns
- **Features include:** `Product_id`, `Price`, `Sale`, `weight`, `resoloution`, `ppi`, `cpu core`, `cpu freq`, `internal mem`, `ram`, `RearCam`, `Front_Cam`, `battery`, `thickness`
- **Target variable:** `Price`

No missing values were present in the dataset.

## Workflow

1. **Data Exploration**
   - Inspected dataset shape, column names, data types, and missing values
   - Generated statistical summaries (`df.describe()`) for numerical features

2. **Correlation Analysis**
   - Built a correlation heatmap across all numerical features
   - Identified the top 4 features most correlated with `Price`:
     - `ram` (0.897)
     - `ppi` (0.818)
     - `internal mem` (0.777)
     - `RearCam` (0.740)
   - Visualized each of these features against `Price` using scatter plots

3. **Data Preparation**
   - Selected `ram`, `ppi`, `internal mem`, and `RearCam` as input features (`X`) and `Price` as the target (`y`)
   - Split the data into training (80%) and testing (20%) sets using `train_test_split`

4. **Model Building & Training**
   - Trained a `LinearRegression` model (scikit-learn) on the training set

5. **Prediction & Evaluation**
   - Generated predictions on the test set
   - Reported model intercept and coefficients
   - Evaluated performance using:
     - **Mean Absolute Error (MAE):** ~252.51
     - **Mean Squared Error (MSE):** ~92,877.41
     - **R² Score:** ~0.811

6. **Feature Scaling Experiment**
   - Re-trained the model on standardized features (`StandardScaler`) to compare performance

## Key Insights

- RAM and internal memory show the strongest positive relationships with price.
- PPI (display density) and rear camera quality are also positively correlated with price, though with more scatter.
- The model explains roughly 81% of the variance in mobile phone prices using just four features.
- Remaining variance is likely driven by other factors (brand, additional specs, market positioning) not captured in this feature set.

## Potential Improvements

- Incorporate additional relevant features
- Handle outliers in the data
- Try feature engineering / transformations
- Experiment with other regression algorithms (e.g., Ridge, Random Forest, Gradient Boosting)
- Use cross-validation for more robust performance estimates

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook Linear_regression.ipynb
   ```

## Author

Nivetha B
- [LinkedIn](https://www.linkedin.com/in/nivetha-b-822545318)
- [GitHub](https://github.com/Nivetha-B93)
