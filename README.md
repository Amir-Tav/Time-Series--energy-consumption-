# Energy Consumption Prediction with XGBoost

Welcome to the **Energy Consumption Prediction** project! This project focuses on predicting **hourly energy consumption** using historical data. We apply the **XGBoost algorithm** to build a robust model capable of forecasting energy usage, which can help optimize energy management and reduce costs.

This project demonstrates how machine learning can be used to analyze time series data and make predictions based on historical patterns, such as hour of day, day of the week, and seasonal trends.

### Key Highlights:
- **Data Source**: The dataset used in this project is publicly available and can be downloaded from [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).
- **Model**: We used **XGBoost**, a powerful and scalable gradient boosting algorithm, to forecast energy consumption.
- **Features**: The model leverages various time-based features like hour, day of the week, month, and more to capture important temporal trends.
- **Evaluation**: The model's performance is assessed using **Root Mean Squared Error (RMSE)**, a standard metric for regression tasks.

### Problem Description:
The goal of this project is to predict **hourly energy consumption** for a particular region using past data. By understanding the factors influencing energy usage, the model can forecast future consumption, allowing utilities to better manage their supply and demand.

### Key Steps in the Process:

1. **Data Loading and Preprocessing**:
    - We start by loading the dataset and preparing it for analysis. The `Datetime` column is converted into the DataFrame's index for time series analysis.

    ```python
    df = pd.read_csv('data.csv')
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)
    ```

---

2. **Feature Engineering**:
    - To help the model understand time-based trends, we create several features like **hour**, **day of the week**, **month**, and **year**. These features are essential for capturing seasonal and hourly patterns.

    ```python
    def create_features(df):
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        return df
    ```

---

3. **Splitting Data into Training and Test Sets**:
    - The dataset is split into training and test sets (80%-20% split). We visualize the split to gain insights into how the training data relates to the test data.

    ```python
    train = df.loc[df.index < '01-01-2015']
    test = df.loc[df.index >= '01-01-2015']
    ```

---

4. **Model Creation and Training**:
    - The **XGBoost** model is then trained using the engineered features. We use early stopping to prevent overfitting and monitor performance on both the training and test sets.

    ```python
    reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01)
    reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50)
    ```

---

5. **Forecasting**:
    - Once the model is trained, we apply it to the test data to make predictions. These predictions are visualized alongside the actual data for comparison.

    ```python
    test['prediction'] = reg.predict(X_test)
    ```

---

6. **Model Evaluation**:
    - The model's performance is evaluated using **RMSE** (Root Mean Squared Error). A lower RMSE indicates better prediction accuracy.

    ```python
    score = np.sqrt(mean_squared_error(test['PJME_MW'], test['prediction']))
    print(f'RMSE Score: {score:.2f}')
    ```

    - We also calculate the error over time and identify periods where the model made the largest errors.

    ```python
    test['error'] = np.abs(test[TARGET] - test['prediction'])
    test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)
    ```

---

### Results:
- The **XGBoost** model successfully predicts energy consumption, but there is room for improvement. The model's **RMSE score** provides an indication of how accurately it performs on unseen data.
- Visualizations show the predicted energy consumption over time, highlighting both the true values and the model's forecasts.

### Conclusion:
The **XGBoost model** provides decent predictions of hourly energy consumption but could benefit from further optimization. Future improvements could include:
- **Tuning the hyperparameters**: This would help in reducing the RMSE and improving prediction accuracy.
- **Adding more features**: Including external factors (weather data, holidays, etc.) might provide more context for the model, improving its performance.
- **Exploring other machine learning models**: Trying different algorithms could help achieve better results.

By refining the model and enhancing the feature set, we could significantly improve the accuracy of energy consumption predictions, making it a valuable tool for energy management.

---

