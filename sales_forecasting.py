import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("SampleSuperstore.csv", encoding="latin1")

print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

print("\nFirst 5 Rows:")
print(df.head())

# =====================================================
# 2. DATA EXPLORATION
# =====================================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# 3. DATA CLEANING
# =====================================================

df.dropna(inplace=True)

# Convert Order Date to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# =====================================================
# 4. FEATURE ENGINEERING
# =====================================================

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day

print("\nFeature Engineering Completed")

# =====================================================
# 5. MONTHLY SALES ANALYSIS
# =====================================================

monthly_sales = df.groupby('Month')['Sales'].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot(kind='bar')

plt.title("Monthly Sales Analysis")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig("monthly_sales.png")
plt.show()

print("monthly_sales.png saved")

# =====================================================
# 6. CATEGORY SALES ANALYSIS
# =====================================================

category_sales = df.groupby('Category')['Sales'].sum()

plt.figure(figsize=(8,5))
category_sales.plot(kind='bar')

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig("category_sales.png")
plt.show()

print("category_sales.png saved")

# =====================================================
# 7. DAILY SALES DATASET
# =====================================================

daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

# Create Time Index
daily_sales['Time'] = range(len(daily_sales))

print("\nDaily Sales Dataset Created")

# =====================================================
# 8. FEATURES AND TARGET
# =====================================================

X = daily_sales[['Time']]
y = daily_sales['Sales']

# =====================================================
# 9. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# =====================================================
# 10. MODEL TRAINING
# =====================================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed")

# =====================================================
# 11. PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# 12. MODEL EVALUATION
# =====================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))

# =====================================================
# 13. ACTUAL VS PREDICTED GRAPH
# =====================================================

plt.figure(figsize=(12,6))

plt.plot(
    y_test.values,
    label='Actual Sales'
)

plt.plot(
    y_pred,
    label='Predicted Sales'
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Time")
plt.ylabel("Sales")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

print("actual_vs_predicted.png saved")

# =====================================================
# 14. FUTURE FORECASTING
# =====================================================

future = pd.DataFrame({
    'Time': range(
        len(daily_sales),
        len(daily_sales) + 30
    )
})

future_sales = model.predict(future)

# =====================================================
# 15. SALES FORECAST GRAPH
# =====================================================

plt.figure(figsize=(12,6))

plt.plot(
    daily_sales['Time'],
    daily_sales['Sales'],
    label='Historical Sales'
)

plt.plot(
    future['Time'],
    future_sales,
    'r--',
    linewidth=3,
    label='Future Forecast'
)

plt.title("Sales Forecasting using Linear Regression")
plt.xlabel("Time")
plt.ylabel("Sales")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("sales_forecast.png")
plt.show()

print("sales_forecast.png saved")

# =====================================================
# 16. BUSINESS INSIGHTS
# =====================================================

print("\n" + "=" * 50)
print("BUSINESS INSIGHTS")
print("=" * 50)

print("1. Sales fluctuate significantly across time.")
print("2. Monthly sales analysis identifies peak sales periods.")
print("3. Category analysis identifies top-performing categories.")
print("4. Forecasting helps estimate future demand.")
print("5. Businesses can optimize inventory and staffing.")
print("6. Data-driven decisions reduce operational risks.")

print("\nProject Completed Successfully!")