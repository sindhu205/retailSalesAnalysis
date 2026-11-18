
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


print("1. LOAD DATASET")
print()

print("RETAIL SALES ANALYSIS")
print()

df = pd.read_csv(
    "retaildata.csv",
    sep=None,
    engine="python",
    encoding="utf-8-sig"
)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("FIRST FIVE ROWS")
print(df.head())
print()

print("DATASET SHAPE")
print(df.shape)
print()

print("COLUMN NAMES")
print(df.columns.tolist())
print()

print("DATASET INFORMATION")
df.info()
print()

print("MISSING VALUES")
print(df.isnull().sum())
print()

print("DUPLICATE ROWS")
print(df.duplicated().sum())
print()


print("2. DATA CLEANING")
print()

required_columns = [
    "OrderID", "CustomerID", "OrderDate",
    "Category", "Quantity", "UnitPrice",
    "Discount", "Sales", "Cost", "Profit",
    "Region", "Segment"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in CSV file: {missing_columns}. "
        f"Available columns: {df.columns.tolist()}"
    )

# Convert date
df["OrderDate"] = pd.to_datetime(
    df["OrderDate"],
    dayfirst=True,
    errors="coerce"
)

# Convert numeric columns
numeric_columns = [
    "Quantity", "UnitPrice", "Discount",
    "Sales", "Cost", "Profit"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Remove invalid records
df = df.dropna(
    subset=[
        "OrderDate", "CustomerID",
        "OrderID", "Sales", "Category"
    ] + numeric_columns
)

# Remove duplicate records
df = df.drop_duplicates().copy()

# Create date columns
df["Year"] = df["OrderDate"].dt.year
df["Month"] = df["OrderDate"].dt.month
df["YearMonth"] = df["OrderDate"].dt.to_period("M")

print("CLEANED DATASET")
print()
print(df.head())
print()

print("CLEANED DATASET SHAPE")
print()
print(df.shape)
print()

print("3. SALES ANALYSIS")
print()

print("SALES ANALYSIS")
print()

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["OrderID"].nunique()
total_customers = df["CustomerID"].nunique()

print("Total Sales:", round(total_sales, 2))
print()
print("Total Profit:", round(total_profit, 2))
print()
print("Total Orders:", total_orders)
print()
print("Total Customers:", total_customers)
print()

print("Average Sales:")
print()
print(round(df["Sales"].mean(), 2))
print()

print("Average Profit:")
print()
print(round(df["Profit"].mean(), 2))
print()

print("CATEGORY-WISE SALES")
print()

category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print(category_sales)
print()

print("REGION-WISE SALES")
print()

region_sales = (
    df.groupby("Region")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print(region_sales)
print()

print("SEGMENT-WISE SALES")

segment_sales = (
    df.groupby("Segment")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print(segment_sales)
print()

print("4. DATA VISUALIZATION")
print()
# Category-wise sales chart

monthly_sales = (
    df.groupby("YearMonth")["Sales"]
    .sum()
    .sort_index()
)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 1. Category-wise Sales
sns.barplot(
    x=category_sales.index,
    y=category_sales.values,
    ax=axes[0, 0]
)
axes[0, 0].set_title("Category-wise Sales")
axes[0, 0].set_xlabel("Category")
axes[0, 0].set_ylabel("Total Sales")
axes[0, 0].tick_params(axis='x', rotation=45)


# 2. Region-wise Sales
sns.barplot(
    x=region_sales.index,
    y=region_sales.values,
    ax=axes[0, 1]
)
axes[0, 1].set_title("Region-wise Sales")
axes[0, 1].set_xlabel("Region")
axes[0, 1].set_ylabel("Total Sales")


# 3. Segment-wise Sales
sns.barplot(
    x=segment_sales.index,
    y=segment_sales.values,
    ax=axes[1, 0]
)
axes[1, 0].set_title("Segment-wise Sales")
axes[1, 0].set_xlabel("Segment")
axes[1, 0].set_ylabel("Total Sales")


# 4. Monthly Sales Trend
sns.lineplot(
    x=monthly_sales.index.astype(str),
    y=monthly_sales.values,
    marker="o",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Monthly Sales Trend")
axes[1, 1].set_xlabel("Month")
axes[1, 1].set_ylabel("Total Sales")
axes[1, 1].tick_params(axis='x', rotation=45)


# Main Title
plt.suptitle(
    "RETAIL SALES ANALYSIS DASHBOARD",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()
plt.show()

print("5. ANOVA ANALYSIS")
print()


# Separate sales based on category
category_groups = [
    group["Sales"].dropna().values
    for name, group in df.groupby("Category")
]

category_names = df["Category"].dropna().unique()

print("Categories:", category_names)
print()

# Check observations
if (
    len(category_groups) >= 2
    and all(len(group) >= 2 for group in category_groups)
):

    f_stat, p_value = f_oneway(*category_groups)

    print("F-Statistic:", round(f_stat, 4))
    print("P-Value:", round(p_value, 6))

    alpha = 0.05

    if p_value < alpha:
        print("Result: Reject Null Hypothesis")
        print(
            "There is a statistically significant "
            "difference in average sales between categories."
        )
    else:
        print("Result: Fail to Reject Null Hypothesis")
        print(
            "There is no statistically significant "
            "difference in average sales between categories."
        )

else:
    print("Not enough observations for ANOVA.")

print()

print("CUSTOMER COHORT ANALYSIS")
print()

# Find each customer's first purchase month
df["CohortMonth"] = (
    df.groupby("CustomerID")["OrderDate"]
    .transform("min")
    .dt.to_period("M")
)

# One record per customer per month
customer_month = df[
    ["CustomerID", "YearMonth", "CohortMonth"]
].drop_duplicates()

# Calculate cohort index
customer_month["CohortIndex"] = (
    (customer_month["YearMonth"].dt.year -
     customer_month["CohortMonth"].dt.year) * 12
    +
    (customer_month["YearMonth"].dt.month -
     customer_month["CohortMonth"].dt.month)
)

# Count unique customers
cohort_data = customer_month.groupby(
    ["CohortMonth", "CohortIndex"]
)["CustomerID"].nunique().reset_index()

# Create cohort table
cohort_table = cohort_data.pivot(
    index="CohortMonth",
    columns="CohortIndex",
    values="CustomerID"
)

# Calculate retention percentage
cohort_size = cohort_table[0]

retention = (
    cohort_table.divide(cohort_size, axis=0) * 100
)

print("COHORT CUSTOMER COUNT")
print()
print(cohort_table)
print()

print("CUSTOMER RETENTION PERCENTAGE")
print()
print(retention.round(2))
print()


# Cohort retention heatmap
plt.figure(figsize=(12, 7))

sns.heatmap(
    retention,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.5
)

plt.title("Customer Cohort Retention Analysis")
plt.xlabel("Months Since First Purchase")
plt.ylabel("Customer Cohort Month")
plt.tight_layout()
plt.show()


print("SALES FORECASTING")
print()

# Aggregate sales by month
monthly_sales = (
    df.groupby("YearMonth")["Sales"]
    .sum()
    .sort_index()
)

# Fill missing months
all_months = pd.period_range(
    start=monthly_sales.index.min(),
    end=monthly_sales.index.max(),
    freq="M"
)

monthly_sales = monthly_sales.reindex(
    all_months,
    fill_value=0
)

# Prepare forecasting dataset
forecast_df = pd.DataFrame({
    "Month": np.arange(len(monthly_sales)),
    "Sales": monthly_sales.values
})

X = forecast_df[["Month"]]
y = forecast_df["Sales"]

# Check minimum data
if len(forecast_df) < 3:
    raise ValueError(
        "At least 3 months of sales data are required."
    )

# Split into training and testing data
split_index = max(
    2,
    int(len(forecast_df) * 0.8)
)

if split_index >= len(forecast_df):
    split_index = len(forecast_df) - 1

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# Train model for evaluation
model = LinearRegression()

model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)


# =====================================================
# 8. MODEL EVALUATION
# =====================================================

print("\n========== MODEL EVALUATION ==========")

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))

if len(y_test) >= 2:
    print(
        "R2 Score:",
        round(r2_score(y_test, y_pred), 2)
    )
else:
    print(
        "R2 Score: Not meaningful "
        "with only one test observation."
    )


print("9. FORECAST NEXT THREE MONTHS")
print()

# Train final model using all available data
final_model = LinearRegression()

final_model.fit(X, y)

# Predict next 3 months
future_months = pd.DataFrame({
    "Month": np.arange(
        len(monthly_sales),
        len(monthly_sales) + 3
    )
})

future_sales = final_model.predict(future_months)

future_dates = pd.period_range(
    start=monthly_sales.index[-1] + 1,
    periods=3,
    freq="M"
)

forecast_result = pd.DataFrame({
    "Month": future_dates.astype(str),
    "Predicted_Sales": future_sales.round(2)
})

print("NEXT THREE MONTHS SALES FORECAST")
print()

print(forecast_result)

print("10. ACTUAL VS FORECASTED SALES CHART")
print()

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_sales.index.astype(str),
    monthly_sales.values,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    future_dates.astype(str),
    future_sales,
    marker="o",
    linestyle="--",
    label="Forecasted Sales"
)

plt.title("Retail Sales Forecasting")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

print("11. PROJECT COMPLETED")
print()

print("RETAIL SALES ANALYSIS COMPLETED")
print()
