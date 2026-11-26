# Retail Sales Analysis, Customer Cohort Analysis & Sales Forecasting

## Project Overview
This project focuses on analyzing retail sales data using Python. It performs data cleaning, Exploratory Data Analysis (EDA), category-wise and region-wise sales analysis, ANOVA testing, customer cohort analysis, and future sales forecasting using Linear Regression.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn

## Dataset
The project uses `retaildata.csv` with the following columns:

- OrderID
- CustomerID
- OrderDate
- Category
- Quantity
- UnitPrice
- Discount
- Sales
- Cost
- Profit
- Region
- Segment

## Project Workflow

1. Load the retail sales dataset.
2. Perform data inspection.
3. Handle missing values and duplicate records.
4. Perform data cleaning and preprocessing.
5. Calculate total sales, total profit, orders, and customers.
6. Perform category-wise, region-wise, and segment-wise sales analysis.
7. Create data visualizations.
8. Perform ANOVA statistical analysis.
9. Perform customer cohort analysis.
10. Calculate customer retention percentages.
11. Build a Linear Regression model for sales forecasting.
12. Evaluate model performance.
13. Forecast sales for the next three months.

## Data Visualization

The project includes the following visualizations:

1. **Category-wise Sales** – Compares sales across different product categories.
2. **Region-wise Sales** – Analyzes sales across different regions.
3. **Segment-wise Sales** – Displays sales based on customer segments.
4. **Monthly Sales Trend** – Shows monthly sales performance.
5. **Customer Cohort Heatmap** – Visualizes customer retention percentages.
6. **Sales Forecast Chart** – Compares actual sales with forecasted sales.

## Statistical Analysis – ANOVA

ANOVA (Analysis of Variance) is used to identify whether there is a statistically significant difference in average sales between different product categories.

- F-Statistic
- P-Value
- Significance Level (0.05)

## Customer Cohort Analysis

Cohort analysis is used to understand customer purchasing behavior and retention.

- Identifies each customer's first purchase month.
- Groups customers based on their first purchase.
- Calculates monthly customer retention.
- Displays retention percentages using a heatmap.

## Machine Learning – Sales Forecasting

**Algorithm: Linear Regression**

Linear Regression is used to analyze historical monthly sales data and
