**Supply Chain Inventory Optimization**

An end-to-end machine learning system for supply chain analytics built on the
DataCo Smart Supply Chain dataset containing 180,519 orders across global markets.


**Problem Statement**

Organizations operating large-scale supply chains face four core inventory problems:


Overstocking products that do not sell, tying up working capital
Understocking high-demand products, leading to lost sales
Failing to predict delivery delays before they occur
Carrying dead stock that has not sold in months, wasting warehouse space


This project addresses all four problems using a combination of machine learning
models and classical inventory optimization formulas, producing a per-product
action recommendation for supply chain managers.


**Dataset**

DataCo Smart Supply Chain for Big Data Analysis


Source: Kaggle
Link: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
Rows: 180,519 orders
Features: 53 columns
Markets covered: Africa, Europe, LATAM, Pacific Asia, USCA
Date range: January 2015 to January 2018



**Project Modules**

Module 1 — Demand Forecasting

Predicts daily units sold per product using historical order data.

Algorithm: XGBoost Regressor
Features: 7, 14, 21, 30-day lag features, 7, 14, 30-day rolling averages,
day of week, month, quarter, week number
Train/test split: Time-based — last 60 days held out as test set
Evaluation: MAE, RMSE, SMAPE, Correlation
Output: demand_forecast_output.csv


Module 2 — Lead Time Analysis

Analyzes the gap between scheduled and actual delivery days to quantify
lead time variability per shipping mode, market, and supplier.

Method: Statistical analysis of Days for shipping (real) vs
Days for shipment (scheduled)
Output: Lead time mean and standard deviation per product,
used as input to safety stock formula


Module 3 — Supply Delay Prediction

Predicts whether an individual shipment will arrive late before it is dispatched,
enabling proactive intervention.

Algorithm: XGBoost Classifier
Target: Late_delivery_risk (binary — 0 on time, 1 late)
Features: Shipping mode, market, customer segment, scheduled delivery days,
order quantity, discount rate, profit ratio, order month, day of week, week number
Train/test split: 80/20 stratified split
ROC-AUC: 0.7375
Output: delay_predictions.csv with per-order delay probability


Module 4 — Dead Stock Detection

Identifies products that are not selling and recommends action — liquidate,
discount, maintain, or monitor.


Method 1 (Rule-based): Flags products where days since last sale exceeds
threshold AND weekly sales velocity falls below threshold
Method 2 (ML-based): Isolation Forest anomaly detection on sales velocity,
recency, quantity sold, and order count
Segmentation: ABC-XYZ analysis

ABC classifies products by cumulative revenue contribution (Pareto principle)
XYZ classifies products by demand variability using coefficient of variation

Output: dead_stock_output.csv with per-product recommendation


Module 5 — Inventory Optimization Engine

Combines outputs from all three models to compute per-product inventory parameters
and generate final action recommendations.

**Formulas used:**

Safety Stock = Z x sqrt(avg_lead_time x std_demand^2 + avg_demand^2 x std_lead_time^2)

Reorder Point = avg_daily_demand x avg_lead_time + safety_stock

EOQ = sqrt(2 x annual_demand x ordering_cost / holding_cost_per_unit)

Where Z = 1.65 (95% service level), ordering cost = 50, holding rate = 20% of unit price.

Output: final_inventory_recommendations.csv with columns for safety stock,
reorder point, EOQ, and action per product.

**How to Run:**

Step 1 — Install dependencies

pip install -r requirements.txt

Step 2 — Download the dataset

Download DataCo Smart Supply Chain dataset from Kaggle:
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Place the CSV file inside the notebooks/ folder.

Step 3 — Run notebooks in order

Open Jupyter Notebook and run notebooks 01 through 06 in sequence.
Each notebook saves its output CSV which the next notebook reads.

Step 4 — Launch the dashboard

cd notebooks
streamlit run app.py

The dashboard opens at http://localhost:8501

**Business Impact:**

Delivery delay model with 0.7375 AUC enables proactive intervention before
shipments go late, reducing customer complaints
Dead stock detection identifies products for liquidation and discounting,
recovering tied-up capital
Per-product reorder point and EOQ calculations prevent both stockouts
and overordering across the full product catalog
ABC-XYZ segmentation ensures inventory investment is prioritized toward
high-value, high-consistency products

**Author**

**Ankita S**
