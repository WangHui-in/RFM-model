 <img width="647" alt="image" src="https://github.com/user-attachments/assets/e2e953b8-ee8d-4339-af43-372876a03150">

## RFM Model Implementation for E-Commerce

This repository features Python code to implement the RFM (Recency, Frequency, Monetary) model for customer segmentation in an E-commerce setting. The Python script available in this repository is designed to process customer transaction data to calculate RFM scores, segment customers, and identify key customer groups based on their purchasing patterns.

## Data Source:

The data used in this project is sourced from Kaggle, specifically tailored for generating marketing insights for an E-commerce company. You can access the dataset https://www.kaggle.com/datasets/rishikumarrajvansh/marketing-insights-for-e-commerce-company 


## Project Purpose:

RFM Analysis: Employ the RFM model to evaluate and segment customers based on their transaction history.
Customer Insight: Gain insights into customer behaviors and categorize them into various segments for targeted marketing strategies.
Code Demonstration: Showcase how to implement RFM scoring and segmentation using Python.

## Code Description
The provided RFM.py script includes the following key operations:

1.Data Preparation:
Load and preprocess the transaction data from the dataset.
Extract necessary attributes such as transaction date, customer ID, and payment amount.

2.Calculation of RFM Metrics:
Calculate 'Recency' as the number of days since the last purchase.
Compute 'Frequency' as the total number of transactions.
Determine 'Monetary' as the total money spent by the customer.

3.Customer Segmentation:
Apply Quantile-based segmentation to classify customers into segments like diamond, gold, silver, or bronze.
This segmentation helps in identifying high-value customers and tailoring marketing efforts accordingly.

4.Features:
RFM Score Calculation: Script to compute RFM scores for each customer.
Customer Segmentation: Segmentation logic that categorizes customers based on their RFM scores.
Data Visualization: (If applicable) Visualization scripts to plot distributions and segmentations of RFM scores.

## File Structure

RFM.py: The Python script that implements RFM calculation and segmentation.
Contributing
If you have ideas for improving the script or extending its functionality, please contribute by submitting a pull request or opening an issue in this repository.

## License

The project is released under the CC0: Public Domain license, allowing free use, modification, and distribution.

## Contact

For questions or feedback about the script, please contact Huiying Wang at wanghuiying95@gmail.com.
