import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
sales_data = pd.read_csv('/Users/apple/Desktop/Uni/BEMM457 Topics in Business Analytics/assess data/archive (13)/Online_Sales.csv')
tax_data = pd.read_csv('/Users/apple/Desktop/Uni/BEMM457 Topics in Business Analytics/assess data/archive (13)/Tax_amount.csv')
discount_coupon_data = pd.read_csv('/Users/apple/Desktop/Uni/BEMM457 Topics in Business Analytics/assess data/archive (13)/Discount_Coupon.csv')
marketing_spend_data = pd.read_csv('/Users/apple/Desktop/Uni/BEMM457 Topics in Business Analytics/assess data/archive (13)/Marketing_Spend.csv')
customers_data = pd.read_csv('/Users/apple/Desktop/Uni/BEMM457 Topics in Business Analytics/assess data/archive (13)/CustomersData.csv')

# Data cleaning and preparation

# Convert 'Transaction_Date' to datetime
sales_data['Transaction_Date'] = pd.to_datetime(sales_data['Transaction_Date'])

# Extract the month from the 'Transaction_Date' and create a new 'Month' column
sales_data['Month'] = sales_data['Transaction_Date'].dt.month
# Check 
print(sales_data.head())

# Use number replace month
month_mapping = {
    'Jan': 1,'Feb': 2,'Mar': 3,'Apr': 4,'May': 5,'Jun': 6,'Jul': 7, 'Aug': 8, 'Sep': 9,'Oct': 10,'Nov': 11,'Dec': 12
}
# Assuming the column with month names is called 'Month'
discount_coupon_data['Month'] = discount_coupon_data['Month'].replace(month_mapping)

# Merge with other datasets
merged_data1 = pd.merge(sales_data, tax_data, on='Product_Category', how='left')
merged_data2 = pd.merge(merged_data1, discount_coupon_data, on=['Product_Category','Month'], how='left')
merged_data3 = pd.merge(merged_data2, customers_data, on='CustomerID', how='left')

# Drop duplicates
merged_data3.drop_duplicates(inplace=True)

# Convert 'GST' from object to float
column_name = 'GST' 
if merged_data3[column_name].dtype == 'object':
    # Convert percentage string to float directly in merged_data3
    merged_data3[column_name] = merged_data3[column_name].str.replace('%', '').astype(float) / 100
else:
    # If 'GST' is already a number (float or int) but formatted as a percentage, divide by 100 directly in merged_data3.
    merged_data3[column_name] = merged_data3[column_name].astype(float) / 100


# Convert 'Discount_pct( Discount Percentage for given coupon)' from percentage value to a decimal which can use to calculate.
column_name2 = 'Discount_pct' 
merged_data3[column_name2] = merged_data3[column_name2].astype(float) / 100


merged_data3['Discount_pct'].fillna(0, inplace = True)

# Calculate invoice value for each transaction 
merged_data3['Invoice_Value'] = ((merged_data3['Quantity'] * merged_data3['Avg_Price']) * 
                                (1.0 - merged_data3['Discount_pct']) * 
                                (1.0 + merged_data3['GST'])) + merged_data3['Delivery_Charges']

# Calculate invoice value for each customer each date
invoice_value_each_customer_each_date = merged_data3. groupby (['CustomerID','Transaction_Date'])['Invoice_Value'].sum().reset_index()
print(invoice_value_each_customer_each_date.head())


 #Perform customer segmentation
from datetime import datetime
current_date = datetime.now()

# RFM
rfm =merged_data3.groupby('CustomerID').agg({
    'Transaction_Date': lambda x: (current_date - x.max()).days,
    'CustomerID': 'count',
    'Invoice_Value': 'sum'
}).rename(columns={'Transaction_Date': 'Recency', 
                   'CustomerID': 'Frequency', 
                   'Invoice_Value': 'Monetary'})
print (rfm)
rfm.describe()

from sklearn.preprocessing import QuantileTransformer

# Creating a copy of the dataset for manipulation
rfm_scores = rfm.copy()

# Using QuantileTransformer for scoring, it will assign scores based on quantiles, from 1 to 5
qt = QuantileTransformer(n_quantiles=100, random_state=0)

# Scoring for Recency: Inverse scoring because lower recency is better
rfm_scores['R_Score'] = qt.fit_transform(rfm[['Recency']]*-1)
rfm_scores['R_Score'] = rfm_scores['R_Score'].apply(lambda x: x * 4 + 1)  # Scaling to 1-5 range

# Scoring for Frequency and Monetary: Direct scoring because higher values are better
rfm_scores['F_Score'] = qt.fit_transform(rfm[['Frequency']])
rfm_scores['F_Score'] = rfm_scores['F_Score'].apply(lambda x: x * 4 + 1)  # Scaling to 1-5 range

rfm_scores['M_Score'] = qt.fit_transform(rfm[['Monetary']])
rfm_scores['M_Score'] = rfm_scores['M_Score'].apply(lambda x: x * 4 + 1)  # Scaling to 1-5 range

# Rounding the scores
rfm_scores = rfm_scores.round({'R_Score': 0, 'F_Score': 0, 'M_Score': 0})

# Displaying the first few rows with scores
rfm_scores.head()


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Use K-means to cluster analysis
rfm_scores2 = rfm_scores.copy()

# Extracting the RFM values
X = rfm_scores2[['R_Score', 'F_Score', 'M_Score']]  

# Using the Elbow Method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the results of the Elbow Method
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method',fontsize=35)
plt.xlabel('Number of clusters',fontsize=25)
plt.ylabel('WCSS',fontsize=25)  # Within cluster sum of squares
plt.xticks(rotation=0, fontsize=20) 
plt.yticks(rotation=0, fontsize=20) 
plt.show()

# Assuming rfm_scores2 is pre-loaded and contains 'R_Score', 'F_Score', 'M_Score', 'CustomerID', and 'Cluster'
# Fit the KMeans model with the optimal number of clusters
number_of_clusters = 4  # Determined from user's elbow plot
model = KMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
rfm_scores2['Cluster'] = model.fit_predict(rfm_scores2[['R_Score', 'F_Score', 'M_Score']])

# Counting the number of customers in each cluster
cluster_counts = rfm_scores2['Cluster'].value_counts().sort_index()

# Plotting the distribution of clusters
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=rfm_scores2, order=[0, 1, 2, 3])
plt.title('Cluster Distribution', fontsize=15)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0, fontsize=10) 
plt.yticks(rotation=0, fontsize=10)
plt.show()

# Assuming rfm_scores2 is pre-loaded and contains 'R_Score', 'F_Score', 'M_Score', 'CustomerID', and 'Cluster'

# Displaying cluster counts
cluster_counts = rfm_scores2['Cluster'].value_counts()
print(cluster_counts)

# Calculate mean RFM values for each cluster and count the number of customers in each
cluster_summary = rfm_scores2.groupby('Cluster').agg({
    'R_Score': 'mean', 
    'F_Score': 'mean', 
    'M_Score': 'mean'
}).round(1)

print(cluster_summary)  # It's a good practice to print the summary for verification

# Melt the data for the distribution plot
df_melt = pd.melt(rfm_scores2.reset_index(),
                  id_vars=['CustomerID', 'Cluster'],
                  value_vars=['R_Score', 'F_Score', 'M_Score'],
                  var_name='Attribute',
                  value_name='Value')

# Line plot to show the distribution of the RFM values with the same color palette
plt.figure(figsize=(10, 6))
palette = sns.color_palette("husl", len(rfm_scores2['Cluster'].unique()))  # Define a palette
sns.lineplot(x='Attribute', y='Value', hue='Cluster', data=df_melt, palette=palette)
plt.title('RFM Values by Cluster', fontsize=15)
plt.xlabel('Attribute', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Cluster', title_fontsize='13', fontsize='10')
plt.grid(True)
plt.show()


