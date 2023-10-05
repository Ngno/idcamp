import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Data Gathering
#orders data
orders_df=pd.read_csv("data\orders_dataset.csv")
##customers data
customers_df=pd.read_csv("data\customers_dataset.csv")
#geolocation data
geolocation_df=pd.read_csv("data\geolocation_dataset.csv")
#order items data
ordered_items_df=pd.read_csv("data\order_items_dataset.csv")
#seller data
sellers_df=pd.read_csv("data\sellers_dataset.csv")
#item review data
review_df=pd.read_csv("data\order_reviews_dataset.csv")
#payment data
payment_df=pd.read_csv("data\order_payments_dataset.csv")
#products data
products_df=pd.read_csv("data\products_dataset.csv")
#english category
english_category_df=pd.read_csv("data\product_category_name_translation.csv")

#DATA CLEANING
#orders_df
#convert data date ke datetime type
orders_date_columns = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

for col in orders_date_columns:
    orders_df[col] = pd.to_datetime(orders_df[col])

# Extract the year and month into separate columns
orders_df['year_order_purchase'] = orders_df['order_purchase_timestamp'].dt.year
orders_df['month_order_purchase'] = orders_df['order_purchase_timestamp'].dt.strftime('%Y-%m')


#membuat tabel 2018
orders_df_2018 = orders_df[orders_df['year_order_purchase'] == 2018]

# Create a copy of the DataFrame 
orders_df_2018 = orders_df_2018.copy()

#customers_df
#mengubah nama kolom 
customers_df.rename(columns={'customer_zip_code_prefix': 'zip_code_prefix'}, inplace=True)

#geolocation_df
#Drop column city dan state karena tabel lain sudah memiliki keterangan city dan state
geolocation_df.drop(columns=['geolocation_city', 'geolocation_state'], inplace=True)
#ubah nama kolom
geolocation_df.rename(columns={'geolocation_zip_code_prefix': 'zip_code_prefix'}, inplace=True)
#drop duplicate data
geolocation_df.drop_duplicates(inplace=True)
#kita ambil mean dari lat dan lng untuk mewakili 1 zip code prefix
# Group by 'zip_code_prefix'
zip_grouped = geolocation_df.groupby('zip_code_prefix')

# Calculate the mean latitude and longitude for each ZIP code
geolocation_df = zip_grouped.agg({
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean'
}).reset_index()

# Rename the columns
geolocation_df.columns = ['zip_code_prefix', 'mean_latitude', 'mean_longitude']

#ordered_items_df
#Convert order_item_id dari int ke object 
ordered_items_df['order_item_id'] = ordered_items_df['order_item_id'].astype(str)
#Convert shipping_limit_date object ke datetime
ordered_items_df['shipping_limit_date'] = pd.to_datetime(ordered_items_df['shipping_limit_date'])

#review_df
#drop unnecessary free-text columns 
# List of column names to drop
columns_to_drop = ['review_comment_title', 'review_comment_message', 'review_creation_date', 'review_answer_timestamp']
# Drop the columns
review_df.drop(columns=columns_to_drop, inplace=True)

#products_df
# Fill missing values in the 'product_category_name' column with "noname"
products_df['product_category_name'].fillna('noname', inplace=True)
# Fill missing values in float columns with 0
columns_float = ['product_name_lenght', 'product_description_lenght', 'product_photos_qty']
products_df[columns_float] = products_df[columns_float].fillna(0)

#english_category_df
#untuk mengimbangi noname di product name,
# Create a new DataFrame with the 'noname' values
new_row = pd.DataFrame({'product_category_name': ['noname'],
                        'product_category_name_english': ['noname']})

# Concatenate the new row DataFrame with the existing DataFrame
english_category_df = pd.concat([english_category_df, new_row], ignore_index=True)


#EDA
#Create a new column with 4 categories: 'on process', 'delivered', 'cancelled', 'unavailable'
# Define category
def categorize_order(status):
    if status in ['approved', 'created', 'invoiced', 'processing', 'shipped']:
        return 'on process'
    elif status == 'delivered':
        return 'delivered'
    elif status == 'canceled':
        return 'cancelled'
    else:
        return 'unavailable'
# Apply the categorize_order function to create a new 'category' column
orders_df_2018['category'] = orders_df_2018['order_status'].apply(categorize_order)

#lama proses penjualan
# Calculate delta between 'order_approved_at' and 'order_purchase_timestamp' in seconds
orders_df_2018['seconds_purchase_to_approved'] = (orders_df_2018['order_approved_at'] - orders_df_2018['order_purchase_timestamp']).dt.total_seconds()

# Calculate delta between 'order_delivered_carrier_date' and 'order_approved_at' in seconds
orders_df_2018['seconds_approved_to_delivered'] = (orders_df_2018['order_delivered_carrier_date'] - orders_df_2018['order_approved_at']).dt.total_seconds()


#Merge dfs
# 1. Merge orders_df with ordered_items_df on order_id
orders_df_2018 = orders_df_2018.merge(ordered_items_df, on='order_id', how='left')

#cek waktu pengiriman seller
# Calculate the difference in days between 'order_delivered_carrier_date' and 'shipping_limit_date'
orders_df_2018['delivery_delay_days'] = (orders_df_2018['order_delivered_carrier_date'] - orders_df_2018['shipping_limit_date']).dt.days
# Create a new column 'delivery_status' where 1 indicates late delivery and 0 indicates on-time or early delivery
orders_df_2018['late_delivery'] = orders_df_2018['delivery_delay_days'].apply(lambda x: 1 if x > 0 else 0)

#reduce table dengan mengambil columns yang dibutuhkan saja
columns_to_keep = [
    'order_id', 'customer_id', 'month_order_purchase', 'category', 'order_purchase_timestamp',
    'seconds_purchase_to_approved', 'seconds_approved_to_delivered',
    'order_item_id', 'product_id', 'seller_id', 'delivery_delay_days', 'late_delivery',
    'price', 'freight_value',
]
orders_2018 = orders_df_2018[columns_to_keep].copy()

#2. Merge customers_df with geolocation_df on zip code prefix 
customers_df = customers_df.merge(geolocation_df, on='zip_code_prefix', how='left')
#3. Merge payment_df dan order_review on order_id as payment_review
payment_review = payment_df.merge(review_df, on='order_id', how='left')
#4. Merge products_df dan english_category_df on product_category_name as products_df
products_df = products_df.merge(english_category_df, on='product_category_name', how='left')
products_df.drop(columns='product_category_name', inplace=True)
#5. Merge orders_2018 and sellers_df on seller_id as df_2018
df_2018=orders_2018.merge(sellers_df, on='seller_id', how='left')
#6. Merge df_2018 and payment_review as df_2018
df_2018=df_2018.merge(payment_review, on='order_id', how='left')
#7. Merge df_2018 and products_df on product_id as df_2018
df_2018=df_2018.merge(products_df, on='product_id', how='left')
#8. Merge df_2018 and customers_df as df_2018
df_2018=df_2018.merge(customers_df, on='customer_id', how='left')

#Make tables based on category
# Separate df_2018 into four DataFrames based on category
cancelled_df = df_2018[df_2018['category'] == 'cancelled']
delivered_df = df_2018[df_2018['category'] == 'delivered']
on_process_df =df_2018[df_2018['category'] == 'on process']
unavailable_df = df_2018[df_2018['category'] == 'unavailable']
unavailable_df.dropna(axis=1, how='all', inplace=True)

#make a geo df to adjust geolocation data
geo_columns = ['order_id', 'customer_id', 'month_order_purchase','order_purchase_timestamp', 'order_item_id','product_id','seller_id',
               'seller_zip_code_prefix', 'seller_city', 'seller_state', 
               'zip_code_prefix', 'customer_city', 'customer_state', 'mean_latitude', 'mean_longitude']
geo_delivered_df= delivered_df[geo_columns]
#drop missing value of geo_delivered_df
geo_delivered_df.dropna(inplace=True)



# Title and Subtitle
st.title("2018 E-commerce Sales Analysis Dashboard")
st.subheader("Explore 2018's Successful/Delivered Sales Trends and Performance Metrics")
st.markdown('<hr>', unsafe_allow_html=True)

# Sidebar with Widgets and Controls
st.sidebar.header("Filter Data")
start_date = st.sidebar.date_input("Start Date", pd.Timestamp("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp("2018-12-31"))
customers_states = customers_df['customer_state'].unique()  
selected_state = st.sidebar.selectbox("Select Customers' State", customers_states)  
selected_payment_type = st.sidebar.multiselect("Select Payment Types", ["credit card", "boleto", "debit card", "voucher"])


# Create a Streamlit app title
st.header("E-Commerce Report Overview")
st.subheader("Key Metrics")

# Calculate total sales (sum of payment_value)
total_sales = delivered_df['payment_value'].sum()
# Calculate the number of orders
total_orders = delivered_df['order_id'].nunique()
# Calculate the average order value (AOV)
average_order_value = total_sales / total_orders
# Calculate the number of unique customers
unique_customers = delivered_df['customer_unique_id'].nunique()
# Calculate the number of late deliveries
late_deliveries = delivered_df['late_delivery'].sum()
# Calculate the overall review score (average)
average_review_score = delivered_df['review_score'].mean()

# Create a two-column layout
col1, col2 = st.columns(2)

# Display metrics in the first column
with col1:
    # Widget for Total Sales
    st.metric("Total Sales", value=f"${total_sales:.2f}")
    # Widget for Total Orders
    st.metric("Total Orders", value=total_orders)
    # Widget for Unique Customers
    st.metric("Total Unique Customers", value=unique_customers)
    

# Display metrics in the second column
with col2:
    # Widget for Average Order Value
    st.metric("Average Order Value", value=f"${average_order_value:.2f}")
    # Widget for Late Deliveries
    st.metric("Late Deliveries", value=late_deliveries)
    # Widget for Average Review Score
    st.metric("Average Review Score", value=f"{average_review_score:.2f}")

st.markdown('<hr>', unsafe_allow_html=True)


st.header("A. Sales Trends")
st.subheader("1. Monthly Sales Performance by Revenue")

# Group the data by month and calculate total revenue
revenue_by_month = df_2018.groupby('month_order_purchase')['payment_value'].sum().reset_index()

# Create the trend plot using Seaborn
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x='month_order_purchase', y='payment_value', data=revenue_by_month, marker='o', color='b', ax=ax)
ax.set_title('Monthly Revenue in 2018')
ax.set_xlabel('Month')
ax.set_ylabel('Total Revenue')
ax.set_xticks(range(1, 11))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'])
ax.grid(True)

# Add labels to data points
for index, row in revenue_by_month.iterrows():
    ax.annotate(f'{row["payment_value"]:.2f}', (row["month_order_purchase"], row["payment_value"]), textcoords="offset points", xytext=(0, 10), ha='center')

# Display the plot using Streamlit
st.pyplot(fig)


# Create a Streamlit app title
st.subheader("2. Monthly Delivered Order Counts")

# Calculate the counts of unique order_id per month for the "Delivered" category
delivered_counts_per_month = delivered_df.groupby('month_order_purchase')['order_id'].nunique()

# Create the line plot using Seaborn
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=delivered_counts_per_month.index, y=delivered_counts_per_month.values, marker='o', ax=ax)
ax.set_xlabel('Month')
ax.set_ylabel('Number of Orders')
ax.set_title('Number of Delivered Orders per Month in 2018')

# Add annotations to data points
for x, y in zip(delivered_counts_per_month.index, delivered_counts_per_month.values):
    ax.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

# Display the plot using Streamlit
st.pyplot(fig)


# Create a Streamlit app title
st.subheader("3. Monthly Non-Delivered Order Counts by Category")

# Create DataFrames for each category
df_on_process = on_process_df.groupby(pd.to_datetime(on_process_df['month_order_purchase'], format='%Y-%m'))['order_id'].nunique().reset_index()
df_unavailable = unavailable_df.groupby(pd.to_datetime(unavailable_df['month_order_purchase'], format='%Y-%m'))['order_id'].nunique().reset_index()
df_cancelled = cancelled_df.groupby(pd.to_datetime(cancelled_df['month_order_purchase'], format='%Y-%m'))['order_id'].nunique().reset_index()

# Create a range of months from January to October
all_months = pd.date_range(start='2018-01-01', end='2018-10-01', freq='M')

# Merge them using 'outer' join on the 'month_order_purchase' column
merged_df = pd.merge(df_on_process, df_unavailable, left_on='month_order_purchase', right_on='month_order_purchase', how='outer')
merged_df = pd.merge(merged_df, df_cancelled, left_on='month_order_purchase', right_on='month_order_purchase', how='outer')

# Fill NaN values with 0
merged_df.fillna(0, inplace=True)

# Rename columns for clarity
merged_df.columns = ['Month', 'On Process', 'Unavailable', 'Cancelled']

# Set the "Month" column as the index
merged_df.set_index('Month', inplace=True)

# Create the line plot using Seaborn
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=merged_df, markers=True, ax=ax)
ax.set_xlabel('Month')
ax.set_ylabel('Number of Orders')
ax.set_title('Monthly Order Counts by Category in 2018')
ax.legend(['On Process', 'Unavailable', 'Cancelled'])

# Add labels to data points
for index, row in merged_df.iterrows():
    ax.annotate(f'{int(row["On Process"])}', (index, row["On Process"]), textcoords="offset points", xytext=(0, 5), ha='center')
    ax.annotate(f'{int(row["Unavailable"])}', (index, row["Unavailable"]), textcoords="offset points", xytext=(0, -15), ha='center')
    ax.annotate(f'{int(row["Cancelled"])}', (index, row["Cancelled"]), textcoords="offset points", xytext=(0, 5), ha='center')

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45)

# Display the plot using Streamlit
st.pyplot(fig)


#question 2
# Create a Streamlit app title
st.header("B. Customer and Product Trends")


# Create a DataFrame with latitude and longitude
locations = geo_delivered_df[['mean_latitude', 'mean_longitude']]

# Clean any rows with missing or invalid coordinates
locations = locations.dropna()

# Rename the columns in the DataFrame
locations = locations.rename(columns={'mean_latitude': 'LAT', 'mean_longitude': 'LON'})

# Create a Streamlit app title
st.subheader("1. Customers' Locations")

# Group the delivered_df by 'customer_state' and count the unique order IDs
customer_state_order_counts = delivered_df.groupby('customer_state')['order_id'].nunique().reset_index(name='unique_order_count')
# Sort the DataFrame by unique order count in descending order
customer_state_order_counts = customer_state_order_counts.sort_values(by='unique_order_count', ascending=False)

# Get the top 5 states
top_5_customer_state_order_counts = customer_state_order_counts.head(5)

# Create a bar chart
st.subheader("Top 5 States by Customer's Orders Count")
fig, ax = plt.subplots(figsize=(8, 5))

states = top_5_customer_state_order_counts['customer_state']
order_counts = top_5_customer_state_order_counts['unique_order_count']

ax.bar(states, order_counts)
ax.set_xlabel('State')
ax.set_ylabel("Customer's Orders Count")
ax.set_title("Top 5 States by Customer's Orders Count")

# Rotate x-axis labels for better readability
plt.xticks(rotation=0)

# Add label values to the bars
def add_label_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

bars = ax.patches
add_label_values(bars)

# Display the chart in Streamlit
st.pyplot(fig)


# Display the map using Streamlit
st.subheader("Map of Customer Locations")
st.map(locations)


# Create a Streamlit app title
st.subheader("2. Payment Trends")
st.subheader("Number of Orders by Payment Type")

# Calculate the number of orders by payment type
bypayment_df = delivered_df.groupby(by="payment_type")['order_id'].nunique().reset_index()
bypayment_df.rename(columns={"order_id": "order_count"}, inplace=True)

# Create a bar plot using Seaborn
plt.figure(figsize=(10, 5))
colors_ = ["#D3D3D3", "#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(
    x="order_count",
    y="payment_type",
    data=bypayment_df.sort_values(by="payment_type", ascending=False),
    palette=colors_
)
plt.title("Number of Orders by Payment Type", loc="center", fontsize=15)
plt.ylabel(None)
plt.xlabel(None)
plt.tick_params(axis='y', labelsize=12)

# Display the plot using Streamlit
st.pyplot(plt)


# Create a Streamlit app title
st.subheader("The Most Frequent Sequential and Installment Payment Combinations")

# Find the top 3 most frequent sequential and installment payment combinations
top_payment_combinations = delivered_df.groupby(['payment_sequential', 'payment_installments']).size().reset_index(name='count').nlargest(3, 'count')

# Create a bar chart using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='count', y=top_payment_combinations.apply(lambda x: f'Seq: {x["payment_sequential"]}, Inst: {x["payment_installments"]}', axis=1), data=top_payment_combinations, palette='Blues_d')
plt.xlabel('Frequency')
plt.ylabel('Sequential and Installment Payment Combination')
plt.title('Top 3 Most Frequent Sequential and Installment Payment Combinations')
plt.xticks(rotation=0)

# Display the plot using Streamlit
st.pyplot(plt)


#RFM
# Define a function to perform RFM analysis and display the results
def rfm_analysis(delivered_df):
    # Define the reference date as the maximum date in the dataset
    reference_date = delivered_df['order_purchase_timestamp'].max()

    # Calculate Recency (R) for each customer_unique_id
    delivered_df['order_purchase_timestamp'] = pd.to_datetime(delivered_df['order_purchase_timestamp'])
    recency_df = delivered_df.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
    recency_df['recency'] = (reference_date - recency_df['order_purchase_timestamp']).dt.days
    recency_df.drop(columns=['order_purchase_timestamp'], inplace=True)

    # Calculate Frequency (F) for each customer_unique_id
    frequency_df = delivered_df.groupby('customer_unique_id')['order_id'].nunique().reset_index()
    frequency_df.rename(columns={'order_id': 'frequency'}, inplace=True)

    # Calculate Monetary (M) for each customer_unique_id
    monetary_df = delivered_df.groupby('customer_unique_id')['payment_value'].sum().reset_index()
    monetary_df.rename(columns={'payment_value': 'monetary'}, inplace=True)

    # Merge the Recency, Frequency, and Monetary DataFrames
    rfm_df = pd.merge(recency_df, frequency_df, on='customer_unique_id', how='inner')
    rfm_df = pd.merge(rfm_df, monetary_df, on='customer_unique_id', how='inner')

    # Display the resulting RFM DataFrame
    st.write(rfm_df.head())

    # Create subplots for Recency, Frequency, and Monetary
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))
    colors = ["#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4"]

    # Plot Recency
    sns.barplot(y="recency", x="customer_unique_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5), palette=colors, ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("By Recency (days)", loc="center", fontsize=18)
    ax[0].tick_params(axis='x', labelsize=15)
    ax[0].tick_params(axis='x', rotation=80)  # Rotate x-axis labels by 45 degrees

    # Plot Frequency
    sns.barplot(y="frequency", x="customer_unique_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors, ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].set_title("By Frequency", loc="center", fontsize=18)
    ax[1].tick_params(axis='x', labelsize=15)
    ax[1].tick_params(axis='x', rotation=80)  # Rotate x-axis labels by 45 degrees

    # Plot Monetary
    sns.barplot(y="monetary", x="customer_unique_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors, ax=ax[2])
    ax[2].set_ylabel(None)
    ax[2].set_xlabel(None)
    ax[2].set_title("By Monetary", loc="center", fontsize=18)
    ax[2].tick_params(axis='x', labelsize=15)
    ax[2].tick_params(axis='x', rotation=80)  # Rotate x-axis labels by 45 degrees

    plt.suptitle("Best Customer Based on RFM Parameters (customer_unique_id)", fontsize=20)

    # Display the plots using Streamlit
    st.pyplot(plt)

# Create a Streamlit app title
st.subheader("3. Top Customers by RFM Analysis")

# Call the RFM analysis function
rfm_analysis(delivered_df)

# Define a function to perform product selling analysis and display the results
def product_selling_analysis(delivered_df):
    # Convert 'order_item_id' to numeric (integer) type
    delivered_df['order_item_id'] = delivered_df['order_item_id'].astype(int)

    # Group by product category and calculate the sum of quantities sold
    category_sales = delivered_df.groupby('product_category_name_english')['order_item_id'].sum().reset_index()

    # Sort the categories by total quantity sold (from highest to lowest)
    category_sales_sorted = category_sales.sort_values(by='order_item_id', ascending=False)

    # Get the top and least selling categories (e.g., top 5 and least 5)
    top_selling_categories = category_sales_sorted.head(5)
    least_selling_categories = category_sales_sorted.tail(5)

    # Create subplots for top and least selling categories
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the top selling categories
    sns.barplot(x='order_item_id', y='product_category_name_english', data=top_selling_categories, ax=axes[0], palette='Blues_d')
    axes[0].set_xlabel('Total Quantity Sold')
    axes[0].set_ylabel('Product Category (English)')
    axes[0].set_title('Top Selling Product Categories')
    axes[0].tick_params(axis='y', labelrotation=0)

    # Add bar value labels for the top selling categories
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_width()):,}', (p.get_width(), p.get_y() + p.get_height() / 2), ha='center', va='center')

    # Plot the least selling categories
    sns.barplot(x='order_item_id', y='product_category_name_english', data=least_selling_categories, ax=axes[1], palette='Reds_d')
    axes[1].set_xlabel('Total Quantity Sold')
    axes[1].set_ylabel('Product Category (English)')
    axes[1].set_title('Least Selling Product Categories')
    axes[1].tick_params(axis='y', labelrotation=0)

    # Add bar value labels for the least selling categories
    for p in axes[1].patches:
        axes[1].annotate(f'{int(p.get_width()):,}', (p.get_width(), p.get_y() + p.get_height() / 2), ha='center', va='center')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plots using Streamlit
    st.pyplot(plt)

# Create a Streamlit app title
st.subheader("4. Product Trends")
st.subheader("Product Selling")
# Call the product selling analysis function
product_selling_analysis(delivered_df)

# Define a function to perform product revenue analysis and display the results
def product_revenue_analysis(delivered_df):
    # Group the delivered_df by product category and calculate the total revenue for each category
    product_revenue = delivered_df.groupby('product_category_name_english')['payment_value'].sum().reset_index()

    # Sort the DataFrame by revenue in descending order to find the top-selling product category
    top_revenue_categories = product_revenue.sort_values(by='payment_value', ascending=False).head(5)

    # Sort the DataFrame by revenue in ascending order to find the least-selling product category
    least_revenue_categories = product_revenue.sort_values(by='payment_value').head(5)

    # Create subplots for top and least revenue categories
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the top revenue categories
    sns.barplot(x='payment_value', y='product_category_name_english', data=top_revenue_categories, ax=axes[0], palette='Blues_d')
    axes[0].set_xlabel('Total Revenue')
    axes[0].set_ylabel('Product Category Name')
    axes[0].set_title('Top Revenue Product Categories')
    axes[0].tick_params(axis='y', labelrotation=0)

    # Plot the least revenue categories
    sns.barplot(x='payment_value', y='product_category_name_english', data=least_revenue_categories, ax=axes[1], palette='Reds_d')
    axes[1].set_xlabel('Total Revenue')
    axes[1].set_ylabel('Product Category Name')
    axes[1].set_title('Least Revenue Product Categories')
    axes[1].tick_params(axis='y', labelrotation=0)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plots using Streamlit
    st.pyplot(plt)

# Create a Streamlit app title
st.subheader("Product Revenue")

# Call the product revenue analysis function
product_revenue_analysis(delivered_df)

# Define a function to perform seller performance analysis and display the results
def seller_performance_analysis(delivered_df):
    # Calculate average review score for each seller
    average_review_scores = delivered_df.groupby('seller_id')['review_score'].mean().reset_index()

    # Calculate total quantity sold for each seller
    total_qty_sold = delivered_df.groupby('seller_id')['order_item_id'].sum().reset_index()
    total_qty_sold.rename(columns={'order_item_id': 'qty'}, inplace=True)  # Rename the column

    # Merge the two DataFrames
    seller_performance = pd.merge(average_review_scores, total_qty_sold, on='seller_id')

    # Rank sellers based on review score and quantity sold separately
    seller_performance['review_score_rank'] = seller_performance['review_score'].rank(ascending=False)
    seller_performance['qty_rank'] = seller_performance['qty'].rank(ascending=False)

    # Calculate a combined rank as the sum of the two ranks
    seller_performance['combined_rank'] = seller_performance['review_score_rank'] + seller_performance['qty_rank']

    # Sort the DataFrame based on the combined rank
    seller_performance_sorted = seller_performance.sort_values(by='combined_rank')

    # Identify the best and worst performing sellers
    best_sellers = seller_performance_sorted.nsmallest(5, 'combined_rank')
    worst_sellers = seller_performance_sorted.nlargest(5, 'combined_rank')

    # Display the best and worst performing sellers using Streamlit
    st.subheader("Best Performing Sellers:")
    st.table(best_sellers)

    st.subheader("Weakest Performing Sellers:")
    st.table(worst_sellers)

# Create a Streamlit app title
st.header("C. Seller Performance")

# Call the seller performance analysis function
seller_performance_analysis(delivered_df)

# Define the Streamlit app title
st.subheader("Seller Rankings")

# Calculate average review score for each seller
average_review_scores = delivered_df.groupby('seller_id')['review_score'].mean().reset_index()

# Calculate total quantity sold for each seller
total_qty_sold = delivered_df.groupby('seller_id')['order_item_id'].sum().reset_index()
total_qty_sold.rename(columns={'order_item_id': 'qty'}, inplace=True)  # Rename the column

# Merge the two DataFrames
seller_performance = pd.merge(average_review_scores, total_qty_sold, on='seller_id')

# Rank sellers based on review score and quantity sold separately
seller_performance['review_score_rank'] = seller_performance['review_score'].rank(ascending=False)
seller_performance['qty_rank'] = seller_performance['qty'].rank(ascending=False)

# Calculate a combined rank as the sum of the two ranks
seller_performance['combined_rank'] = seller_performance['review_score_rank'] + seller_performance['qty_rank']

# Sort the DataFrame based on the combined rank
seller_performance_sorted = seller_performance.sort_values(by='combined_rank')

# Get the top 5 best and worst performing sellers
best_sellers = seller_performance_sorted.nsmallest(5, 'combined_rank')
worst_sellers = seller_performance_sorted.nlargest(5, 'combined_rank')

# Create subplots for best and worst performing sellers
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot the top best performing sellers by review score
sns.barplot(x='review_score', y='seller_id', data=best_sellers, ax=axes[0, 0], palette='Blues_d')
axes[0, 0].set_xlabel('Average Review Score')
axes[0, 0].set_ylabel('Seller ID')
axes[0, 0].set_title('Top Best Performing Sellers by Review Score')
axes[0, 0].tick_params(axis='y', labelrotation=0)

# Plot the top worst performing sellers by review score
sns.barplot(x='review_score', y='seller_id', data=worst_sellers, ax=axes[0, 1], palette='Reds_d')
axes[0, 1].set_xlabel('Average Review Score')
axes[0, 1].set_ylabel('Seller ID')
axes[0, 1].set_title('Top Worst Performing Sellers by Review Score')
axes[0, 1].tick_params(axis='y', labelrotation=0)

# Plot the top best performing sellers by quantity sold
sns.barplot(x='qty', y='seller_id', data=best_sellers, ax=axes[1, 0], palette='Blues_d')
axes[1, 0].set_xlabel('Quantity Sold')
axes[1, 0].set_ylabel('Seller ID')
axes[1, 0].set_title('Top Best Performing Sellers by Quantity Sold')
axes[1, 0].tick_params(axis='y', labelrotation=0)

# Plot the top worst performing sellers by quantity sold
sns.barplot(x='qty', y='seller_id', data=worst_sellers, ax=axes[1, 1], palette='Reds_d')
axes[1, 1].set_xlabel('Quantity Sold')
axes[1, 1].set_ylabel('Seller ID')
axes[1, 1].set_title('Top Weakest Performing Sellers by Quantity Sold')
axes[1, 1].tick_params(axis='y', labelrotation=0)

# Adjust spacing between subplots
plt.tight_layout()

# Display the plots in Streamlit
st.pyplot(fig)

#Sellers location
# Sellers location
seller_state_order_counts = delivered_df.groupby('seller_state')['order_id'].nunique().reset_index(name='unique_order_count')
seller_state_order_counts = seller_state_order_counts.sort_values(by='unique_order_count', ascending=False)

# Customers location
customer_state_order_counts = delivered_df.groupby('customer_state')['order_id'].nunique().reset_index(name='unique_order_count')
customer_state_order_counts = customer_state_order_counts.sort_values(by='unique_order_count', ascending=False)

# Combine seller and customer data into one DataFrame
combined_data = pd.merge(seller_state_order_counts, customer_state_order_counts, left_on='seller_state', right_on='customer_state', suffixes=('_seller', '_customer'))

# Create a bar chart
st.subheader("Seller vs Customer Orders Counts by State")
fig, ax = plt.subplots(figsize=(10, 6))

states = combined_data['seller_state']
seller_counts = combined_data['unique_order_count_seller']
customer_counts = combined_data['unique_order_count_customer']

bar_width = 0.35
index = range(len(states))

bar1 = ax.bar(index, seller_counts, bar_width, label='Sellers')
bar2 = ax.bar([i + bar_width for i in index], customer_counts, bar_width, label='Customers')

ax.set_xlabel('State')
ax.set_ylabel('Count')
ax.set_title('Seller vs Customer Counts by State')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(states, rotation=0)  
ax.legend()

# Add label values to the bars diagonally
def add_label_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    rotation=90, fontsize=7, fontweight='normal')  # Rotate and style labels

add_label_values(bar1)
add_label_values(bar2)

st.pyplot(fig)


# Display the combined data
st.write("Combined Data:")
st.write(combined_data)

st.markdown('<hr>', unsafe_allow_html=True)

# Conclusion Section
st.header("Conclusion")

# Conclusion for Question 1
# 2018 Sales Performance
st.subheader("Sales Performance")
st.write("In 2018, the sales performance exhibited some noteworthy trends:")
st.markdown("- **Revenue Fluctuation**: Total revenue and order volume showed fluctuations throughout the year. For instance, in August, both revenue and the number of orders decreased significantly, amounting to $1.24 million compared to $1.42 million in January.")
st.markdown("- **Incomplete Data**: Notably, there is no data available for delivered orders in September and October, although revenue data is available for September.")
st.markdown("- **Delivered Orders Decline**: The number of delivered orders also experienced fluctuations, with a substantial decline from 7,062 orders in January to 6,351 in August. The lowest order count occurred in June, totaling 6,099 orders.")
st.markdown("- **Canceled Orders**: Analysis of canceled orders reveals an increase in cancellations during February and August, with 70 and 84 orders, respectively. Moreover, there are records of canceled orders for September and October, totaling 15 and 4 orders.")

# Conclusion for Question 2
st.subheader("Customer and Product Trends")
st.write("1. **Customer Demographics**:")
st.write("   - The majority of customers are from Sao Paulo, SP.")
st.write("   - The most frequent customer has placed 7 orders.")
st.write("   - The highest-spending customer generated revenue of nearly $45,000.")
st.write("   - The RFM chart indicates no purchases made in the recent period.")

st.write("2. **Payment Methods**:")
st.write("   - Credit card is the most popular payment method, with installment and sequential 1, accounting for almost 30,000 orders.")

st.write("3. **Product Insights**:")
st.write("   - The most purchased product category is 'bed_bath_table' with 7,695 purchases.")
st.write("   - 'Health beauty' is the product category with the highest revenue, followed by 'bed_bath_table' and 'computer_accessories'.")
st.write("   - The least popular products are 'cds dvds musicals', 'fashion children clothes', and 'la cuisine'.")
st.write("   - However, even the third least-selling product, 'la cuisine', still contributes to revenue.")

# Conclusion for Question 3
# The Best and Weakest Performing Seller
st.subheader("Seller Performance Analysis")
st.markdown("- **Top Performers**: The top 5 performing sellers achieved remarkable success, each completing between 56 and 73 sales, all while maintaining near-perfect 4.8 to 5-star ratings.")
st.markdown("- **Struggling Sellers**: Conversely, the weakest performing sellers had minimal sales volume, with each completing just one sale, and they received low 1-star reviews.")
st.markdown("- **Seller Dominance in São Paulo (SP)**: Sellers located in São Paulo (SP) exhibit a significant dominance in sales performance compared to sellers in other states, where the seller performance is comparatively lower.")

#suggestion# Suggestion
st.header("Suggestion")
st.markdown("1. **Order Status Update**: There are instances of on-process data in the past month in the dataset, which ideally should not be there if the system keeps updating the order status regularly. To address this issue, consider implementing a more real-time order status update system to ensure data accuracy and transparency throughout the order processing.")
st.markdown("2. **Customer Feedback Analysis**: Consider providing more specific categorized review feedback, such as 'product quality issues' or 'late deliveries,' rather than free-text format. This allows for more detailed analysis of customer feedback and reviews, enabling data-driven decisions to enhance product quality and service. This can ultimately lead to improved customer satisfaction and loyalty.")
st.markdown("3. **Training and Support Programs**: Consider provide sellers with resources, workshops, and guidance on improving their sales strategies, customer engagement, and product quality. This hands-on assistance can empower sellers to enhance their performance and competitiveness in the market.")
st.markdown("4. **Market Expansion**: Explore opportunities to expand sellers and customers numbers into regions with fewer sellers and customers to tap into new markets and customer bases.")

# Footer Text
st.sidebar.markdown("---")
st.sidebar.text("© 2023 Anggi Novitasari")

# Additional Styling
# Modify the styling to set a dark background color and white font color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #333; /* Set dark background color */
        color: white; /* Set font color to white */
    }
    .stSidebar {
        background-color: #444; /* Set dark background color for the sidebar */
        color: white; /* Set font color to white in the sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
