import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter # Added for Pareto chart

# --- Configuration ---
# Define the path to your dataset
file_path = 'Amazon Sale Report.csv'
# List of return statuses for calculating returns
return_statuses = [
    'Shipped - Returned to Seller',
    'Shipped - Returning to Seller',
    'Shipped - Rejected by Buyer'
]
# Order of weekdays for plotting
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# --- Data Loading ---
# Load the dataset, trying different encodings
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='latin-1')

print(f"Data loaded successfully from '{file_path}'. Initial shape: {df.shape}")

# --- Data Cleaning and Preparation (Common Steps) ---
# Convert 'Date' to datetime objects, coercing errors
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Convert 'Amount' and 'Qty' to numeric, coercing errors
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')

# Standardize text columns used for grouping/filtering
for col in ['ship-state', 'ship-city', 'Category', 'Size', 'SKU', 'Fulfilment', 'Status', 'Courier Status', 'ship-service-level']:
    if col in df.columns:
        if df[col].dtype == 'object': # Only apply to object/string columns
            df[col] = df[col].str.upper().str.strip()

# Fill missing 'Courier Status' values with 'Unknown'
if 'Courier Status' in df.columns:
    df['Courier Status'].fillna('UNKNOWN', inplace=True)


# --- Analysis and Plotting ---

# 1. Time-based Sales Trends (Daily, Weekly, Monthly)
print("\nAnalyzing Time-based Sales Trends...")
df_time = df.dropna(subset=['Date', 'Amount']).copy()
df_time.set_index('Date', inplace=True)

# Daily Sales
daily_sales = df_time['Amount'].resample('D').sum()
plt.figure(figsize=(12, 6))
plt.plot(daily_sales.index, daily_sales.values, marker='o', linestyle='-')
plt.title('Daily Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales (Amount)')
plt.grid(True)
plt.tight_layout()
plt.savefig('daily_sales_trend.png')
plt.close() # Close the plot to free up memory

# Weekly Sales
weekly_sales = df_time['Amount'].resample('W').sum()
plt.figure(figsize=(12, 6))
plt.plot(weekly_sales.index, weekly_sales.values, marker='o', linestyle='-')
plt.title('Weekly Sales Trend')
plt.xlabel('Week')
plt.ylabel('Total Sales (Amount)')
plt.grid(True)
plt.tight_layout()
plt.savefig('weekly_sales_trend.png')
plt.close()

# Monthly Sales (Using 'ME' for month end as 'M' is deprecated)
monthly_sales = df_time['Amount'].resample('ME').sum()
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linestyle='-')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales (Amount)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_sales_trend.png')
plt.close()
print("Time-based sales trend charts saved.")


# 2. Category-wise Sales Analysis (Pie and Bar)
print("\nAnalyzing Category-wise Sales...")
df_category = df.dropna(subset=['Amount', 'Category']).copy()

category_sales = df_category.groupby('Category')['Amount'].sum().sort_values(ascending=False)

# Pie Chart (Top 5 + Other)
if len(category_sales) > 5:
    top_5 = category_sales.head(5).copy()
    other_sum = category_sales.iloc[5:].sum()
    if other_sum > 0:
        top_5['Other'] = other_sum
    plot_data_pie = top_5
else:
    plot_data_pie = category_sales

plt.figure(figsize=(10, 8))
plt.pie(plot_data_pie, labels=plot_data_pie.index, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'white'})
plt.title('Category-wise Sales Distribution', fontsize=16)
plt.axis('equal')
plt.savefig('category_sales_pie_chart.png')
plt.close()

# Horizontal Bar Chart (All categories)
plt.figure(figsize=(12, 8))
category_sales.sort_values(ascending=True).plot(kind='barh', color='skyblue')
plt.title('Category-wise Sales Performance', fontsize=16)
plt.xlabel('Total Sales (Amount)', fontsize=12)
plt.ylabel('Product Category', fontsize=12)
plt.tight_layout()
plt.savefig('category_sales_bar_chart.png')
plt.close()
print("Category-wise sales charts saved.")


# 3. Order Status Analysis (Delivered vs. Cancelled Donut)
print("\nAnalyzing Order Status...")
df_status = df.copy() # No subset needed initially, as we'll count all
total_orders = len(df_status)
delivered_count = df_status[df_status['Status'] == 'SHIPPED - DELIVERED TO BUYER'].shape[0]
cancelled_count = df_status[df_status['Status'] == 'CANCELLED'].shape[0]
other_count = total_orders - (delivered_count + cancelled_count)

status_data = {'Delivered': delivered_count, 'Cancelled': cancelled_count, 'Other Statuses': other_count}
status_series = pd.Series(status_data)

plt.figure(figsize=(10, 8))
colors = ['#2ca02c', '#d62728', '#ff7f0e']
plt.pie(status_series, labels=status_series.index, autopct='%1.1f%%', startangle=90,
        pctdistance=0.85, colors=colors, wedgeprops={'edgecolor': 'white'})
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Order Status: Delivered vs. Cancelled', fontsize=16)
plt.axis('equal')
plt.savefig('delivered_vs_cancelled_donut_chart.png')
plt.close()
print("Order status donut chart saved.")

# 4. Geographical Sales Analysis (Top States and Cities)
print("\nAnalyzing Geographical Sales...")
df_geo = df.dropna(subset=['Amount', 'ship-city', 'ship-state']).copy()

# Top 10 States
top_10_states = df_geo.groupby('ship-state')['Amount'].sum().nlargest(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_10_states.values, y=top_10_states.index, palette='viridis')
plt.title('Top 10 States by Sales Revenue', fontsize=16)
plt.xlabel('Total Sales (Amount)', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.tight_layout()
plt.savefig('top_10_states_by_sales.png')
plt.close()

# Top 10 Cities by Sales Revenue
top_10_cities_sales = df_geo.groupby('ship-city')['Amount'].sum().nlargest(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_10_cities_sales.values, y=top_10_cities_sales.index, palette='plasma')
plt.title('Top 10 Cities by Sales Revenue', fontsize=16)
plt.xlabel('Total Sales (Amount)', fontsize=12)
plt.ylabel('City', fontsize=12)
plt.tight_layout()
plt.savefig('top_10_cities_by_sales.png')
plt.close()

# Top 10 Cities by Number of Orders (Using original df before dropping NaNs)
df_cities_orders = df.dropna(subset=['ship-city']).copy()
top_10_cities_orders = df_cities_orders['ship-city'].value_counts().nlargest(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_10_cities_orders.values, y=top_10_cities_orders.index, palette='rocket')
plt.title('Top 10 Cities by Number of Orders', fontsize=16)
plt.xlabel('Total Number of Orders', fontsize=12)
plt.ylabel('City', fontsize=12)
plt.tight_layout()
plt.savefig('top_10_cities_by_orders.png')
plt.close()
print("Geographical sales charts saved.")


# 5. Fulfilment Type vs. Order Status (Stacked Bar)
print("\nAnalyzing Fulfilment Type vs. Order Status...")
df_fulfilment = df.dropna(subset=['Fulfilment', 'Status']).copy()

# Simplify Status categories
df_fulfilment['Status_Simplified'] = df_fulfilment['Status'].replace({
    'SHIPPED - DELIVERED TO BUYER': 'DELIVERED',
    'SHIPPED - RETURNED TO SELLER': 'RETURNED',
    'SHIPPED - RETURNING TO SELLER': 'RETURNED',
    'SHIPPED - PICKED UP': 'SHIPPED',
    'SHIPPED - OUT FOR DELIVERY': 'SHIPPED',
    'SHIPPED - REJECTED BY BUYER': 'RETURNED',
    'SHIPPED - LOST IN TRANSIT': 'OTHER ISSUE',
    'SHIPPED - DAMAGED': 'OTHER ISSUE',
    'PENDING - WAITING FOR PICK UP': 'PENDING'
})

fulfillment_status_counts = pd.crosstab(df_fulfilment['Fulfilment'], df_fulfilment['Status_Simplified'])

plt.figure(figsize=(12, 8))
fulfillment_status_counts.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis', ax=plt.gca())
plt.title('Order Status by Fulfilment Type', fontsize=16)
plt.xlabel('Fulfilment Type', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Order Status')
plt.tight_layout()
plt.savefig('fulfilment_comparison_chart.png')
plt.close()
print("Fulfilment comparison chart saved.")


# 6. Sales by Day of the Week (Highlighted Peak/Low)
print("\nAnalyzing Sales by Day of the Week...")
df_weekday = df.dropna(subset=['Date', 'Amount']).copy()
df_weekday['Weekday'] = df_weekday['Date'].dt.day_name()

weekday_sales = df_weekday.groupby('Weekday')['Amount'].sum().reindex(weekday_order)

peak_day = weekday_sales.idxmax()
low_day = weekday_sales.idxmin()

colors_weekday = [('g' if day == peak_day else 'r' if day == low_day else 'grey') for day in weekday_order]

plt.figure(figsize=(12, 7))
sns.barplot(x=weekday_sales.index, y=weekday_sales.values, palette=colors_weekday, order=weekday_order)
plt.title('Total Sales by Day of the Week (Peak vs. Low)', fontsize=16)
plt.xlabel('Day of the Week', fontsize=12)
plt.ylabel('Total Sales (Amount)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

peak_patch = plt.Rectangle((0,0),1,1,fc="g", edgecolor = 'none')
low_patch = plt.Rectangle((0,0),1,1,fc="r", edgecolor = 'none')
other_patch = plt.Rectangle((0,0),1,1,fc="grey", edgecolor = 'none')
plt.legend([peak_patch, low_patch, other_patch], ['Peak Sales Day', 'Low Sales Day', 'Standard Day'])

plt.savefig('sales_by_weekday_highlighted.png')
plt.close()
print(f"Sales by weekday chart saved. Peak Day: {peak_day}, Low Day: {low_day}")


# 7. Monthly Net Revenue and Returns (Dual-Axis Line)
print("\nAnalyzing Monthly Net Revenue and Returns...")
df_revenue_returns = df.dropna(subset=['Date', 'Amount', 'Status']).copy()
df_revenue_returns['Returned_Amount'] = df_revenue_returns.apply(
    lambda row: row['Amount'] if row['Status'] in [s.upper() for s in return_statuses] else 0,
    axis=1
)
df_revenue_returns.set_index('Date', inplace=True)

monthly_gross_revenue = df_revenue_returns['Amount'].resample('ME').sum()
monthly_returns_value = df_revenue_returns['Returned_Amount'].resample('ME').sum()
monthly_net_revenue = monthly_gross_revenue - monthly_returns_value

fig, ax1 = plt.subplots(figsize=(14, 8))

color = 'tab:blue'
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Net Revenue (Amount)', color=color, fontsize=12)
ax1.plot(monthly_net_revenue.index, monthly_net_revenue.values, color=color, marker='o', label='Net Revenue')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Value of Returns (Amount)', color=color, fontsize=12)
ax2.plot(monthly_returns_value.index, monthly_returns_value.values, color=color, marker='x', linestyle='--', label='Value of Returns')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Monthly Trend: Net Revenue and Returns', fontsize=16)
fig.tight_layout()
fig.autofmt_xdate(rotation=45)
plt.savefig('revenue_vs_returns_dual_axis.png')
plt.close()
print("Monthly revenue vs returns chart saved.")


# 8. Pareto Chart of Sales by SKU (Top 30)
print("\nAnalyzing Sales by SKU (Pareto Chart)...")
df_pareto = df.dropna(subset=['Amount', 'SKU']).copy()

sku_sales = df_pareto.groupby('SKU')['Amount'].sum().sort_values(ascending=False)

df_pareto_calc = pd.DataFrame(sku_sales)
df_pareto_calc.columns = ['Sales']
df_pareto_calc['Cumulative Sum'] = df_pareto_calc['Sales'].cumsum()
df_pareto_calc['Cumulative Percentage'] = df_pareto_calc['Cumulative Sum'] / df_pareto_calc['Sales'].sum() * 100

top_skus_pareto = df_pareto_calc.head(30)

fig, ax1 = plt.subplots(figsize=(16, 9))
sns.barplot(x=top_skus_pareto.index, y=top_skus_pareto['Sales'], ax=ax1, palette='summer')
ax1.set_xlabel('Product SKU', fontsize=12)
ax1.set_ylabel('Sales (Amount)', fontsize=12)
ax1.tick_params(axis='x', rotation=90)

ax2 = ax1.twinx()
ax2.plot(top_skus_pareto.index, top_skus_pareto['Cumulative Percentage'], color='red', marker='o', ms=5)
ax2.set_ylabel('Cumulative Percentage', fontsize=12, color='red')
ax2.yaxis.set_major_formatter(PercentFormatter())
ax2.tick_params(axis='y', labelcolor='red')
ax2.axhline(80, color='orange', linestyle='--', linewidth=2)
ax2.text(0, 81, '80% Mark', color='orange', va='bottom', ha='left')

plt.title('Pareto Chart of Sales by SKU (Top 30 SKUs)', fontsize=16)
plt.tight_layout()
plt.savefig('pareto_chart_sku_sales.png')
plt.close()
print("Pareto chart saved.")


# 9. Top Products with Highest Return Rate
print("\nAnalyzing Product Return Rates...")
df_return_rate = df.dropna(subset=['SKU', 'Status', 'Qty']).copy()

df_return_rate['is_shipped'] = df_return_rate['Status'].isin([
    'SHIPPED', 'SHIPPED - DELIVERED TO BUYER', 'SHIPPED - RETURNED TO SELLER',
    'SHIPPED - RETURNING TO SELLER', 'SHIPPED - PICKED UP',
    'SHIPPED - OUT FOR DELIVERY', 'SHIPPED - REJECTED BY BUYER'
])
df_return_rate['is_returned'] = df_return_rate['Status'].isin([s.upper() for s in return_statuses])

shipped_qty_sku = df_return_rate[df_return_rate['is_shipped']].groupby('SKU')['Qty'].sum()
returned_qty_sku = df_return_rate[df_return_rate['is_returned']].groupby('SKU')['Qty'].sum()

product_performance = pd.DataFrame({
    'ShippedQty': shipped_qty_sku,
    'ReturnedQty': returned_qty_sku
}).fillna(0)

product_performance = product_performance[product_performance['ShippedQty'] >= 5] # Filter out low volume SKUs
product_performance['ReturnRate'] = (product_performance['ReturnedQty'] / product_performance['ShippedQty'] * 100).fillna(0)

top_20_high_returns = product_performance.nlargest(20, 'ReturnRate')

plt.figure(figsize=(12, 10))
sns.barplot(x='ReturnRate', y=top_20_high_returns.index, data=top_20_high_returns, palette='Reds_r')
plt.title('Top 20 Products with the Highest Return Rate', fontsize=16)
plt.xlabel('Return Rate (%)', fontsize=12)
plt.ylabel('Product SKU', fontsize=12)
plt.tight_layout()
plt.savefig('top_20_high_return_rate_products.png')
plt.close()
print("Top 20 high return rate products chart saved.")


# 10. Analysis of Courier Status
print("\nAnalyzing Courier Status Breakdown...")
df_courier = df.copy() # Use df with filled Unknown status

status_counts_courier = df_courier['Courier Status'].value_counts()

plt.figure(figsize=(10, 7))
ax = sns.barplot(x=status_counts_courier.index, y=status_counts_courier.values, palette='plasma')
plt.title('Breakdown by Courier Status', fontsize=16)
plt.xlabel('Courier Status', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)

for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')

plt.tight_layout()
plt.savefig('courier_status_breakdown.png')
plt.close()
print("Courier status breakdown chart saved.")


# 11. Sales by Shipping Service Level
print("\nAnalyzing Sales by Shipping Service Level...")
df_service_level = df.dropna(subset=['Amount', 'ship-service-level']).copy()

service_level_sales = df_service_level.groupby('ship-service-level')['Amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 7))
ax = sns.barplot(x=service_level_sales.index, y=service_level_sales.values, palette='coolwarm')
plt.title('Total Sales by Shipping Service Level', fontsize=16)
plt.xlabel('Shipping Service Level', fontsize=12)
plt.ylabel('Total Sales (Amount)', fontsize=12)
plt.xticks(rotation=0)

for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')

plt.tight_layout()
plt.savefig('sales_by_service_level.png')
plt.close()
print("Sales by shipping service level chart saved.")


# 12. Sales by Product Size
print("\nAnalyzing Sales by Product Size...")
df_size = df.dropna(subset=['Amount', 'Size']).copy()

top_sizes = df_size.groupby('Size')['Amount'].sum().nlargest(10)

plt.figure(figsize=(12, 7))
sns.barplot(x=top_sizes.index, y=top_sizes.values, palette='crest')
plt.title('Top 10 Product Sizes by Sales Revenue', fontsize=16)
plt.xlabel('Product Size', fontsize=12)
plt.ylabel('Total Sales (Amount)', fontsize=12)
plt.tight_layout()
plt.savefig('sales_by_product_size.png')
plt.close()
print("Sales by product size chart saved.")

# 13. Monthly Average Order Value (AOV) Trend
print("\nAnalyzing Monthly AOV Trend...")
df_aov = df.dropna(subset=['Date', 'Amount', 'Order ID']).copy()
df_aov.set_index('Date', inplace=True)

monthly_revenue_aov = df_aov['Amount'].resample('ME').sum()
monthly_orders_aov = df_aov['Order ID'].resample('ME').nunique()
monthly_aov = monthly_revenue_aov / monthly_orders_aov
monthly_aov.fillna(0, inplace=True) # Fill NaN in case of months with no orders

plt.figure(figsize=(14, 7))
sns.lineplot(x=monthly_aov.index, y=monthly_aov.values, marker='o', linestyle='-')
plt.title('Monthly Average Order Value (AOV) Trend', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Order Value (AOV)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('monthly_aov_trend.png')
plt.close()
print("Monthly AOV trend chart saved.")


# 14. Monthly Trend of Cancelled and Returned Orders
print("\nAnalyzing Monthly Trend of Cancelled and Returned Orders...")
df_cancel_return_trend = df.dropna(subset=['Date', 'Status']).copy()
df_cancel_return_trend.set_index('Date', inplace=True)

df_cancel_return_trend['is_cancelled'] = df_cancel_return_trend['Status'] == 'CANCELLED'
df_cancel_return_trend['is_returned'] = df_cancel_return_trend['Status'].isin([s.upper() for s in return_statuses])

monthly_cancellations_count = df_cancel_return_trend['is_cancelled'].resample('ME').sum()
monthly_returns_count = df_cancel_return_trend['is_returned'].resample('ME').sum()

fig, ax1 = plt.subplots(figsize=(14, 8))

color = 'tab:red'
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Number of Cancelled Orders', color=color, fontsize=12)
ax1.plot(monthly_cancellations_count.index, monthly_cancellations_count.values, color=color, marker='o', label='Cancelled Orders')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Number of Returned Orders', color=color, fontsize=12)
ax2.plot(monthly_returns_count.index, monthly_returns_count.values, color=color, marker='x', linestyle='--', label='Returned Orders')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Monthly Trend of Cancelled and Returned Orders', fontsize=16)
fig.tight_layout()
fig.autofmt_xdate(rotation=45)
plt.savefig('cancelled_vs_returned_trends.png')
plt.close()
print("Monthly cancelled and returned orders trend chart saved.")

print("\nAll analyses complete and charts saved.")