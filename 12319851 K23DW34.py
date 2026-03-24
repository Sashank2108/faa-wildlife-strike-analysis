

import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Load the data
df = pd.read_excel("faa_data_subset.xlsx", sheet_name="FAA Wildlife Strikes")

# -------------------- Data Cleaning -------------------- #
# Convert date and time
df['Collision Date and Time'] = pd.to_datetime(df['Collision Date and Time'], errors='coerce')
df['Year'] = df['Collision Date and Time'].dt.year

# Drop unnecessary or mostly empty columns
df_clean = df.drop(columns=[
    'Record ID', 'Country', 'Origin State Code', 'Wildlife: Species ID'
], errors='ignore')

# Handle missing values
df_clean = df_clean.dropna(subset=['Wildlife: Species', 'When: Phase of flight'])

# -------------------- Basic EDA -------------------- #
print("Top 5 states with most strikes:")
print(df_clean['Origin State'].value_counts().head())

print("\nTop 5 wildlife species involved:")
print(df_clean['Wildlife: Species'].value_counts().head())

print("\nMost affected flight phases:")
print(df_clean['When: Phase of flight'].value_counts())

print("\nTotal damage cost:")
print("${:,.2f}".format(df_clean['Cost: Total $'].sum()))
print()

print("Description of the dataset:")
print(df.describe())
# -------------------- Data Visualization -------------------- #
sns.set(style="whitegrid")

# Top 10 States
plt.figure(figsize=(10, 6))
sns.countplot(data=df_clean, y='Origin State',
              order=df_clean['Origin State'].value_counts().head(10).index)
plt.title("Top 10 States by Wildlife Strike Count")
plt.xlabel("Number of Strikes")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# Top 10 Species
plt.figure(figsize=(10, 6))
sns.countplot(data=df_clean, y='Wildlife: Species',
              order=df_clean['Wildlife: Species'].value_counts().head(10).index)
plt.title("Top 10 Wildlife Species Involved in Strikes")
plt.xlabel("Number of Strikes")
plt.ylabel("Species")
plt.tight_layout()
plt.show()

# Cost by Flight Phase
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean, x='When: Phase of flight', y='Cost: Total $')
plt.title("Damage Cost by Flight Phase")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Barplot of Damage Types
plt.figure(figsize=(10, 6))
sns.countplot(data=df_clean, x='Effect: Indicated Damage', palette="Set2",
              order=df_clean['Effect: Indicated Damage'].value_counts().index)
plt.title("Distribution of Indicated Damage")
plt.xlabel("Damage Type")
plt.ylabel("Number of Strikes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Aircraft types with most strikes
plt.figure(figsize=(10, 6))
sns.countplot(data=df_clean, y='Aircraft: Type',
              order=df_clean['Aircraft: Type'].value_counts().head(10).index,
              palette="Pastel1")
plt.title("Top 10 Aircraft Types Involved in Wildlife Strikes")
plt.xlabel("Number of Strikes")
plt.ylabel("Aircraft Type")
plt.tight_layout()
plt.show()

# Histogram of Total Cost (filtered to avoid skew from outliers)
plt.figure(figsize=(10, 6))
filtered_costs = df_clean[df_clean['Cost: Total $'] < 100000]['Cost: Total $']
sns.histplot(filtered_costs, bins=30, kde=True, color='coral')
plt.title("Distribution of Total Damage Costs (Under $100,000)")
plt.xlabel("Cost ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Add Month column
df_clean['Month'] = df_clean['Collision Date and Time'].dt.month

# Plot strikes per month
plt.figure(figsize=(10, 6))
sns.countplot(x='Month', data=df_clean, palette="cool")
plt.title("Monthly Trend of Wildlife Strikes")
plt.xlabel("Month")
plt.ylabel("Number of Strikes")
plt.tight_layout()
plt.show()

# Top 5 species by total cost of damage
top_species_cost = df.groupby('Wildlife: Species')['Cost: Total $'].sum().sort_values(ascending=False).head(5)

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_species_cost, labels=top_species_cost.index, autopct='%1.1f%%', startangle=140)
plt.title("Proportion of Total Wildlife Strike Costs by Top 5 Species")
plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
plt.tight_layout()
plt.show()

# -------------------- Yearly Trends -------------------- #
yearly_counts = df_clean['Year'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o')
plt.title("Wildlife Strikes Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Strikes")
plt.tight_layout()
plt.show()

# -------------------- Advanced Analysis -------------------- #
# Most costly species (average cost per strike)
costly_species = df_clean.groupby('Wildlife: Species')['Cost: Total $'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=costly_species.values, y=costly_species.index, palette="flare")
plt.title("Top 10 Most Costly Species (Avg Cost per Strike)")
plt.xlabel("Average Cost ($)")
plt.tight_layout()
plt.show()

# Heatmap of numerical features
# Convert datetime and extract features
df['Collision Date and Time'] = pd.to_datetime(df['Collision Date and Time'], errors='coerce')
df['Year'] = df['Collision Date and Time'].dt.year
df['Month'] = df['Collision Date and Time'].dt.month
df['Hour'] = df['Collision Date and Time'].dt.hour

# Prepare safe encoding for selected categorical columns
for col in ['Effect: Indicated Damage', 'When: Phase of flight', 'Aircraft: Type']:
    if col in df.columns:
        df[col] = df[col].astype('category').cat.codes

# Choose numeric and encoded columns that exist
possible_columns = [
    'Year', 'Month', 'Hour', 'Cost: Total $',
    'Feet above ground', 'Miles from airport',
    'Aircraft: Number of engines',
    'Effect: Indicated Damage', 'When: Phase of flight', 'Aircraft: Type'
]

heatmap_df = df[[col for col in possible_columns if col in df.columns]].dropna()

# Generate correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Enhanced Correlation Heatmap of FAA Wildlife Data")
plt.tight_layout()
plt.show()
