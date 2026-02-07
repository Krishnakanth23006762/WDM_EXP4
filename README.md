### EX4 Implementation of Cluster and Visitor Segmentation for Navigation patterns
### DATE: 07-02-2026
### AIM: To implement Cluster and Visitor Segmentation for Navigation patterns in Python.
### Description:
<div align= "justify">Cluster visitor segmentation refers to the process of grouping or categorizing visitors to a website, 
  application, or physical location into distinct clusters or segments based on various characteristics or behaviors they exhibit. 
  This segmentation allows businesses or organizations to better understand their audience and tailor their strategies, marketing efforts, 
  or services to meet the specific needs and preferences of each cluster.</div>
  
### Procedure:
1) Read the CSV file: Use pd.read_csv to load the CSV file into a pandas DataFrame.
2) Define Age Groups by creating a dictionary containing age group conditions using Boolean conditions.
3) Segment Visitors by iterating through the dictionary and filter the visitors into respective age groups.
4) Visualize the result using matplotlib.

### Program 1:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("clustervisitor (1).csv")

# Define age groups
cluster = {
    "young": (df['Age'] <= 30),
    "middle": (df['Age'] > 30) & (df['Age'] <= 50),
    "old": (df['Age'] > 50)
}

# Initialize count CLEAN
count = []

for group, condition in cluster.items():
    visitors = df[condition]
    visitor_count = len(visitors)
    count.append(visitor_count)

    print(f"Visitors in {group} age group:")
    print(visitors)
    print(f"Total visitors: {visitor_count}")
    print("-" * 30)

# Plot
labels = list(cluster.keys())

plt.figure(figsize=(8, 6))
plt.bar(labels, count, color='skyblue')
plt.xlabel('Age Groups')
plt.ylabel('Number of Visitors')
plt.title('Visitor Distribution Across Age Groups')
plt.show()


```
### Output:

<img width="386" height="681" alt="image" src="https://github.com/user-attachments/assets/205b1fad-8f7a-4c0a-af64-49127cacf129" />

<img width="712" height="523" alt="image" src="https://github.com/user-attachments/assets/f213ab65-c24f-48d7-a656-40728602becd" />



### Program 2:
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("clustervisitor (1).csv")

# Select features
X = df[['Age', 'Income']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Display dataframe
print(df, "\n")

# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['Income'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Income (in thousands)')
plt.title('K-Means Clustering of Visitors')
plt.colorbar(label='Cluster')
plt.show()

```
### Output:

<img width="451" height="509" alt="image" src="https://github.com/user-attachments/assets/ef92f73b-24b6-4c47-97d4-24cc7c646626" />

<img width="691" height="520" alt="image" src="https://github.com/user-attachments/assets/4fb12c9a-1e99-4e7f-a2f5-238dffed1121" />




### Result:

Thus, cluster and Visitor Segmentation for Navigation patterns have been implemented successfully.
