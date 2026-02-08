import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset.csv")

# Convert items into list
transactions = data['Items'].apply(lambda x: x.split(","))

# Transaction Encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# Apply Apriori Algorithm
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules[['antecedents','consequents','support','confidence','lift']])

# Visualization
frequent_itemsets['support'].plot(kind='bar')
plt.title("Frequent Itemsets Support")
plt.ylabel("Support")
plt.xlabel("Itemsets")
plt.show()
