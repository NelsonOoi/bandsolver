import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("bandgap_dataset.csv")

# Example visualization: 3D scatter plot of eps_rod, r_over_a vs bandgap
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['eps_rod'], df['r_over_a'], df['bandgap'], c=df['bandgap'], cmap='viridis', s=60)
ax.set_xlabel('eps_rod')
ax.set_ylabel('r_over_a')
ax.set_zlabel('bandgap')
ax.set_title('Bandgap vs eps_rod and r_over_a')

plt.tight_layout()
plt.show()
