import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


def get_metrics(group, name):
    mae = mean_absolute_error(group['won'], group['predicted_wins'])
    r2 = r2_score(group['won'], group['predicted_wins'])
    return {'Scope': name, 'MAE': mae, 'R2': r2}

# load pred file
pred_file=pd.read_csv("basketballPlayoffs/super_teams_projected_s10.csv")

#load actual file
actual_res=pd.read_csv("basketballPlayoffs/teams.csv")
actual_res = actual_res[actual_res["year"]==10]

actual_res = actual_res[["tmID","won","rank"]]

combined = pd.merge(pred_file, actual_res, on="tmID")

metrics = [get_metrics(combined, 'Overall')]
# Overall

# Per Conference
for conf, group in combined.groupby('confID'):
    metrics.append(get_metrics(group, f'Conference {conf}'))

metrics_df = pd.DataFrame(metrics)
print("--- Model Performance Metrics ---")
print(metrics_df)

# 3. Visualization

# Set global style
sns.set(style="whitegrid")

# PLOT A: Actual vs Predicted Wins (Color-coded by Conference)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=combined,
    x='won',
    y='predicted_wins',
    hue='confID',
    style='confID',
    s=100,
    palette='viridis'
)

# Add "Perfect Prediction" diagonal line
min_limit = min(combined['won'].min(), combined['predicted_wins'].min()) - 2
max_limit = max(combined['won'].max(), combined['predicted_wins'].max()) + 2
plt.plot([min_limit, max_limit], [min_limit, max_limit], 'r--', alpha=0.5, label='Perfect Prediction')

plt.title('Actual vs Predicted Wins: Conference Comparison')
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.legend(title='Conference')
plt.tight_layout()
plt.savefig('actual_vs_predicted_conf.png')
plt.show()

# PLOT B: Ranking Accuracy (Actual Rank vs Predicted Rank)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=combined,
    x='rank',
    y='rank_conf',
    hue='confID',
    s=150,
    palette='deep',
    alpha=0.8
)

# Add diagonal line for perfect ranking
plt.plot([0, 15], [0, 15], 'k--', alpha=0.3, label='Perfect Rank')
# Invert axis because Rank 1 is "higher"/better than Rank 10
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.title('Ranking Accuracy: Predicted vs Actual Conference Rank')
plt.xlabel('Actual Rank')
plt.ylabel('Predicted Rank')
plt.legend()
plt.tight_layout()
plt.savefig('rank_accuracy_conf.png')
plt.show()
