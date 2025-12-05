import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from advanced import prepare_df, split_into_train_and_val, extract_cols

# 1. Setup Data
input = [["points_dif_pyth_rolling", "new_players", "coach_playoff_wins_prev"],
         ["confID", "points_dif_pyth_rolling", "new_players", "coach_playoff_wins_prev"]]

team_df = prepare_df()
train_df, val_df = split_into_train_and_val(team_df)
for input_cols_cols in input:
    x_train = extract_cols(train_df, input_cols_cols)
    x_val = extract_cols(val_df, input_cols_cols)

    y_train = train_df["win_per"]
    y_val = val_df["win_per"]

    # 2. Get Conference Labels
    conf_labels = val_df["confID"].map({1: 'East', 0: 'West'})

    # 3. Scale
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    # 4. Train Model
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(x_train_scaled, y_train)
    prediction = model.predict(x_val_scaled)

    # 5. Metrics
    r2 = r2_score(y_val, prediction)
    mae = mean_absolute_error(y_val, prediction)
    print(f"MAE: {mae:.4f}")
    print(f"R²:  {r2:.4f}")

    # --- PLOT 1: Actual vs Predicted (With R2 Score) ---
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    plot_df = pd.DataFrame({
        'Actual Win %': y_val,
        'Predicted Win %': prediction,
        'Conference': conf_labels
    })

    sns.scatterplot(
        data=plot_df,
        x='Actual Win %',
        y='Predicted Win %',
        hue='Conference',
        style='Conference',
        palette={'East': '#2b7bba', 'West': '#e74c3c'},
        s=100,
        alpha=0.8
    )

    # Perfect prediction line
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=1.5, label='Perfect Prediction')

    # ADDED: R2 Score Annotation
    plt.text(0.05, 0.95, f'$R^2 = {r2:.3f}$\n$MAE = {mae:.3f}$',
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title('Actual vs. Predicted Win %', fontsize=14, fontweight='bold')
    plt.legend(title='Conference', loc='lower right')
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Feature Importance ---
    plt.figure(figsize=(8, 6))

    importances = pd.DataFrame({
        'Feature': input_cols_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    sns.barplot(data=importances, x='Importance', y='Feature', palette='viridis')

    plt.title('Feature Importance (Gradient Boosting)', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Importance')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()
