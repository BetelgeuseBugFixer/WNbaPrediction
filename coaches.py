import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


def prepare_data_final(coaches, teams, players_teams, awards):
    coaches['total_games'] = coaches['won'] + coaches['lost']
    df = coaches.sort_values('total_games', ascending=False).drop_duplicates(subset=['tmID', 'year']).copy()
    df = df.sort_values(['tmID', 'coachID', 'year'])
    df['tenure'] = df.groupby(['tmID', 'coachID']).cumcount() + 1

    coach_awards = awards[awards['award'] == 'Coach of the Year'].copy()
    awards_set = set(zip(coach_awards['playerID'], coach_awards['year']))
    def get_award_credit(row):
        credit = 0
        if (row['coachID'], row['year']-1) in awards_set: credit += 3
        if (row['coachID'], row['year']-2) in awards_set: credit += 2
        if (row['coachID'], row['year']-3) in awards_set: credit += 1
        return credit
    df['award_credit'] = df.apply(get_award_credit, axis=1)

    teams = teams.sort_values(['tmID', 'year'])
    teams['win_pct'] = teams['won'] / (teams['won'] + teams['lost'])
    teams['win_pct_change'] = teams.groupby('tmID')['win_pct'].diff().fillna(0)
    teams['made_playoff'] = teams['playoff'].apply(lambda x: 1 if x == 'Y' else 0)
    teams['pyth_win_pct'] = (teams['o_pts']**14) / ((teams['o_pts']**14) + (teams['d_pts']**14))
    teams['wins_over_expectation'] = teams['won'] - (teams['pyth_win_pct'] * (teams['won'] + teams['lost']))

    df = pd.merge(df[['tmID', 'year', 'coachID', 'tenure', 'award_credit']],
                  teams[['tmID', 'year', 'win_pct', 'win_pct_change', 'rank', 'made_playoff', 'wins_over_expectation']],
                  on=['tmID', 'year'], how='inner')

    pt = players_teams.copy()
    pt['eff'] = (pt['points'] + pt['rebounds'] + pt['assists'] + pt['steals'] + pt['blocks']
                 - (pt['fgAttempted'] - pt['fgMade']) - (pt['ftAttempted'] - pt['ftMade']) - pt['turnovers'])
    team_talent = pt.groupby(['tmID', 'year'])['eff'].sum().reset_index().rename(columns={'eff': 'team_talent_raw'})
    df = pd.merge(df, team_talent, on=['tmID', 'year'], how='left')

    def z(s): return (s - s.mean()) / s.std()
    df['talent_z'] = df.groupby('year')['team_talent_raw'].transform(z)
    df['wins_z'] = df.groupby('year')['win_pct'].transform(z)
    df['underperformance_index'] = df['talent_z'] - df['wins_z']

    df.sort_values(['tmID', 'year'], inplace=True)
    df['target_change'] = (df['coachID'] != df.groupby('tmID')['coachID'].shift(-1)).astype(int)
    df['trainable'] = df.groupby('tmID')['year'].shift(-1).notna()
    return df


coaches = pd.read_csv("basketballPlayoffs/coaches.csv")
teams = pd.read_csv("basketballPlayoffs/teams.csv")
players_teams = pd.read_csv("basketballPlayoffs/players_teams.csv")
awards = pd.read_csv("basketballPlayoffs/awards_players.csv")

df = prepare_data_final(coaches, teams, players_teams, awards)
train = df[df['trainable'] == True].copy()
y10 = df[df['year'] == 10].copy()

feats = ['win_pct', 'made_playoff', 'rank', 'tenure', 'win_pct_change', 'underperformance_index', 'wins_over_expectation', 'award_credit']
X_train = train[feats].fillna(0)
y_train = train['target_change']
X_y10 = y10[feats].fillna(0)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_y10_sc = scaler.transform(X_y10)
old = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced', C=0.5)
old.fit(X_train_sc, y_train)
old_pred = old.predict(X_train_sc)


new = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42)
new.fit(X_train, y_train)
new_pred = new.predict(X_train)

perf = []
for name, pred in [('old (Logistic)', old_pred), ('new (RF)', new_pred)]:
    perf.append({
        'Model': name,
        'Accuracy': accuracy_score(y_train, pred),
        'Recall': recall_score(y_train, pred),
        'Precision': precision_score(y_train, pred),
        'F1 Score': f1_score(y_train, pred)
    })

print("=== COMPARAÇÃO FINAL DE PERFORMANCE ===")
print(pd.DataFrame(perf).round(3).to_string(index=False))

imp = pd.DataFrame({'feature': feats, 'new (RF) Importance': new.feature_importances_})
print("\n=== IMPORTÂNCIA DAS FEATURES (Modelo new - Random Forest) ===")
print(imp.sort_values('new (RF) Importance', ascending=False).round(3).to_string(index=False))

y10_preds = y10[['tmID', 'coachID']].copy()
y10_preds['old Prob'] = old.predict_proba(X_y10_sc)[:, 1]
y10_preds['new Prob'] = new.predict_proba(X_y10)[:, 1]
print("\n=== PREVISÕES ÉPOCA 10 (Ordenadas pelo new model) ===")
print(y10_preds.sort_values('new Prob', ascending=False).round(3).to_string(index=False))