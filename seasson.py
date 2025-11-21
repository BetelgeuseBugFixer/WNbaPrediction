import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

# ==============================================================================
# 1. HELPER: PLAYER PROJECTIONS (DAY 0)
# ==============================================================================
def create_player_projections(players_teams, target_year):
    history = players_teams[players_teams['year'] < target_year].copy()
    
    if 'efficiency' not in history.columns:
        history['efficiency'] = (history['points'] + history['rebounds'] + history['assists'] + 
                                 history['steals'] + history['blocks'] - history['turnovers'])

    projections = []
    for pid, group in history.groupby('playerID'):
        group = group.sort_values('year')
        if target_year - group['year'].max() > 3: continue
        
        recent = group.tail(3)
        weights = np.arange(1, len(recent) + 1)
        
        proj = {'playerID': pid, 'tmID': group.iloc[-1]['tmID']}
        for col in ['points', 'efficiency', 'turnovers', 'minutes']:
            proj[col] = np.average(recent[col], weights=weights)
        projections.append(proj)
        
    return pd.DataFrame(projections)

# ==============================================================================
# 2. HELPER: AWARDS HISTORY
# ==============================================================================
def get_roster_awards_history(roster_df, awards_df, target_year):
    # Filter awards won BEFORE target_year
    past_awards = awards_df[awards_df['year'] < target_year]
    
    # Count awards per player
    # We can weight them: MVP=3, Finals MVP=2, Others=1
    def calc_award_points(row):
        if 'Most Valuable Player' in row['award']: return 3
        if 'Finals' in row['award']: return 2
        if 'Defensive' in row['award']: return 2
        return 1
    
    past_awards = past_awards.copy()
    past_awards['points'] = past_awards.apply(calc_award_points, axis=1)
    
    player_award_points = past_awards.groupby('playerID')['points'].sum().reset_index().rename(columns={'points': 'award_score'})
    player_mvp_count = past_awards[past_awards['award'] == 'Most Valuable Player'].groupby('playerID').size().reset_index(name='mvp_count')
    
    # Merge with current roster
    roster_awards = pd.merge(roster_df[['playerID', 'tmID']], player_award_points, on='playerID', how='left').fillna(0)
    roster_awards = pd.merge(roster_awards, player_mvp_count, on='playerID', how='left').fillna(0)
    
    # Aggregate per team
    team_awards = roster_awards.groupby('tmID')[['award_score', 'mvp_count']].sum().add_prefix('roster_')
    return team_awards

# ==============================================================================
# 3. HELPER: PLAYER BIO (Experience & Age)
# ==============================================================================
def get_roster_bio_stats(roster_df, players_df, target_year):
    # Merge roster with players bio
    # roster_df has playerID, players_df has bioID
    merged = pd.merge(roster_df[['playerID', 'tmID']], players_df, left_on='playerID', right_on='bioID', how='inner')
    
    # Calculate Experience (Years in league)
    # Assuming 'firstseason' is reliable. If 0, we can use 'year' from players_teams min.
    # Let's use target_year - firstseason
    # If firstseason is 0 or future, handle it.
    # Alternatively, assume a rookie age if missing.
    
    # Let's use simple logic: target_year - firstseason
    # (Note: User mentioned firstseason might be 0 in data, let's check. If so, we skip or use calculated)
    # From previous turns, firstseason was mostly 0. Better to use players_teams min year.
    pass # Logic moved inside main feature prep for efficiency with existing data
    return pd.DataFrame() # Placeholder

# ==============================================================================
# 4. FEATURE ENGINEERING SUPER-ROBUST
# ==============================================================================
def prepare_super_features(players_teams, teams, coaches, players, awards, target_year):
    # --- A. Base Projections ---
    projections = create_player_projections(players_teams, target_year)
    
    # Roster Real do Ano Alvo
    roster = players_teams[players_teams['year'] == target_year][['playerID', 'tmID']].drop_duplicates(subset=['playerID'])
    
    # Calculate Experience from History (Reliable)
    first_years = players_teams.groupby('playerID')['year'].min().reset_index().rename(columns={'year': 'first_year'})
    roster = pd.merge(roster, first_years, on='playerID', how='left')
    roster['experience'] = target_year - roster['first_year']
    
    # Merge Projections
    team_proj = pd.merge(roster, projections.drop(columns=['tmID']), on='playerID', how='inner')
    
    # --- B. Aggregations (Stats) ---
    # Star Power (Top 2)
    team_proj = team_proj.sort_values(['tmID', 'efficiency'], ascending=False)
    top2 = team_proj.groupby('tmID').head(2).groupby('tmID')[['points', 'efficiency']].sum().add_prefix('star_')
    # Totals
    totals = team_proj.groupby('tmID')[['points', 'efficiency', 'turnovers']].sum().add_prefix('total_')
    # Experience Avg
    exp_avg = team_proj.groupby('tmID')['experience'].mean().reset_index().rename(columns={'experience': 'avg_experience'})
    
    features = pd.merge(totals, top2, on='tmID', how='left')
    features = pd.merge(features, exp_avg, on='tmID', how='left')
    
    # --- C. Awards History (NEW!) ---
    awards_feats = get_roster_awards_history(roster, awards, target_year)
    features = pd.merge(features, awards_feats, on='tmID', how='left').fillna(0)
    
    # --- D. Coach & Team History ---
    team_history = teams[teams['year'] == target_year][['tmID', 'confID', 'year']].copy()
    prev_year = teams[teams['year'] == target_year - 1][['tmID', 'won', 'lost', 'rank']]
    prev_year['prev_win_pct'] = prev_year['won'] / (prev_year['won'] + prev_year['lost'])
    prev_year = prev_year.rename(columns={'won': 'prev_won', 'rank': 'prev_rank'})
    
    # Coach Lifetime
    current_coaches = coaches[coaches['year'] == target_year].sort_values('stint').drop_duplicates(subset=['tmID'], keep='last')
    coach_hist = coaches[coaches['year'] < target_year].copy()
    coach_hist['games'] = coach_hist['won'] + coach_hist['lost']
    coach_stats = coach_hist.groupby('coachID')[['won', 'games']].sum().reset_index()
    coach_stats['coach_lifetime_win_pct'] = coach_stats['won'] / coach_stats['games']
    current_coaches = pd.merge(current_coaches, coach_stats[['coachID', 'coach_lifetime_win_pct']], on='coachID', how='left').fillna(0.5)
    
    features = pd.merge(features, team_history, on='tmID', how='inner')
    features = pd.merge(features, prev_year[['tmID', 'prev_won', 'prev_win_pct']], on='tmID', how='left').fillna(0.5)
    features = pd.merge(features, current_coaches[['tmID', 'coach_lifetime_win_pct']], on='tmID', how='left').fillna(0.5)
    
    return features

# ==============================================================================
# 5. EXECUTION
# ==============================================================================
def run_super_model():
    print("--- SUPER ROBUST MODEL (STATS + COACH + AWARDS + EXPERIENCE) ---")
    try:
        pt = pd.read_csv("basketballPlayoffs/players_teams.csv")
        teams = pd.read_csv("basketballPlayoffs/teams.csv")
        coaches = pd.read_csv("basketballPlayoffs/coaches.csv")
        players = pd.read_csv("basketballPlayoffs/players.csv")
        awards = pd.read_csv("basketballPlayoffs/awards_players.csv")
    except:
        print("Erro loading files.")
        return

    # Train (Years 4-9)
    print("Training...")
    train_data = []
    for y in range(4, 10):
        df = prepare_super_features(pt, teams, coaches, players, awards, y)
        target = teams[teams['year'] == y][['tmID', 'won']]
        train_data.append(pd.merge(df, target, on='tmID'))
    
    train_df = pd.concat(train_data)
    
    feats = ['total_efficiency', 'star_efficiency', 'roster_award_score', 'roster_mvp_count', 
             'avg_experience', 'prev_won', 'prev_win_pct', 'coach_lifetime_win_pct']
    
    model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(train_df[feats], train_df['won'])
    
    # Metrics
    y_pred = model.predict(train_df[feats])
    mae = mean_absolute_error(train_df['won'], y_pred)
    r2 = r2_score(train_df['won'], y_pred)
    print(f"\n>>> Training Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")
    
    # Feature Importance
    imp = pd.DataFrame({'feature': feats, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    print("\n>>> Feature Importance:")
    print(imp)
    
    # Predict Year 10
    print("\nPredicting Season 10...")
    test = prepare_super_features(pt, teams, coaches, players, awards, 10)
    test['predicted_wins'] = model.predict(test[feats])
    test['rank_conf'] = test.groupby('confID')['predicted_wins'].rank(ascending=False)
    
    # Save CSVs
    test[['tmID', 'year', 'confID', 'predicted_wins', 'rank_conf', 'prev_won']].to_csv("super_teams_projected_s10.csv", index=False)
    
    # Playoff Sim
    print("\nSimulating Playoffs...")
    east = test[test['confID'] == 'EA'].sort_values('predicted_wins', ascending=False).head(4).reset_index(drop=True)
    west = test[test['confID'] == 'WE'].sort_values('predicted_wins', ascending=False).head(4).reset_index(drop=True)
    
    playoff_res = []
    def sim(t1, t2, name):
        winner = t1 if t1['predicted_wins'] > t2['predicted_wins'] else t2
        playoff_res.append({'Round': name, 'Team1': t1['tmID'], 'Wins1': round(t1['predicted_wins'], 1), 
                            'Team2': t2['tmID'], 'Wins2': round(t2['predicted_wins'], 1), 'Winner': winner['tmID']})
        print(f"{name}: {t1['tmID']} vs {t2['tmID']} -> {winner['tmID']}")
        return winner
        
    e1 = sim(east.iloc[0], east.iloc[3], "East Semi")
    e2 = sim(east.iloc[1], east.iloc[2], "East Semi")
    w1 = sim(west.iloc[0], west.iloc[3], "West Semi")
    w2 = sim(west.iloc[1], west.iloc[2], "West Semi")
    ec = sim(e1, e2, "East Final")
    wc = sim(w1, w2, "West Final")
    champ = sim(ec, wc, "WNBA Final")
    
    pd.DataFrame(playoff_res).to_csv("super_playoffs_simulation_s10.csv", index=False)
    
    print(f"\n🏆 CHAMPION: {champ['tmID']}")
    print("\n--- EAST STANDINGS ---")
    print(test[test['confID']=='EA'][['tmID', 'predicted_wins', 'rank_conf']].sort_values('rank_conf'))
    print("\n--- WEST STANDINGS ---")
    print(test[test['confID']=='WE'][['tmID', 'predicted_wins', 'rank_conf']].sort_values('rank_conf'))

if __name__ == "__main__":
    run_super_model()