import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

def main():
    print("--- ULTRA MVP PREDICTION (COM DADOS DE EQUIPA PROJETADOS) ---")
    try:
        pt = pd.read_csv("basketballPlayoffs/players_teams.csv")
        aw = pd.read_csv("basketballPlayoffs/awards_players.csv")
        # CARREGAR PREVISÕES DE EQUIPA (O NOSSO TRUNFO)
        team_proj = pd.read_csv("basketballPlayoffs/super_teams_projected_s10.csv")
        teams = pd.read_csv("basketballPlayoffs/teams.csv") # Para treino histórico
    except FileNotFoundError:
        print("Erro: Faltam ficheiros. Certifica-te que correste o 'final_season_simulation.py' primeiro.")
        return

    # --- 1. PREPARAR TREINO (ANOS 4-9) ---
    # Usamos histórico real para treinar
    train = pt[(pt['year'] >= 4) & (pt['year'] < 10)].copy()
    # Feature Engineering
    train['ppg'] = train['points'] / train['GP']
    train['eff_pg'] = (train['points'] + train['rebounds'] + train['assists'] + train['steals'] + train['blocks'] - train['turnovers']) / train['GP']
    
    # Juntar vitórias REAIS para o treino
    real_wins = teams[['tmID', 'year', 'won']]
    train = pd.merge(train, real_wins, on=['tmID', 'year'])
    
    # Target
    mvp_aw = aw[aw['award'] == 'Most Valuable Player'][['playerID', 'year']]
    mvp_aw['is_mvp'] = 1
    train = pd.merge(train, mvp_aw, on=['playerID', 'year'], how='left').fillna(0)
    
    # Filtro de Elite (Reduz ruído)
    train = train[(train['GP'] > 20) & (train['ppg'] > 12)]

    # Modelo Poderoso
    features = ['ppg', 'eff_pg', 'won']
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    clf.fit(train[features], train['is_mvp'])

    # ACCURACY REPORT
    print("\n>>> Accuracy do Modelo (Treino):")
    print(classification_report(train['is_mvp'], clf.predict(train[features])))

    # --- 2. PREVISÃO ÉPOCA 10 (DAY 0) ---
    # A. Projetar Jogadores
    print("A projetar estatísticas individuais...")
    projections = []
    for pid, group in pt[pt['year'] < 10].groupby('playerID'):
        if 10 - group['year'].max() > 3: continue
        group = group.sort_values('year')
        recent = group.tail(3)
        weights = np.arange(1, len(recent) + 1)
        
        # Projeção Ponderada
        proj_pts = np.average(recent['points'] / recent['GP'], weights=weights)
        proj_eff = np.average((recent['points'] + recent['rebounds'] + recent['assists'] + recent['steals'] + recent['blocks'] - recent['turnovers']) / recent['GP'], weights=weights)
        
        projections.append({'playerID': pid, 'tmID': group.iloc[-1]['tmID'], 'ppg': proj_pts, 'eff_pg': proj_eff})
    
    test = pd.DataFrame(projections)
    
    # B. Juntar com PREVISÃO DE EQUIPA (O Segredo)
    # Aqui usamos o 'predicted_wins' do ficheiro Ultra em vez de dados passados
    test = pd.merge(test, team_proj[['tmID', 'predicted_wins']], on='tmID', how='left')
    test = test.rename(columns={'predicted_wins': 'won'}) # Renomear para bater certo com modelo
    
    # Prever
    test['mvp_prob'] = clf.predict_proba(test[features].fillna(0))[:, 1]
    
    print("\n>>> TOP 5 MVP CANDIDATES (Época 10):")
    print(test[['playerID', 'tmID', 'ppg', 'won', 'mvp_prob']].sort_values('mvp_prob', ascending=False).head(5))

if __name__ == "__main__":
    main()