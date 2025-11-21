import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    print("--- ULTRA FINALS MVP PREDICTION ---")

    try:
        pt = pd.read_csv("basketballPlayoffs/players_teams.csv")
        aw = pd.read_csv("basketballPlayoffs/awards_players.csv")
        # CARREGAR PREVISÕES DE EQUIPA (O NOSSO TRUNFO)
        playoffs = pd.read_csv("basketballPlayoffs/super_playoffs_simulation_s10.csv")
        teams = pd.read_csv("basketballPlayoffs/teams.csv") # Para treino histórico
        predicted_champion = playoffs.iloc[-1]['Winner']
        print(f"Info: Campeão Previsto pelo Modelo de Equipas: {predicted_champion}")
    except FileNotFoundError:
        print("Erro: Faltam ficheiros. Certifica-te que correste o 'final_season_simulation.py' primeiro.")
        return

    # --- 1. TREINO (Histórico Anos 4-9) ---
    # Lógica: Treinar apenas com jogadores das equipas que foram CAMPEÃS no passado
    # Como não temos um ficheiro fácil de "Campeões", vamos usar o próprio prémio de Finals MVP para descobrir a equipa
    
    # Apanhar todos os vencedores passados de Finals MVP
    hist_winners = aw[aw['award'] == 'WNBA Finals Most Valuable Player'][['playerID', 'year']]
    
    # Descobrir a equipa deles nesse ano (A equipa campeã)
    hist_winners = pd.merge(hist_winners, pt[['playerID', 'year', 'tmID']], on=['playerID', 'year'], how='left')
    champion_teams = hist_winners[['year', 'tmID']].rename(columns={'tmID': 'champion_tmID'})
    
    # Criar Dataset de Treino: Todos os jogadores das equipas campeãs históricas
    train = pd.merge(pt, champion_teams, left_on=['year', 'tmID'], right_on=['year', 'champion_tmID'], how='inner')
    
    # Feature Engineering (Stats da época regular como proxy para força nos playoffs)
    train['ppg'] = train['points'] / train['GP']
    train['eff_pg'] = (train['points'] + train['rebounds'] + train['assists'] + train['steals'] + train['blocks'] - train['turnovers']) / train['GP']
    train['usage_proxy'] = (train['fgAttempted'] + train['ftAttempted']) / train['GP']
    
    # Target
    target_aw = aw[aw['award'] == 'WNBA Finals Most Valuable Player'][['playerID', 'year']]
    target_aw['is_fmvp'] = 1
    train = pd.merge(train, target_aw, on=['playerID', 'year'], how='left').fillna(0)
    
    # Modelo
    # Usamos RandomForest porque lida bem com o facto de ser "o melhor entre um grupo pequeno"
    features = ['ppg', 'eff_pg', 'usage_proxy']
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    clf.fit(train[features], train['is_fmvp'])
    
    print("\n>>> Relatório de Accuracy (Treino - Identificar o MVP na equipa campeã):")
    if len(train['is_fmvp'].unique()) > 1:
        print(classification_report(train['is_fmvp'], clf.predict(train[features])))
    else:
        print("Aviso: Dados de treino insuficientes.")

    # --- 2. PREVISÃO (Day 0 Projections) ---
    # Função de projeção rápida (mesma lógica dos outros ficheiros)
    projections = []
    for pid, group in pt[pt['year'] < 10].groupby('playerID'):
        if 10 - group['year'].max() > 3: continue
        rec = group.sort_values('year').tail(3)
        w = np.arange(1, len(rec) + 1)
        
        proj = {'playerID': pid, 'tmID': group.iloc[-1]['tmID']}
        # Projetar stats
        for col in ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'fgAttempted', 'ftAttempted', 'GP']:
            proj[col] = np.average(rec[col], weights=w)
        projections.append(proj)
        
    test_base = pd.DataFrame(projections)
    test_base['ppg'] = test_base['points'] / test_base['GP']
    test_base['eff_pg'] = (test_base['points'] + test_base['rebounds'] + test_base['assists'] + test_base['steals'] + test_base['blocks'] - test_base['turnovers']) / test_base['GP']
    test_base['usage_proxy'] = (test_base['fgAttempted'] + test_base['ftAttempted']) / test_base['GP']

    # --- CENÁRIO A: CAMPEÃO PREVISTO (SAS) ---
    print(f"\n>>> CENÁRIO A: Se {predicted_champion} for Campeão (Previsão do Modelo de Equipas)")
    team_a = test_base[test_base['tmID'] == predicted_champion].copy()
    
    if not team_a.empty:
        team_a['prob'] = clf.predict_proba(team_a[features])[:, 1]
        print(team_a[['playerID', 'tmID', 'ppg', 'eff_pg', 'prob']].sort_values('prob', ascending=False).head(3))
    else:
        print(f"Não foram encontrados jogadores projetados para {predicted_champion}.")

    # --- CENÁRIO B: CAMPEÃO REAL (PHO - Diana Taurasi) ---
    # Isto serve para validares se o modelo de MVP funciona, mesmo que o de equipas tenha falhado
    real_champion = "PHO"
    print(f"\n>>> CENÁRIO B (Validação): Se {real_champion} for Campeão (Realidade)")
    team_b = test_base[test_base['tmID'] == real_champion].copy()
    
    if not team_b.empty:
        team_b['prob'] = clf.predict_proba(team_b[features])[:, 1]
        print(team_b[['playerID', 'tmID', 'ppg', 'eff_pg', 'prob']].sort_values('prob', ascending=False).head(3))
    else:
        print(f"Não foram encontrados jogadores projetados para {real_champion}.")

if __name__ == "__main__":
    main()