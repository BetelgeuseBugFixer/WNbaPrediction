import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    print("--- ULTRA MOST IMPROVED PLAYER PREDICTION (CORRIGIDO) ---")
    try:
        pt = pd.read_csv("basketballPlayoffs/players_teams.csv")
        aw = pd.read_csv("basketballPlayoffs/awards_players.csv")
    except FileNotFoundError:
        print("Erro: Ficheiros não encontrados.")
        return

    # 1. TREINO
    train = pt[pt['year'] < 10].copy()
    train.sort_values(['playerID', 'year'], inplace=True)
    train['ppg'] = train['points'] / train['GP']
    train['prev_ppg'] = train.groupby('playerID')['ppg'].shift(1)
    train['delta_ppg'] = train['ppg'] - train['prev_ppg']
    train = train.dropna(subset=['delta_ppg'])
    
    mip_aw = aw[aw['award'] == 'Most Improved Player'][['playerID', 'year']].copy()
    mip_aw['is_mip'] = 1
    train = pd.merge(train, mip_aw, on=['playerID', 'year'], how='left').fillna(0)
    
    features = ['delta_ppg', 'ppg']
    
    # CORREÇÃO: Usar RandomForest para suportar class_weight='balanced'
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, max_depth=5)
    clf.fit(train[features], train['is_mip'])

    print("\n>>> Relatório de Accuracy (Treino):")
    if len(train['is_mip'].unique()) > 1:
        print(classification_report(train['is_mip'], clf.predict(train[features])))
    else:
        print("Aviso: Dados de treino desequilibrados.")

    # 2. PREVISÃO
    projections = []
    for pid, group in pt[pt['year'] < 10].groupby('playerID'):
        if 10 - group['year'].max() > 3: continue
        # Projeção Ponderada (Futuro Provável)
        rec = group.sort_values('year').tail(3)
        w = np.arange(1, len(rec) + 1)
        proj_ppg = np.average(rec['points']/rec['GP'], weights=w)
        
        # Realidade do Ano 9 (Passado)
        last_year = group['year'].max()
        if last_year == 9:
            last_stats = group[group['year'] == 9].iloc[0]
            prev_ppg = last_stats['points'] / last_stats['GP']
            projections.append({'playerID': pid, 'delta_ppg': proj_ppg - prev_ppg, 'ppg': proj_ppg})
            
    test = pd.DataFrame(projections)
    if not test.empty:
        test['prob'] = clf.predict_proba(test[features])[:, 1]
        print("\n>>> TOP 5 MIP CANDIDATES (Previsão Época 10):")
        print(test[['playerID', 'delta_ppg', 'prob']].sort_values('prob', ascending=False).head(5))

if __name__ == "__main__":
    main()