import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

def main():
    print("--- ULTRA DEFENSIVE PLAYER PREDICTION ---")
    pt = pd.read_csv("basketballPlayoffs/players_teams.csv")
    aw = pd.read_csv("basketballPlayoffs/awards_players.csv")

    # 1. TREINO
    train = pt[(pt['year'] >= 4) & (pt['year'] < 10)].copy()
    train['reb_pg'] = train['rebounds'] / train['GP']
    train['stl_pg'] = train['steals'] / train['GP']
    train['blk_pg'] = train['blocks'] / train['GP']
    
    dpoy_aw = aw[aw['award'] == 'Defensive Player of the Year'][['playerID', 'year']].copy()
    dpoy_aw['is_dpoy'] = 1
    train = pd.merge(train, dpoy_aw, on=['playerID', 'year'], how='left').fillna(0)
    
    train = train[train['minutes'] > 500] # Filtro consistência

    features = ['reb_pg', 'stl_pg', 'blk_pg']
    clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
    clf.fit(train[features], train['is_dpoy'])
    
    print("\n>>> Relatório de Accuracy (Treino):")
    print(classification_report(train['is_dpoy'], clf.predict(train[features])))

    # 2. PREVISÃO
    projections = []
    for pid, group in pt[pt['year'] < 10].groupby('playerID'):
        if 10 - group['year'].max() > 3: continue
        rec = group.sort_values('year').tail(3)
        w = np.arange(1, len(rec) + 1)
        
        proj = {'playerID': pid, 'tmID': group.iloc[-1]['tmID']}
        for col in ['rebounds', 'steals', 'blocks', 'GP', 'minutes']:
            proj[col] = np.average(rec[col], weights=w)
        projections.append(proj)
        
    test = pd.DataFrame(projections)
    test['reb_pg'] = test['rebounds'] / test['GP']
    test['stl_pg'] = test['steals'] / test['GP']
    test['blk_pg'] = test['blocks'] / test['GP']
    
    test = test[test['minutes'] > 500]
    
    test['prob'] = clf.predict_proba(test[features])[:, 1]
    
    print("\n>>> TOP 5 DPOY CANDIDATES (Previsão Época 10):")
    print(test[['playerID', 'reb_pg', 'stl_pg', 'blk_pg', 'prob']].sort_values('prob', ascending=False).head(5))

if __name__ == "__main__":
    main()