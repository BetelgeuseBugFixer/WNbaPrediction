import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    print("--- ULTRA SIXTH WOMAN PREDICTION (CORRIGIDO) ---")
    try:
        pt = pd.read_csv("basketballPlayoffs/players_teams.csv")
        aw = pd.read_csv("basketballPlayoffs/awards_players.csv")
    except FileNotFoundError:
        print("Erro: Ficheiros não encontrados.")
        return

    # 1. TREINO
    # Filtro para jogadoras que são maioritariamente suplentes
    train = pt[(pt['GP'] > 2 * pt['GS']) & (pt['GP'] > 10) & (pt['year'] < 10)].copy()
    train['ppg'] = train['points'] / train['GP']
    train['minutes_pg'] = train['minutes'] / train['GP']
    
    sw_aw = aw[aw['award'] == 'Sixth Woman of the Year'][['playerID', 'year']].copy()
    sw_aw['is_sw'] = 1
    train = pd.merge(train, sw_aw, on=['playerID', 'year'], how='left').fillna(0)
    
    features = ['ppg', 'minutes_pg']
    
    # CORREÇÃO: Usar RandomForest para suportar class_weight='balanced'
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, max_depth=5)
    clf.fit(train[features], train['is_sw'])
    
    print("\n>>> Relatório de Accuracy (Treino):")
    if len(train['is_sw'].unique()) > 1:
        print(classification_report(train['is_sw'], clf.predict(train[features])))
    else:
        print("Aviso: Dados de treino desequilibrados.")

    # 2. PREVISÃO
    projections = []
    for pid, group in pt[pt['year'] < 10].groupby('playerID'):
        if 10 - group['year'].max() > 3: continue
        stats = group.mean(numeric_only=True)
        # Projetar se vai ser banco (Média de jogos > 2x Titular)
        if stats['GP'] > 2 * stats['GS']:
             projections.append({'playerID': pid, 'ppg': stats['points']/stats['GP'], 'minutes_pg': stats['minutes']/stats['GP']})
             
    test = pd.DataFrame(projections)
    if not test.empty:
        test['prob'] = clf.predict_proba(test[features])[:, 1]
        print("\n>>> TOP 5 SIXTH WOMAN CANDIDATES (Previsão Época 10):")
        print(test[['playerID', 'ppg', 'prob']].sort_values('prob', ascending=False).head(5))

if __name__ == "__main__":
    main()