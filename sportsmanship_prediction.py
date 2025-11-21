import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def create_projections(players_teams, target_year):
    # Criar cópia para evitar avisos
    history = players_teams[players_teams['year'] < target_year].copy()
    projections = []
    
    # Agrupar por jogador
    for pid, group in history.groupby('playerID'):
        # Se não joga há mais de 3 anos, ignorar
        if target_year - group['year'].max() > 3: continue
        
        # CORREÇÃO 1: .copy() para resolver o SettingWithCopyWarning
        recent = group.tail(3).copy()
        weights = np.arange(1, len(recent) + 1)
        
        # Projetar Faltas e Minutos
        recent['fouls_pg'] = recent['PF'] / recent['GP']
        
        proj_fouls = np.average(recent['fouls_pg'], weights=weights)
        proj_min_total = np.average(recent['minutes'], weights=weights)
        
        projections.append({
            'playerID': pid, 
            'tmID': group.iloc[-1]['tmID'],
            'fouls_pg': proj_fouls, 
            # CORREÇÃO 2: Nome da coluna alterado para 'minutes' para bater certo com o treino
            'minutes': proj_min_total  
        })
    return pd.DataFrame(projections)

def main():
    print("--- SPORTSMANSHIP PREDICTION (VERSÃO FINAL CORRIGIDA) ---")
    try:
        pt = pd.read_csv("basketballPlayoffs/players_teams.csv")
        aw = pd.read_csv("basketballPlayoffs/awards_players.csv")
    except FileNotFoundError:
        print("Erro: Ficheiros csv não encontrados. Verifica o caminho.")
        return

    # 1. TRAIN: Dados Históricos Reais (Anos 1-9)
    # Filtrar apenas quem tem minutos suficientes
    train = pt[(pt['minutes'] > 400) & (pt['year'] < 10)].copy()
    train['fouls_pg'] = train['PF'] / train['GP']

    # Target
    sp_aw = aw[aw['award'] == 'Kim Perrot Sportsmanship Award'][['playerID', 'year']]
    sp_aw['is_sp'] = 1
    train = pd.merge(train, sp_aw, on=['playerID', 'year'], how='left').fillna(0)
    
    # Modelo
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    
    # Features (Devem ser iguais no treino e na previsão)
    features = ['fouls_pg', 'minutes']
    
    X_train = train[features]
    y_train = train['is_sp']
    
    # Verificação de segurança: O modelo precisa de pelo menos 2 classes (0 e 1)
    if len(y_train.unique()) < 2:
        print("ERRO: O conjunto de treino não tem vencedores suficientes para aprender.")
        return

    rf.fit(X_train, y_train)

    # ACCURACY REPORT
    print("\n>>> Training Accuracy Stats:")
    print(classification_report(y_train, rf.predict(X_train)))

    # 2. PREDICT: Day 0 Projection (Year 10)
    proj = create_projections(pt, 10)
    
    if not proj.empty:
        # Filtrar candidatos com minutos suficientes
        candidates = proj[proj['minutes'] > 400].copy()
        
        if not candidates.empty:
            # CORREÇÃO 3: Usar nomes de colunas corretos
            candidates['prob'] = rf.predict_proba(candidates[features])[:, 1]
            
            print("\n>>> Top 5 Candidatas Sportsmanship (Época 10 Projection):")
            print(candidates[['playerID', 'fouls_pg', 'minutes', 'prob']].sort_values('prob', ascending=False).head(5))
        else:
            print("Nenhum candidato cumpre os requisitos de minutos projetados.")
    else:
        print("Não foi possível gerar projeções.")

if __name__ == "__main__":
    main()