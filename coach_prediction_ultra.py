import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

def main():
    print("--- ULTRA COACH OF THE YEAR PREDICTION ---")
    try:
        coaches = pd.read_csv("basketballPlayoffs/coaches.csv")
        aw = pd.read_csv("basketballPlayoffs/awards_players.csv")
        teams = pd.read_csv("basketballPlayoffs/teams.csv")
        # Tentar carregar o melhor ficheiro de projeção disponível
        try:
            team_proj = pd.read_csv("basketballPlayoffs/super_teams_projected_s10.csv")
            print("Info: Usando 'super_teams_projected_s10.csv'.")
        except FileNotFoundError:
            try:
                team_proj = pd.read_csv("robust_teams_projected_s10.csv")
                print("Info: Usando 'robust_teams_projected_s10.csv'.")
            except FileNotFoundError:
                team_proj = pd.read_csv("teams_projected_s10.csv")
                print("Info: Usando 'teams_projected_s10.csv'.")
    except FileNotFoundError:
        print("Erro: Faltam ficheiros essenciais (coaches, teams, awards).")
        return

    # --- LIMPEZA DE DADOS ---
    # Garantir que IDs são strings limpas e anos são inteiros
    for df in [coaches, teams, team_proj]:
        if 'tmID' in df.columns: df['tmID'] = df['tmID'].astype(str).str.strip()
        if 'year' in df.columns: df['year'] = df['year'].astype(int)
    
    # --- 1. TREINO (Histórico) ---
    
    # CORREÇÃO CRÍTICA: Calcular deltas ANTES de filtrar os anos
    teams_full = teams.copy().sort_values(['tmID', 'year'])
    teams_full['prev_won'] = teams_full.groupby('tmID')['won'].shift(1)
    teams_full['delta_wins'] = teams_full['won'] - teams_full['prev_won']
    
    # Agora sim, filtrar apenas os anos de treino (4 a 9)
    # Usamos dropna() aqui para garantir que só usamos anos onde temos dados do ano anterior
    history = teams_full[(teams_full['year'] >= 4) & (teams_full['year'] < 10)].dropna(subset=['delta_wins'])

    # Preparar Coaches para o merge (remover colunas duplicadas)
    coaches_clean = coaches.drop(columns=['won', 'lost'], errors='ignore')
    
    # Merge: Histórico da Equipa + Treinador daquela época
    train = pd.merge(coaches_clean, history[['tmID', 'year', 'won', 'delta_wins']], on=['tmID', 'year'], how='inner')
    
    # Target (Prémios)
    # Nota: No csv awards, coachID está na coluna playerID
    coy_aw = aw[aw['award'] == 'Coach of the Year'][['playerID', 'year']].copy()
    coy_aw = coy_aw.rename(columns={'playerID': 'coachID'})
    coy_aw['is_coy'] = 1
    
    train = pd.merge(train, coy_aw, on=['coachID', 'year'], how='left').fillna(0)
    
    # Debug se continuar vazio
    if len(train) == 0:
        print("Erro Crítico: DataFrame vazio. Debug:")
        print("Teams (Filtrado):", len(history))
        print("Coaches:", len(coaches_clean))
        print("Merge Intersection:", pd.merge(coaches_clean, history, on=['tmID', 'year']).shape)
        return

    # Modelo
    features = ['won', 'delta_wins']
    clf = GradientBoostingClassifier(n_estimators=150, random_state=42)
    clf.fit(train[features], train['is_coy'])
    
    print("\n>>> Relatório de Accuracy (Treino):")
    # Verificar se há classes suficientes
    if len(train['is_coy'].unique()) > 1:
        print(classification_report(train['is_coy'], clf.predict(train[features])))
    else:
        print("Aviso: Apenas uma classe no target (0). O modelo não aprendeu vencedores.")

    # --- 2. PREVISÃO ÉPOCA 10 (Day 0) ---
    candidates = team_proj[team_proj['year'] == 10].copy()
    
    # Calcular Delta para a época 10 (Projetado - Real Ano 9)
    # Buscar 'prev_won' ao ficheiro de projeção (se existir) ou calcular
    if 'prev_won' not in candidates.columns:
        last_year_real = teams[teams['year'] == 9][['tmID', 'won']].rename(columns={'won': 'prev_won'})
        candidates = pd.merge(candidates, last_year_real, on='tmID', how='left')
    
    candidates['delta_wins'] = candidates['predicted_wins'] - candidates['prev_won']
    candidates = candidates.rename(columns={'predicted_wins': 'won'})
    
    # Juntar Treinador da Época 10
    current_coaches = coaches[coaches['year'] == 10][['tmID', 'coachID']].drop_duplicates('tmID')
    candidates = pd.merge(candidates, current_coaches, on='tmID', how='inner')
    
    # Prever
    candidates['prob'] = clf.predict_proba(candidates[features].fillna(0))[:, 1]
    
    print("\n>>> TOP 5 COACH OF THE YEAR (Previsão Época 10):")
    cols_show = ['coachID', 'tmID', 'won', 'delta_wins', 'prob']
    print(candidates[cols_show].sort_values('prob', ascending=False).head(5))

if __name__ == "__main__":
    main()