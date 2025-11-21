import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. FEATURE ENGINEERING (HISTORICAL ONLY)
# ==============================================================================
def prepare_data_day_zero(coaches, teams, players_teams, awards):
    
    # --- A. Tratamento de Coaches (Histórico) ---
    # Calcular vitórias acumuladas e Tenure ATÉ ao início da época
    coaches['games'] = coaches['won'] + coaches['lost']
    coaches = coaches.sort_values(['coachID', 'year'])
    
    # Features cumulativas (Carreira até ao ano anterior)
    # 1. Calcular o que aconteceu no ano anterior (Shift)
    coaches['prev_won'] = coaches.groupby('coachID')['won'].shift(1).fillna(0)
    coaches['prev_lost'] = coaches.groupby('coachID')['lost'].shift(1).fillna(0)
    coaches['prev_games'] = coaches.groupby('coachID')['games'].shift(1).fillna(0) # CORREÇÃO AQUI
    
    # 2. Calcular os totais acumulados de carreira usando as colunas "prev_" que acabámos de criar
    coaches['career_wins'] = coaches.groupby('coachID')['prev_won'].cumsum()
    coaches['career_games'] = coaches.groupby('coachID')['prev_games'].cumsum() # CORREÇÃO AQUI
    
    # Win % de carreira (entrando na época)
    # Evitar divisão por zero
    coaches['career_win_pct'] = (coaches['career_wins'] / coaches['career_games']).fillna(0.5) 
    
    # Tenure (já tínhamos, é seguro pois sabemos há quanto tempo lá está)
    coaches['tenure'] = coaches.groupby(['tmID', 'coachID']).cumcount() + 1

    # --- B. Tratamento de Teams (Performance do Ano Anterior) ---
    teams = teams.sort_values(['tmID', 'year'])
    
    # Criar features "LAGGED" (do ano anterior)
    # O que sabemos no dia 0 do ano N? Sabemos o rank e vitórias do ano N-1.
    # Shift(1) empurra os dados do ano 9 para a linha do ano 10
    teams['prev_won'] = teams.groupby('tmID')['won'].shift(1)
    teams['prev_lost'] = teams.groupby('tmID')['lost'].shift(1)
    teams['prev_win_pct'] = teams['prev_won'] / (teams['prev_won'] + teams['prev_lost'])
    
    teams['prev_rank'] = teams.groupby('tmID')['rank'].shift(1)
    teams['prev_made_playoff'] = teams.groupby('tmID')['playoff'].shift(1).apply(lambda x: 1 if x == 'Y' else 0)
    
    # Tendência (Diferença entre N-1 e N-2) - Sabemos isto no dia 0!
    teams['trend_2yr'] = teams.groupby('tmID')['prev_win_pct'].diff().fillna(0)

    # Juntar Teams e Coaches
    # Nota: Usamos o 'year' atual para juntar, mas as features que trazemos são do passado (prev_*)
    df = pd.merge(coaches, teams[['tmID', 'year', 'prev_win_pct', 'prev_rank', 'prev_made_playoff', 'trend_2yr']], 
                  on=['tmID', 'year'], how='inner')

    # --- C. Target (O que queremos prever) ---
    # Queremos saber se o treinador que começa a época (no ficheiro coaches) MUDARÁ no ano seguinte.
    # Target = 1 se o coachID da equipa no ano N+1 for diferente.
    df = df.sort_values(['tmID', 'year'])
    df['next_year_coach'] = df.groupby('tmID')['coachID'].shift(-1)
    
    # ATENÇÃO: Se for o último ano dos dados de treino, não temos target.
    df['target_change'] = (df['coachID'] != df['next_year_coach']).astype(int)
    df['has_target'] = df.groupby('tmID')['year'].shift(-1).notna()

    return df

# ==============================================================================
# 2. EXECUÇÃO DO MODELO "DAY 0"
# ==============================================================================
# Carregar ficheiros
coaches = pd.read_csv("basketballPlayoffs/coaches.csv")
teams = pd.read_csv("basketballPlayoffs/teams.csv")
players_teams = pd.read_csv("basketballPlayoffs/players_teams.csv")
awards = pd.read_csv("basketballPlayoffs/awards_players.csv")

# Preparar Dados
df_day0 = prepare_data_day_zero(coaches, teams, players_teams, awards)

# Filtrar: Só treinadores principais (mais jogos na época) para evitar ruído de interinos
df_day0['total_games_season'] = df_day0['won'] + df_day0['lost']
df_day0 = df_day0.sort_values('total_games_season', ascending=False).drop_duplicates(subset=['tmID', 'year'])

# Separar Treino e Previsão
# Treino: Anos onde sabemos o resultado (temos o ano seguinte para comparar)
train_df = df_day0[df_day0['has_target'] == True].copy()

# Teste/Previsão (Época 10): Queremos prever o que acontece APÓS a época 10.
# As features usadas serão as estatísticas da Época 9 (que já sabemos no dia 0 da Época 10).
predict_df = df_day0[df_day0['year'] == 10].copy()

# Features V7 (Apenas Passado!)
features_day0 = ['prev_win_pct', 'prev_rank', 'prev_made_playoff', 'trend_2yr', 'tenure', 'career_win_pct']

# Modelo (Random Forest continua a ser o melhor)
rf_day0 = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight='balanced')
rf_day0.fit(train_df[features_day0].fillna(0), train_df['target_change'])

# Avaliação (Interna)
y_pred = rf_day0.predict(train_df[features_day0].fillna(0))
print("--- Performance do Modelo 'Day 0' (Treino) ---")
print("Accuracy:", accuracy_score(train_df['target_change'], y_pred))
print(classification_report(train_df['target_change'], y_pred))

# Importância das Features
imp = pd.DataFrame({'feature': features_day0, 'importance': rf_day0.feature_importances_}).sort_values('importance', ascending=False)
print("\n--- O que define o risco no Início da Época? ---")
print(imp)

# Previsões Reais para o Teste (Época 10)
predict_df['prob_change'] = rf_day0.predict_proba(predict_df[features_day0].fillna(0))[:, 1]
final_prediction = predict_df[['tmID', 'coachID', 'prev_win_pct', 'tenure', 'prob_change']].sort_values('prob_change', ascending=False)

print("\n--- Risco de Mudança (Previsão feita no Dia 0 da Época 10) ---")
print(final_prediction)