#!/usr/bin/env python3
"""
ЭТАП 21: МЕТА-МОДЕЛЬ — ВЫБОР ЛУЧШЕЙ ЛИНИИ
Одна модель предсказывает OU 2.5, Team 1.5, Team 0.5
И сама выбирает самый уверенный исход с лучшим EV.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from itertools import product
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 ЭТАП 21: МЕТА-МОДЕЛЬ — ВЫБОР ЛУЧШЕЙ ЛИНИИ")
print("=" * 60)

# ============================================================
# 1. ЗАГРУЗКА И ПРИЗНАКИ
# ============================================================
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)
df['odds_over'] = pd.to_numeric(df.get('B365>2.5', None), errors='coerce')
df['odds_under'] = pd.to_numeric(df.get('B365<2.5', None), errors='coerce')
df = df.dropna(subset=['odds_over', 'odds_under'])

# Merge xG
xgd = pd.read_csv('data/sstats_epl.csv')
xgd['date'] = pd.to_datetime(xgd['date'])
df = df.merge(xgd[['date','home','homeXg','awayXg']], left_on=['Date','HomeTeam'], right_on=['date','home'], how='left')
df = df.drop(columns=['date','home'], errors='ignore')
df['has_real_xg'] = df['homeXg'].notna().astype(int)

# --- Все признаки (сокращённо) ---
st = {}; hp, ap = [], []
for _, row in df.iterrows():
    for team in [row['HomeTeam'], row['AwayTeam']]:
        if team not in st: st[team] = {'pts': 0}
    s = sorted(st.items(), key=lambda x: x[1]['pts'], reverse=True)
    p = {t: i+1 for i, (t, _) in enumerate(s)}
    hp.append(p[row['HomeTeam']]); ap.append(p[row['AwayTeam']])
    hg, ag = row['FTHG'], row['FTAG']
    if hg > ag: st[row['HomeTeam']]['pts'] += 3
    elif hg == ag: st[row['HomeTeam']]['pts'] += 1; st[row['AwayTeam']]['pts'] += 1
    else: st[row['AwayTeam']]['pts'] += 3
df['home_position'] = hp; df['away_position'] = ap

tg, ta = {}, {}
hg5, ha5, hg3, ha3 = [], [], [], []
ag5, aa5, ag3, aa3 = [], [], [], []
for _, row in df.iterrows():
    for team, g5, a5, g3, a3, gf, ga in [
        (row['HomeTeam'], hg5, ha5, hg3, ha3, row['FTHG'], row['FTAG']),
        (row['AwayTeam'], ag5, aa5, ag3, aa3, row['FTAG'], row['FTHG'])
    ]:
        gh = tg.get(team, []); ah = ta.get(team, [])
        g5.append(np.mean(gh[-5:]) if gh else 0); a5.append(np.mean(ah[-5:]) if ah else 0)
        g3.append(np.mean(gh[-3:]) if gh else 0); a3.append(np.mean(ah[-3:]) if ah else 0)
        gh.append(gf); ah.append(ga); tg[team] = gh; ta[team] = ah
df['home_goals_for_5'] = hg5; df['home_goals_against_5'] = ha5
df['home_goals_for_3'] = hg3; df['home_goals_against_3'] = ha3
df['away_goals_for_5'] = ag5; df['away_goals_against_5'] = aa5
df['away_goals_for_3'] = ag3; df['away_goals_against_3'] = aa3

to = {}; ho, ao = [], []
to_h, to_a = {}, {}; ho_h, ao_a = [], []
for _, row in df.iterrows():
    ov = 1 if row['FTHG'] + row['FTAG'] > 2.5 else 0
    for team, lst in [(row['HomeTeam'], ho), (row['AwayTeam'], ao)]:
        h = to.get(team, []); lst.append(np.mean(h[-20:]) if h else 0.5); h.append(ov); to[team] = h
    hh = to_h.get(row['HomeTeam'], []); ho_h.append(np.mean(hh[-10:]) if hh else 0.5); hh.append(ov); to_h[row['HomeTeam']] = hh
    ah = to_a.get(row['AwayTeam'], []); ao_a.append(np.mean(ah[-10:]) if ah else 0.5); ah.append(ov); to_a[row['AwayTeam']] = ah
df['home_over_25_pct'] = ho; df['away_over_25_pct'] = ao
df['home_over_home_pct'] = ho_h; df['away_over_away_pct'] = ao_a

lg = {}; la = []
for _, row in df.iterrows():
    league = row['league']; t = row['FTHG'] + row['FTAG']
    if league not in lg: lg[league] = []
    la.append(np.mean(lg[league]) if lg[league] else 2.5); lg[league].append(t)
df['league_avg_goals'] = la

h2h = {}; hl = []
for _, row in df.iterrows():
    key = tuple(sorted([row['HomeTeam'], row['AwayTeam']]))
    h = h2h.get(key, []); hl.append(np.mean(h[-5:]) if h else 0.5)
    h.append(1 if row['FTHG'] + row['FTAG'] > 2.5 else 0); h2h[key] = h
df['h2h_over_pct'] = hl

last = {}; hr, ar = [], []
for _, row in df.iterrows():
    d = row['Date']
    for team, lst in [(row['HomeTeam'], hr), (row['AwayTeam'], ar)]:
        prev = last.get(team, d - pd.Timedelta(days=7))
        lst.append(min((d - prev).days, 14)); last[team] = d
df['home_rest_days'] = hr; df['away_rest_days'] = ar

FEATURES = [
    'home_over_25_pct', 'away_position', 'home_position', 'league_avg_goals',
    'home_goals_against_3', 'away_goals_against_5', 'away_over_25_pct',
    'home_goals_for_5', 'h2h_over_pct', 'away_goals_for_3',
    'home_rest_days', 'away_goals_for_5', 'home_goals_against_5',
    'home_goals_for_3', 'away_rest_days', 'away_goals_against_3',
    'home_over_home_pct', 'away_over_away_pct'
]

# ============================================================
# 2. СОЗДАЁМ НЕСКОЛЬКО ЦЕЛЕВЫХ ПЕРЕМЕННЫХ
# ============================================================
df['target_ou25'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
df['target_h15'] = (df['FTHG'] > 1.5).astype(int)
df['target_a15'] = (df['FTAG'] > 1.5).astype(int)
df['target_h05'] = (df['FTHG'] > 0.5).astype(int)
df['target_a05'] = (df['FTAG'] > 0.5).astype(int)

# ============================================================
# 3. ОБУЧАЕМ 5 МОДЕЛЕЙ (по одной на каждый исход)
# ============================================================
print("\n🧠 ОБУЧЕНИЕ 5 МОДЕЛЕЙ")

split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

targets = {
    'OU 2.5': 'target_ou25',
    'Home 1.5': 'target_h15',
    'Away 1.5': 'target_a15',
    'Home 0.5': 'target_h05',
    'Away 0.5': 'target_a05',
}

models = {}
probs = {}

for name, target_col in targets.items():
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
    model.fit(train[FEATURES], train[target_col])
    models[name] = model
    probs[name] = model.predict_proba(test[FEATURES])[:, 1]
    over_pct = test[target_col].mean() * 100
    print(f"   {name}: проходимость {over_pct:.1f}% | LogLoss: {log_loss(test[target_col], probs[name]):.4f}")

# ============================================================
# 4. МЕТА-СТРАТЕГИЯ: ВЫБИРАЕМ ЛУЧШИЙ ИСХОД ПО EV
# ============================================================
print("\n💵 МЕТА-СТРАТЕГИЯ: ВЫБОР ЛУЧШЕГО ИСХОДА")

# Приблизительные кэфы для разных линий
def get_odds(line, is_over, xg_hint=1.5):
    """Возвращает кэф для конкретной линии"""
    odds_map = {
        ('OU 2.5', True): 1.90, ('OU 2.5', False): 1.90,
        ('Home 1.5', True): 1.70, ('Home 1.5', False): 1.70,
        ('Away 1.5', True): 1.80, ('Away 1.5', False): 1.80,
        ('Home 0.5', True): 1.25, ('Home 0.5', False): 3.50,
        ('Away 0.5', True): 1.30, ('Away 0.5', False): 3.00,
    }
    return odds_map.get((line, is_over), 1.80)

BANK = 10000
STAKE_PCT = 0.02
bank = BANK
bets = wins = 0

for i, (_, row) in enumerate(test.iterrows()):
    total = row['FTHG'] + row['FTAG']
    hg = row['FTHG']
    ag = row['FTAG']
    stake = bank * STAKE_PCT

    best_ev = -999
    best_bet = None

    # Проверяем все 5 моделей × 2 направления (Over/Under)
    for name in targets:
        p = probs[name][i]

        # Over
        odds_over = get_odds(name, True)
        ev_over = p * odds_over - 1

        # Under
        odds_under = get_odds(name, False)
        ev_under = (1-p) * odds_under - 1

        # Выбираем лучший EV для этой линии
        if ev_over > best_ev and ev_over > 0.05 and p > 0.70:
            best_ev = ev_over
            best_bet = (name, 'Over', odds_over, p, ev_over)

        if ev_under > best_ev and ev_under > 0.05 and (1-p) > 0.65:
            best_ev = ev_under
            best_bet = (name, 'Under', odds_under, 1-p, ev_under)

    # Ставим лучший исход
    if best_bet:
        line, direction, odds_val, prob_val, ev_val = best_bet
        bets += 1

        # Определяем результат
        if line == 'OU 2.5':
            success = (total > 2.5) if direction == 'Over' else (total < 2.5)
        elif line == 'Home 1.5':
            success = (hg > 1.5) if direction == 'Over' else (hg < 1.5)
        elif line == 'Away 1.5':
            success = (ag > 1.5) if direction == 'Over' else (ag < 1.5)
        elif line == 'Home 0.5':
            success = (hg > 0.5) if direction == 'Over' else (hg < 0.5)
        elif line == 'Away 0.5':
            success = (ag > 0.5) if direction == 'Over' else (ag < 0.5)

        if success:
            bank += stake * (odds_val - 1)
            wins += 1
        else:
            bank -= stake

roi = (bank - BANK) / BANK * 100
hit = wins / bets * 100 if bets > 0 else 0

print(f"\n{'='*60}")
print(f"🏆 РЕЗУЛЬТАТ МЕТА-МОДЕЛИ")
print(f"{'='*60}")
print(f"   Банк:  {BANK:,} → {bank:,.0f}₽")
print(f"   ROI:   {roi:+.2f}%")
print(f"   Ставок: {bets}")
print(f"   Hit:   {hit:.1f}%")
print(f"\n📊 СРАВНЕНИЕ:")
print(f"   Старая система (3 модели): 403-1429 ставок, ROI +88%")
print(f"   Мета-модель:               {bets} ставок, ROI {roi:+.2f}%")

print("\n✅ ЭТАП 21 ЗАВЕРШЁН")
