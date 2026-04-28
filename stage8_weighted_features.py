#!/usr/bin/env python3
"""
ЭТАП 8: УЛУЧШЕННЫЕ ПРИЗНАКИ
- Взвешенная форма (последний матч ×2)
- Over% дома/в гостях отдельно
- Мотивация (очки до зоны ЛЧ и вылета)
- Разница в отдыхе
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("🚀 ЭТАП 8: УЛУЧШЕННЫЕ ПРИЗНАКИ + БЭКТЕСТ")
print("=" * 55)

# ============================================================
# 1. ЗАГРУЗКА
# ============================================================
print("\n📥 ЗАГРУЗКА")
df = pd.read_csv('data/all_leagues.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values('Date').reset_index(drop=True)

df['odds_over'] = pd.to_numeric(df.get('B365>2.5', None), errors='coerce')
df['odds_under'] = pd.to_numeric(df.get('B365<2.5', None), errors='coerce')
df = df.dropna(subset=['odds_over', 'odds_under'])
print(f"   {len(df):,} матчей")

# ============================================================
# 2. ВСЕ ПРИЗНАКИ
# ============================================================
print("🔧 Признаки...")

# --- Турнирная таблица (позиции + мотивация) ---
standings = {}
home_pos, away_pos = [], []
home_pts_to_top4, away_pts_to_top4 = [], []
home_pts_to_releg, away_pts_to_releg = [], []

for _, row in df.iterrows():
    league = row['league']
    for team in [row['HomeTeam'], row['AwayTeam']]:
        if team not in standings:
            standings[team] = {'pts': 0, 'league': league}

    # Сортируем команды ТОЛЬКО из той же лиги
    league_teams = {t: s for t, s in standings.items() if s['league'] == league}
    sorted_teams = sorted(league_teams.items(), key=lambda x: x[1]['pts'], reverse=True)
    positions = {t: i+1 for i, (t, _) in enumerate(sorted_teams)}
    all_pts = sorted([s['pts'] for s in league_teams.values()], reverse=True)

    hp = positions.get(row['HomeTeam'], 10)
    ap = positions.get(row['AwayTeam'], 10)
    home_pos.append(hp); away_pos.append(ap)

    # Мотивация: очки до top-4 и до 17 места
    top4_pts = all_pts[3] if len(all_pts) > 3 else 0
    releg_pts = all_pts[16] if len(all_pts) > 16 else 0
    home_pts_to_top4.append(max(0, top4_pts - standings[row['HomeTeam']]['pts']))
    away_pts_to_top4.append(max(0, top4_pts - standings[row['AwayTeam']]['pts']))
    home_pts_to_releg.append(max(0, standings[row['HomeTeam']]['pts'] - releg_pts))
    away_pts_to_releg.append(max(0, standings[row['AwayTeam']]['pts'] - releg_pts))

    # Обновляем очки
    hg, ag = row['FTHG'], row['FTAG']
    if hg > ag: standings[row['HomeTeam']]['pts'] += 3
    elif hg == ag: standings[row['HomeTeam']]['pts'] += 1; standings[row['AwayTeam']]['pts'] += 1
    else: standings[row['AwayTeam']]['pts'] += 3

df['home_position'] = home_pos; df['away_position'] = away_pos
df['home_pts_to_top4'] = home_pts_to_top4; df['away_pts_to_top4'] = away_pts_to_top4
df['home_pts_to_releg'] = home_pts_to_releg; df['away_pts_to_releg'] = away_pts_to_releg

# --- ВЗВЕШЕННАЯ форма (последний матч ×2, предпоследний ×1.5) ---
team_gf, team_ga = {}, {}
home_gf_weighted, home_ga_weighted = [], []
away_gf_weighted, away_ga_weighted = [], []

for _, row in df.iterrows():
    for team, gf_list, ga_list, gf, ga in [
        (row['HomeTeam'], home_gf_weighted, home_ga_weighted, row['FTHG'], row['FTAG']),
        (row['AwayTeam'], away_gf_weighted, away_ga_weighted, row['FTAG'], row['FTHG'])
    ]:
        gf_hist = team_gf.get(team, [])
        ga_hist = team_ga.get(team, [])

        if len(gf_hist) >= 5:
            weights = [0.5, 0.7, 1.0, 1.5, 2.0]  # старый → новый
            weighted_gf = np.average(gf_hist[-5:], weights=weights)
            weighted_ga = np.average(ga_hist[-5:], weights=weights)
        elif gf_hist:
            weighted_gf = np.mean(gf_hist)
            weighted_ga = np.mean(ga_hist)
        else:
            weighted_gf = 0
            weighted_ga = 0

        gf_list.append(weighted_gf)
        ga_list.append(weighted_ga)
        gf_hist.append(gf); ga_hist.append(ga)
        team_gf[team] = gf_hist; team_ga[team] = ga_hist

df['home_goals_for_w'] = home_gf_weighted
df['home_goals_against_w'] = home_ga_weighted
df['away_goals_for_w'] = away_gf_weighted
df['away_goals_against_w'] = away_ga_weighted

# --- Голы за 5 (обычные) ---
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

# --- Over% ДОМА и В ГОСТЯХ отдельно ---
team_over_home, team_over_away = {}, {}
home_over_home_pct, away_over_away_pct = [], []

for _, row in df.iterrows():
    is_over = 1 if row['FTHG'] + row['FTAG'] > 2.5 else 0

    # Хозяева — смотрим их домашние матчи
    h_hist = team_over_home.get(row['HomeTeam'], [])
    home_over_home_pct.append(np.mean(h_hist[-10:]) if h_hist else 0.5)
    h_hist.append(is_over)
    team_over_home[row['HomeTeam']] = h_hist

    # Гости — смотрим их выездные матчи
    a_hist = team_over_away.get(row['AwayTeam'], [])
    away_over_away_pct.append(np.mean(a_hist[-10:]) if a_hist else 0.5)
    a_hist.append(is_over)
    team_over_away[row['AwayTeam']] = a_hist

df['home_over_home_pct'] = home_over_home_pct
df['away_over_away_pct'] = away_over_away_pct

# --- Общий Over% (как раньше) ---
to = {}; ho, ao = [], []
for _, row in df.iterrows():
    ov = 1 if row['FTHG'] + row['FTAG'] > 2.5 else 0
    for team, lst in [(row['HomeTeam'], ho), (row['AwayTeam'], ao)]:
        h = to.get(team, []); lst.append(np.mean(h[-20:]) if h else 0.5)
        h.append(ov); to[team] = h
df['home_over_25_pct'] = ho; df['away_over_25_pct'] = ao

# --- Лига ---
lg = {}; la = []
for _, row in df.iterrows():
    league = row['league']; t = row['FTHG'] + row['FTAG']
    if league not in lg: lg[league] = []
    la.append(np.mean(lg[league]) if lg[league] else 2.5)
    lg[league].append(t)
df['league_avg_goals'] = la

# --- H2H ---
h2h = {}; hl = []
for _, row in df.iterrows():
    key = tuple(sorted([row['HomeTeam'], row['AwayTeam']]))
    h = h2h.get(key, []); hl.append(np.mean(h[-5:]) if h else 0.5)
    h.append(1 if row['FTHG'] + row['FTAG'] > 2.5 else 0); h2h[key] = h
df['h2h_over_pct'] = hl

# --- Отдых + разница ---
last = {}; hr, ar = [], []
for _, row in df.iterrows():
    d = row['Date']
    for team, lst in [(row['HomeTeam'], hr), (row['AwayTeam'], ar)]:
        prev = last.get(team, d - pd.Timedelta(days=7))
        lst.append(min((d - prev).days, 14)); last[team] = d
df['home_rest_days'] = hr; df['away_rest_days'] = ar
df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']

# --- Целевая ---
df['target'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
df['total_goals'] = df['FTHG'] + df['FTAG']

# ============================================================
# 3. НОВЫЙ НАБОР ПРИЗНАКОВ
# ============================================================
FEATURES_V8 = [
    # Взвешенная форма (новые!)
    'home_goals_for_w', 'home_goals_against_w',
    'away_goals_for_w', 'away_goals_against_w',
    # Over% дома/в гостях (новые!)
    'home_over_home_pct', 'away_over_away_pct',
    # Мотивация (новые!)
    'home_pts_to_top4', 'away_pts_to_top4',
    'home_pts_to_releg', 'away_pts_to_releg',
    # Разница в отдыхе (новая!)
    'rest_advantage',
    # Старые проверенные
    'home_over_25_pct', 'away_position', 'home_position',
    'league_avg_goals', 'h2h_over_pct',
    'home_goals_for_5', 'away_goals_for_5',
    'home_goals_against_3', 'away_goals_against_5',
    'home_rest_days', 'away_rest_days',
]

print(f"   Признаков: {len(FEATURES_V8)}")

# ============================================================
# 4. ОБУЧЕНИЕ И БЭКТЕСТ
# ============================================================
print("\n🧠 ОБУЧЕНИЕ")
split = int(len(df) * 0.8)
train = df.iloc[:split]; test = df.iloc[split:]

model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
model.fit(train[FEATURES_V8], train['target'])

probs = model.predict_proba(test[FEATURES_V8])[:, 1]
ll = log_loss(test['target'], probs)
acc = accuracy_score(test['target'], model.predict(test[FEATURES_V8]))
print(f"   LogLoss: {ll:.4f} | Точность: {acc:.2%}")

print("\n💵 БЭКТЕСТ")
BANK_START = 10000
STAKE_PCT = 0.02
bank = BANK_START
bets = {'over': 0, 'under': 0}
wins = losses = skips = 0

for idx, row in test.iterrows():
    idx = bets['over'] + bets['under'] + wins + losses
    if idx >= len(probs): idx = len(probs) - 1
    if idx < 0: idx = 0
    prob_over = probs[idx]
    total = row['total_goals']
    stake = bank * STAKE_PCT
    bet = False

    if prob_over > 0.70 and row['odds_over'] > 0:
        bets['over'] += 1; bet = True
        bank += stake * (row['odds_over'] - 1) if total > 2.5 else -stake
        wins += 1 if total > 2.5 else 0; losses += 0 if total > 2.5 else 1
    elif prob_over < 0.30 and row['odds_under'] > 0:
        bets['under'] += 1; bet = True
        bank += stake * (row['odds_under'] - 1) if total < 2.5 else -stake
        wins += 1 if total < 2.5 else 0; losses += 0 if total < 2.5 else 1

    if not bet: skips += 1

total_bets = bets['over'] + bets['under']
roi = (bank - BANK_START) / BANK_START * 100

print(f"\n📊 РЕЗУЛЬТАТЫ:")
print(f"   Банк: {BANK_START:,} → {bank:,.0f}₽ | PnL: {bank - BANK_START:+,.0f}₽")
print(f"   ROI: {roi:+.2f}% | Проходимость: {wins/total_bets*100:.1f}%" if total_bets > 0 else "   Нет ставок")
print(f"   Ставок: {total_bets} (ТБ: {bets['over']}, ТМ: {bets['under']}) | Выиграно: {wins} | Проиграно: {losses} | Пропущено: {skips}")
print(f"   Этап 7 (ROI): -6.44% → Этап 8 (ROI): {roi:+.2f}%")

print(f"\n📊 ВАЖНОСТЬ (топ-12):")
imp = model.feature_importances_
for i, j in enumerate(np.argsort(imp)[::-1][:12]):
    tag = ' 🆕' if any(k in FEATURES_V8[j] for k in ['_w', 'home_over_home', 'away_over_away', 'pts_to', 'rest_advantage']) else ''
    print(f"   {i+1}. {FEATURES_V8[j]:<25}: {imp[j]:.4f} {tag}")

print("\n✅ ЭТАП 8 ЗАВЕРШЁН")
