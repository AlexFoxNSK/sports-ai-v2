#!/usr/bin/env python3
"""
ЭТАП 1: БАЗОВАЯ МОДЕЛЬ XGBoost ДЛЯ OVER/UNDER 2.5

Что делает скрипт:
1. Загружает исторические данные 9 лиг (2022-2026)
2. Создаёт признаки, которые могут влиять на тотал голов
3. Обучает XGBoost предсказывать Over/Under 2.5
4. Проверяет качество на тестовой выборке
5. Показывает какие признаки самые важные

Все признаки считаются ТОЛЬКО на основе данных ДО текущего матча.
Это исключает утечку информации из будущего.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ФУНКЦИЯ 1: Загрузка данных
# ============================================================
def load_data(filepath):
    """
    Читает CSV-файл с историческими матчами.

    ЧТО ДЕЛАЕТ:
    - Загружает all_leagues.csv
    - Преобразует даты в правильный формат
    - Создаёт целевую переменную (Over/Under 2.5)

    ПАРАМЕТРЫ:
    - filepath: путь к CSV-файлу

    ВОЗВРАЩАЕТ:
    - DataFrame с отсортированными по дате матчами

    ПОЧЕМУ ВАЖНО:
    Без правильной сортировки по дате мы не сможем
    делать честные предсказания (будущее попадёт в обучение).
    """
    print("📥 ЗАГРУЗКА ДАННЫХ")

    # Читаем CSV
    # low_memory=False — pandas не пытается угадать типы колонок
    # это ускоряет загрузку больших файлов
    df = pd.read_csv(filepath, low_memory=False)

    # Преобразуем дату из текста "15/08/2022" в специальный формат datetime
    # dayfirst=True — потому что в UK-формате день идёт первым
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Удаляем строки где нет результата (пустые матчи)
    df = df.dropna(subset=['FTHG', 'FTAG'])

    # Сортируем по дате ОБЯЗАТЕЛЬНО
    # Это гарантирует что прошлое не перемешается с будущим
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"   Загружено: {len(df):,} матчей")
    print(f"   Лиг: {df['league'].nunique()}")
    print(f"   Диапазон дат: {df['Date'].min().date()} → {df['Date'].max().date()}")

    return df


# ============================================================
# ФУНКЦИЯ 2: Создание признаков
# ============================================================
def create_features(df):
    """
    Создаёт признаки, которые модель будет использовать для предсказания.

    ЧТО ДЕЛАЕТ:
    Для КАЖДОГО матча считает статистику команд на основе
    ТОЛЬКО предыдущих матчей (без утечки будущего).

    КАКИЕ ПРИЗНАКИ СОЗДАЮТСЯ:
    1. Турнирная таблица (позиция команды ДО матча)
    2. Средние голы команды за последние 5 матчей
    3. Средние пропущенные за последние 5 матчей
    4. Процент матчей Over 2.5 у команды в этом сезоне
    5. Процент матчей Over 2.5 в личных встречах

    ПОЧЕМУ ЭТО ВАЖНО:
    Модель не может "видеть" будущее. Все признаки считаются
    строго <= дата текущего матча.
    """
    print("🔧 СОЗДАНИЕ ПРИЗНАКОВ")
    df = df.copy()

    # --- 2.1: ТУРНИРНАЯ ТАБЛИЦА (rolling positions) ---
    # Создаём словарь для хранения очков, голов, пропущенных
    # для каждой команды. Обновляется ПОСЛЕ каждого матча.
    standings = {}

    # Списки для хранения позиций
    home_positions = []
    away_positions = []

    # Проходим по каждому матчу в хронологическом порядке
    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Если команду видим впервые — создаём запись
        for team in [home_team, away_team]:
            if team not in standings:
                standings[team] = {'points': 0, 'goals_for': 0, 'goals_against': 0, 'matches': 0}

        # Считаем ТЕКУЩУЮ позицию (до обновления очков за этот матч)
        # Сортируем команды по очкам
        sorted_teams = sorted(standings.items(), key=lambda x: x[1]['points'], reverse=True)
        # Создаём маппинг: название → позиция
        positions = {team: i+1 for i, (team, _) in enumerate(sorted_teams)}

        # Записываем позиции ДО этого матча
        home_positions.append(positions[home_team])
        away_positions.append(positions[away_team])

        # Обновляем статистику ПОСЛЕ матча (для следующих игр)
        home_goals = row['FTHG']
        away_goals = row['FTAG']

        # Очки: победа = 3, ничья = 1, поражение = 0
        if home_goals > away_goals:
            standings[home_team]['points'] += 3
        elif home_goals == away_goals:
            standings[home_team]['points'] += 1
            standings[away_team]['points'] += 1
        else:
            standings[away_team]['points'] += 3

        # Обновляем голы
        standings[home_team]['goals_for'] += home_goals
        standings[home_team]['goals_against'] += away_goals
        standings[away_team]['goals_for'] += away_goals
        standings[away_team]['goals_against'] += home_goals

    # Добавляем позиции в датафрейм
    df['home_position'] = home_positions
    df['away_position'] = away_positions

    print("   ✅ Турнирная таблица рассчитана")


    # --- 2.2: ГОЛЫ ЗА ПОСЛЕДНИЕ 5 МАТЧЕЙ ---
    # Храним историю голов для каждой команды
    team_goals_for = {}   # сколько забили
    team_goals_against = {}  # сколько пропустили

    home_goals_for_5 = []
    home_goals_against_5 = []
    away_goals_for_5 = []
    away_goals_against_5 = []

    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Для каждой команды: берём последние 5 матчей из истории
        for team, is_home, goals_for_list, goals_against_list, gf, ga in [
            (home_team, True, home_goals_for_5, home_goals_against_5, row['FTHG'], row['FTAG']),
            (away_team, False, away_goals_for_5, away_goals_against_5, row['FTAG'], row['FTHG'])
        ]:
            # Получаем историю команды (или пустой список если новая команда)
            gf_history = team_goals_for.get(team, [])
            ga_history = team_goals_against.get(team, [])

            # Считаем среднее за последние 5 матчей
            avg_gf = np.mean(gf_history[-5:]) if len(gf_history) >= 5 else np.mean(gf_history) if gf_history else 0
            avg_ga = np.mean(ga_history[-5:]) if len(ga_history) >= 5 else np.mean(ga_history) if ga_history else 0

            goals_for_list.append(avg_gf)
            goals_against_list.append(avg_ga)

            # Добавляем текущий матч в историю (для БУДУЩИХ предсказаний)
            gf_history.append(gf)
            ga_history.append(ga)
            team_goals_for[team] = gf_history
            team_goals_against[team] = ga_history

    df['home_goals_for_5'] = home_goals_for_5
    df['home_goals_against_5'] = home_goals_against_5
    df['away_goals_for_5'] = away_goals_for_5
    df['away_goals_against_5'] = away_goals_against_5

    print("   ✅ Голы за 5 матчей рассчитаны")


    # --- 2.3: ПРОЦЕНТ OVER 2.5 У КОМАНДЫ ---
    # Сколько матчей команды закончились с тоталом > 2.5
    team_over_history = {}

    home_over_pct = []
    away_over_pct = []

    for _, row in df.iterrows():
        total_goals = row['FTHG'] + row['FTAG']
        is_over = 1 if total_goals > 2.5 else 0

        for team, pct_list in [(row['HomeTeam'], home_over_pct), (row['AwayTeam'], away_over_pct)]:
            history = team_over_history.get(team, [])
            pct = np.mean(history[-20:]) if len(history) >= 20 else np.mean(history) if history else 0.5
            pct_list.append(pct)

            history.append(is_over)
            team_over_history[team] = history

    df['home_over_25_pct'] = home_over_pct
    df['away_over_25_pct'] = away_over_pct

    print("   ✅ Процент Over 2.5 рассчитан")


    # --- 2.4: СРЕДНИЙ ТОТАЛ В ЛИГЕ ---
    # Считаем среднее голов в каждой лиге на основе ВСЕХ предыдущих матчей
    league_goals = {}

    league_avg = []
    for _, row in df.iterrows():
        league = row['league']
        total = row['FTHG'] + row['FTAG']

        if league not in league_goals:
            league_goals[league] = []

        # Среднее по всем предыдущим матчам лиги
        avg = np.mean(league_goals[league]) if league_goals[league] else 2.5
        league_avg.append(avg)

        league_goals[league].append(total)

    df['league_avg_goals'] = league_avg

    print("   ✅ Средний тотал по лиге рассчитан")


    # --- 2.5: ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ---
    # То что мы предсказываем: Over 2.5 (1) или Under 2.5 (0)
    df['target'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)

    # Сколько Over vs Under в данных
    over_count = df['target'].sum()
    under_count = len(df) - over_count
    print(f"   ✅ Целевая переменная: Over={over_count} ({over_count/len(df)*100:.1f}%), Under={under_count} ({under_count/len(df)*100:.1f}%)")

    return df


# ============================================================
# ФУНКЦИЯ 3: Обучение и оценка модели
# ============================================================
def train_and_evaluate(df):
    """
    Обучает XGBoost и проверяет качество на тестовой выборке.

    ЧТО ДЕЛАЕТ:
    - Делит данные на train (80% ранних матчей) и test (20% поздних)
    - Обучает XGBoost на train
    - Предсказывает на test
    - Считает LogLoss и другие метрики
    - Показывает важность признаков

    ПОЧЕМУ ТАКОЙ СПЛИТ:
    Хронологический сплит (80/20 по времени) — это честно.
    Мы не можем использовать будущие матчи для обучения.
    """
    print("\n🧠 ОБУЧЕНИЕ МОДЕЛИ")

    # --- 3.1: Выбираем признаки ---
    # Список колонок которые модель будет использовать
    feature_columns = [
        'home_position', 'away_position',
        'home_goals_for_5', 'home_goals_against_5',
        'away_goals_for_5', 'away_goals_against_5',
        'home_over_25_pct', 'away_over_25_pct',
        'league_avg_goals'
    ]

    # --- 3.2: Хронологический сплит ---
    # Делим по времени: первые 80% матчей — обучение, последние 20% — тест
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_columns]
    y_train = train_df['target']
    X_test = test_df[feature_columns]
    y_test = test_df['target']

    print(f"   Train: {len(train_df):,} матчей ({train_df['Date'].min().date()} → {train_df['Date'].max().date()})")
    print(f"   Test:  {len(test_df):,} матчей ({test_df['Date'].min().date()} → {test_df['Date'].max().date()})")

    # Проверяем что нет пересечения дат (защита от утечки)
    assert train_df['Date'].max() <= test_df['Date'].min(), "❌ Утечка: даты train и test пересекаются!"
    print("   ✅ Утечка данных исключена")

    # --- 3.3: Обучение XGBoost ---
    # Создаём модель с базовыми параметрами
    model = xgb.XGBClassifier(
        n_estimators=200,       # 200 деревьев решений
        max_depth=5,            # Максимальная глубина дерева (ограничиваем чтобы не переобучаться)
        learning_rate=0.05,     # Скорость обучения (маленькая = стабильнее)
        random_state=42,        # Фиксируем случайность для воспроизводимости
        verbosity=0             # Не выводим лишнюю информацию
    )

    # Обучаем модель на тренировочных данных
    model.fit(X_train, y_train)

    print("   ✅ Модель обучена")

    # --- 3.4: Предсказание и оценка ---
    # Получаем вероятности для тестовых данных
    # predict_proba возвращает ДВЕ вероятности: [Under, Over]
    # Нам нужна вероятность Over (второй столбец)
    test_probs = model.predict_proba(X_test)[:, 1]

    # Предсказанные классы (0 или 1)
    test_preds = model.predict(X_test)

    # Считаем метрики
    ll = log_loss(y_test, test_probs)           # LogLoss — основная метрика
    acc = accuracy_score(y_test, test_preds)     # Точность

    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   LogLoss: {ll:.4f}  (базовый уровень: 0.693)")
    print(f"   Точность: {acc:.2%}")

    # Сравниваем с "всегда говори Under"
    baseline_acc = 1 - y_test.mean()  # доля Under в тесте
    print(f"   Базовая точность (всегда Under): {baseline_acc:.2%}")

    # --- 3.5: Важность признаков ---
    print("\n📊 ВАЖНОСТЬ ПРИЗНАКОВ:")
    # Получаем важность от модели
    importance = model.feature_importances_

    # Сортируем по важности
    indices = np.argsort(importance)[::-1]

    for i, idx in enumerate(indices):
        feature_name = feature_columns[idx]
        imp_value = importance[idx]
        # Рисуем простой бар
        bar = '█' * int(imp_value * 50)
        print(f"   {i+1}. {feature_name:<25}: {imp_value:.4f} {bar}")

    return model, feature_columns, ll


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("🚀 ЭТАП 1: БАЗОВАЯ МОДЕЛЬ XGBoost ДЛЯ ТОТАЛОВ")
    print("=" * 55)

    # Шаг 1: Загружаем данные
    df = load_data('data/all_leagues.csv')

    # Шаг 2: Создаём признаки
    df = create_features(df)

    # Шаг 3: Обучаем и оцениваем
    model, features, logloss = train_and_evaluate(df)

    print("\n✅ ЭТАП 1 ЗАВЕРШЁН")
    print(f"   LogLoss: {logloss:.4f}")
    if logloss < 0.693:
        print("   ✅ Модель лучше случайного угадывания!")
    else:
        print("   ⚠️  Модель не лучше случайного — нужно больше признаков")
