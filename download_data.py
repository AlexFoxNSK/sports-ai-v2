#!/usr/bin/env python3
"""
Этап 0: Загрузка данных с football-data.co.uk
Скачивает все сезоны для топ-лиг и объединяет в один CSV
"""
import pandas as pd
import requests
import os
from io import StringIO

# Конфигурация
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Лиги и их коды на football-data.co.uk
# Формат: (папка_для_raw, код_лиги, название_лиги, сезоны)
LEAGUES = [
    ('raw', 'E0', 'England_Premier', ['2122', '2223', '2324', '2425', '2526']),
    ('raw', 'SP1', 'Spain_LaLiga', ['2122', '2223', '2324', '2425', '2526']),
    ('raw', 'D1', 'Germany_Bundesliga', ['2122', '2223', '2324', '2425', '2526']),
    ('raw', 'I1', 'Italy_SerieA', ['2122', '2223', '2324', '2425', '2526']),
    ('raw', 'F1', 'France_Ligue1', ['2122', '2223', '2324', '2425', '2526']),
    ('raw', 'N1', 'Netherlands_Eredivisie', ['2122', '2223', '2324', '2425', '2526']),
    ('raw', 'P1', 'Portugal_Primeira', ['2122', '2223', '2324', '2425', '2526']),
    ('raw', 'B1', 'Belgium_ProLeague', ['2122', '2223', '2324', '2425', '2526']),
    ('raw', 'SC0', 'Scotland_Premiership', ['2122', '2223', '2324', '2425', '2526']),
]

# Базовый URL
BASE_URL = 'https://www.football-data.co.uk/mmz4281'

print('=' * 50)
print('📥 Загрузка данных с football-data.co.uk')
print('=' * 50)

all_dfs = []

for folder, code, league_name, seasons in LEAGUES:
    print(f'\n📊 {league_name}:')
    
    for season in seasons:
        url = f'{BASE_URL}/{season}/{code}.csv'
        print(f'   ⬇️  {season}...', end=' ')
        
        try:
            df = pd.read_csv(url, low_memory=False)
            df['league'] = league_name
            df['season'] = f'{season[:2]}/{season[2:]}'
            all_dfs.append(df)
            print(f'{len(df)} матчей ✅')
        except Exception as e:
            print(f'❌ {e}')

if all_dfs:
    # Объединяем все лиги
    result = pd.concat(all_dfs, ignore_index=True)
    
    # Оставляем только нужные колонки
    columns_to_keep = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
        'HTHG', 'HTAG', 'HTR',
        'B365H', 'B365D', 'B365A',  # Bet365 кэфы
        'B365>2.5', 'B365<2.5',     # Bet365 тотал!
        'HS', 'AS', 'HST', 'AST',    # Удары
        'HC', 'AC',                   # Угловые
        'league', 'season'
    ]
    
    # Оставляем только те колонки, которые есть в данных
    available_columns = [c for c in columns_to_keep if c in result.columns]
    result = result[available_columns]
    
    # Сохраняем
    output_path = os.path.join(DATA_DIR, 'all_leagues.csv')
    result.to_csv(output_path, index=False)
    
    print(f'\n✅ Сохранено: {output_path}')
    print(f'   📊 Всего матчей: {len(result):,}')
    print(f'   🏆 Лиг: {result["league"].nunique()}')
    print(f'   📅 Период: {result["Date"].min()} — {result["Date"].max()}')
else:
    print('\n❌ Не удалось загрузить данные')
