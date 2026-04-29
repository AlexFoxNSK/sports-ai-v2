# Восстановление проекта на новом сервере

## 1. Копируем архив
scp root@45.153.69.37:~/sports-ai-v2-backup-*.tar.gz .

## 2. Распаковываем
tar -xzf sports-ai-v2-backup-*.tar.gz

## 3. Устанавливаем зависимости
cd sports-ai-v2
python3 -m venv .venv
source .venv/bin/activate
pip install xgboost pandas numpy scikit-learn flask requests

## 4. Запускаем дашборд
nohup python3 dashboard.py > logs/dashboard.log 2>&1 &

## 5. Live-прогноз
python3 live_predictions_v2.py
