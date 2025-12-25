# Abalone Model Dashboard

Система для обучения модели градиентного бустинга на данных abalone и создания дашборда для интерпретации модели в отдельных Docker контейнерах.

## Архитектура

- **abalone-app**: Основное приложение с Flask API для предсказаний (порт 5000)
- **abalone-dashboard**: Дашборд с explainerdashboard для интерпретации модели (порт 9050)
- **nginx**: Обратный прокси для маршрутизации запросов (порт 80)

## Требования

- Docker
- Docker Compose

## Быстрый запуск

```bash
# Запуск всех сервисов
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

## Доступ к сервисам

- **Главная страница**: http://localhost/ → автоматически перенаправляется на дашборд
- **API предсказаний**: http://localhost/api/predict
- **Дашборд**: http://localhost/dashboard/
- **Nginx**: http://localhost/

## API Endpoints

### POST /api/predict

Предсказание количества колец abalone.

**Запрос:**
```json
{
  "Sex": "M",
  "Length": 0.5,
  "Diameter": 0.4,
  "Height": 0.1,
  "Whole_weight": 0.5,
  "Shucked_weight": 0.2,
  "Viscera_weight": 0.1,
  "Shell_weight": 0.1
}
```

**Ответ:**
```json
{
  "Rings": 10.5
}
```