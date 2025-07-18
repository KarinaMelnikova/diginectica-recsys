# 🧠 Product Co-View Recommender

Скрипт для построения товарных рекомендаций на основе совместных просмотров (co-view) из пользовательских сессий.  

Алгоритм анализирует сессии пользователей из датасета DIGINETICA и строит item-to-item рекомендации: *если пользователи часто смотрели товары A и B вместе — система будет рекомендовать товар B тем, кто смотрит A.*

---

## 📌 Что делает этот проект

✅ Загружает сессии пользователей  
✅ Находит пары товаров, часто встречающихся вместе  
✅ Генерирует рекомендации (top-N) для каждого товара  
✅ Сохраняет результат в `.csv` файле

---

## 🚀 Быстрый старт

1. Установите зависимости:

    ```bash
    pip install pandas
    ```

2. Подготовьте входной файл `input_data/train-item-views.csv`

3. Запустите скрипт:

    ```bash
    python build_recommendations.py
    ```

4. Готовый файл появится `output_data/recommendations.csv`

---

### 📥 Где взять входной файл

Датасет DIGINETICA доступен для скачивания по ссылке:

🔗 [www.kaggle.com](https://www.kaggle.com/datasets/profalbusdumbledore/diginetica-dataset?resource=download&select=train-item-views.csv)

Скачайте архив, распакуйте его и поместите файл `train-item-views.csv` в папку `input_data/`.

> **Важно:** используется только файл `train-item-views.csv`. Остальные можно не загружать.

---

## 📄 Формат входного файла

```text
sessionId;userId;itemId;timeframe;eventdate
1;NA;81766;526309;2016-05-09
1;NA;31331;1031018;2016-05-09
```

- `sessionId` — уникальный идентификатор сессии пользователя (группировка взаимодействий). **Обязательный.**
- `userId` — идентификатор пользователя (может отсутствовать, не используется).
- `itemId` — уникальный идентификатор товара. **Обязательный.**
- `timeframe` — временная метка внутри сессии (порядок событий, не используется).
- `eventdate` — дата события (не используется).

---

## 📄 Формат выходного файла

```text
seed_item_id,recommended_product_id,score
81766,31331,12
81766,32118,7
...
```

- `seed_item_id` — товар, для которого строится рекомендация
- `recommended_product_id` — товар, рекомендуемый к нему
- `score` — количество сессий, в которых эти товары были вместе

---

## ⚙️ Настройки по умолчанию

| Параметр    | Значение | Описание                                   |
|-------------|----------|--------------------------------------------|
| `top_k`     | 10       | Сколько рекомендаций на каждый товар       |
| `min_score` | 5        | Минимум совпадений для включения в вывод  |

---

## 👩‍💻 Применение

Готовый результат можно использовать для интеграции в рекомендательные сервисы или для дальнейшего анализа пользовательского поведения.

Данный подход легко масштабируется и адаптируется под различные типы данных и задачи рекомендаций.
