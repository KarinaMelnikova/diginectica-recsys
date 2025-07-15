import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    Загружает датасет DIGINETICA с нужными колонками.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    df = pd.read_csv(filepath, sep=";", usecols=["sessionId", "itemId"])
    return df


def build_co_view_counts(df: pd.DataFrame) -> dict:
    """
    Создаёт словарь co-view счётчиков: для каждого товара - список сопутствующих с их частотой.
    """
    session_groups = df.groupby("sessionId")["itemId"].apply(list)
    co_view = defaultdict(Counter)

    for items in session_groups:
        unique_items = list(set(items))
        for item_a, item_b in combinations(unique_items, 2):
            co_view[item_a][item_b] += 1
            co_view[item_b][item_a] += 1

    return co_view


def generate_recommendations(
    co_view: dict, top_k: int = 10, min_score: int = 5
) -> pd.DataFrame:
    """
    Генерирует рекомендации для каждого товара на основе co-view данных.
    """
    rows = []

    for seed_item, rec_items in co_view.items():
        for rec_item, score in rec_items.most_common(top_k):
            if score >= min_score:
                rows.append((seed_item, rec_item, score))

    return pd.DataFrame(
        rows, columns=["seed_item_id", "recommended_product_id", "score"]
    )


def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    """
    Сохраняет рекомендации в CSV-файл.
    """
    df.to_csv(filename, index=False)
    print(f"Файл сохранён: {filename}")


def main():
    input_path = "input_data/train-item-views.csv"
    output_path = "output_data/recommendations.csv"

    print("📥 Загрузка данных...")
    df = load_data(input_path)

    print("🔢 Подсчёт совместных просмотров...")
    co_view_counts = build_co_view_counts(df)

    print("🧠 Генерация рекомендаций...")
    recommendations_df = generate_recommendations(co_view_counts)

    print("💾 Сохранение результатов...")
    save_to_csv(recommendations_df, output_path)

    print("✅ Готово!")


if __name__ == "__main__":
    main()
