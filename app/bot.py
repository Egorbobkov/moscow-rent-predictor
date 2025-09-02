import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import pandas as pd
import joblib
import numpy as np
from catboost import CatBoostRegressor
from dotenv import load_dotenv
import os

load_dotenv()
API_TOKEN = os.getenv("TELEGRAM_TOKEN")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Загружаем модель и данные
model = CatBoostRegressor()
model.load_model("models/catboost_model.cbm")
feature_columns = joblib.load("models/feature_columns.pkl")
df = pd.read_csv("data/cian_train_10features.csv")


@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer(
        "Привет! \n"
        "Отправь мне ссылку на квартиру с ЦИАН"
        "и я оценю её цену аренды.\n"
        "Например: https://www.cian.ru/some-apartment"
    )

@dp.message()
async def handle_url(message: types.Message):
    url = message.text.strip()
    row = df[df["url"] == url]

    if row.empty:
        await message.answer("К сожалению, этой квартиры нет в моём датасете")
        return

    features = row[feature_columns]
    log_pred = model.predict(features)[0]
    price = float(np.exp(log_pred))

    # Данные для описания
    street = row["street"].values[0] if "street" in row else "неизвестно"
    rooms = row["rooms_count"].values[0] if "rooms_count" in row else "?"
    area = row["total_meters"].values[0] if "total_meters" in row else "?"
    metro_dist = row["metro_distance_m"].values[0] if "metro_distance_m" in row else "неизвестно"
    real_price = row["average_price"].values[0] if "average_price" in row else None

    # Разница с реальной ценой
    if real_price:
        diff = round(price - real_price)
        if diff > 0:
            price_comment = f"Эта цена завышена примерно на {diff} ₽"
        elif diff < 0:
            price_comment = f"Эта цена ниже средней на примерно {abs(diff)} ₽"
        else:
            price_comment = "Цена совпадает с реальной средней ценой"
    else:
        price_comment = "Нет данных о реальной цене для сравнения"

    msg = (
        f"Квартира на улице: {street}\n"
        f"Комнат: {rooms}, Площадь: {area} м²\n"
        f"До метро: {round(metro_dist)} м\n"
        f"Оценочная цена аренды: {round(price)} ₽/мес\n\n"
        f"{price_comment}"
    )

    await message.answer(msg)

async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
