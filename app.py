from fastapi import FastAPI
import asyncio
import os
from bot import TelegramBot

app = FastAPI()

bot_instance = None

@app.on_event("startup")
async def startup_event():
    global bot_instance
    telegram_token = os.environ.get('telegram_token')
    bot_instance = TelegramBot(telegram_token)
    # Запуск бота в фоне
    asyncio.create_task(bot_instance.run())

@app.get("/")
async def read_root():
    return {"message": "Telegram bot is running"}

@app.get("/stop")
async def stop_bot():
    if bot_instance:
        await bot_instance.shutdown()
        return {"message": "Bot stopped"}
    return {"message": "Bot was not running"}
