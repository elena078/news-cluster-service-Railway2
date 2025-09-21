import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class Config:
    # Токен бота из переменных окружения
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    
    # Проверяем, что токен есть
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN не найден в переменных окружения!")
    
    # Настройки для Railway
    PORT = int(os.getenv('PORT', 8000))
    HOST = os.getenv('HOST', '0.0.0.0')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Создаем экземпляр конфигурации
config = Config()
