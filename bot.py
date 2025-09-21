import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import pandas as pd
import os
import tempfile
import io
from datetime import datetime
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import pymorphy3
import umap.umap_ as umap
import hdbscan
import openpyxl
import asyncio
import nest_asyncio
import signal
import sys
import nest_asyncio
nest_asyncio.apply()
# Затем запускайте
await main()
# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# ПРЕДВАРИТЕЛЬНАЯ ЗАГРУЗКА STOPWORDS
try:
    nltk.download('stopwords', quiet=True)
except:
    pass
class NewsClusterer:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.vectorizer = None
        self.clusterer = None
        self.tfidf_matrix = None
        self.performance_metrics = {}
        self._stop_words = None
        self.umap_reducer = None
        self.svd_reducer = None
    def _initialize_stopwords(self):
        """Инициализация стоп-слов"""
        if self._stop_words is None:
            russian_stop_words = [
                'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то',
                'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за',
                'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет',
                'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если',
                'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять',
                'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они',
                'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была',
                'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
                'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем',
                'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
                'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два',
                'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас',
                'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем',
                'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя',
                'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
            ]
            try:
                english_stop_words = stopwords.words('english')
            except:
                english_stop_words = []
            self._stop_words = list(set(russian_stop_words + english_stop_words))
    def preprocess_text(self, text):
        """Предобработка русского текста"""
        if not isinstance(text, str):
            return ""
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление пунктуации, цифр, лишних символов
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Токенизация и лемматизация
        words = text.split()
        lemmas = []
        for word in words:
            if len(word) > 2:  # Игнорируем очень короткие слова
                try:
                    parsed = self.morph.parse(word)[0]
                    lemma = parsed.normal_form
                    lemmas.append(lemma)
                except:
                    lemmas.append(word)
        return ' '.join(lemmas)
    def vectorize_texts(self, texts):
        """Векторизация текстов с помощью TF-IDF"""
        self._initialize_stopwords()
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words=self._stop_words
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        return tfidf_matrix
    def reduce_dimensionality(self, tfidf_matrix):
        """Снижение размерности с помощью UMAP"""
        n_components = min(50, tfidf_matrix.shape[1], tfidf_matrix.shape[0] - 1)
        # Используем TruncatedSVD для sparse матриц
        self.svd_reducer = TruncatedSVD(n_components=n_components, random_state=42)
        svd_features = self.svd_reducer.fit_transform(tfidf_matrix)
        self.umap_reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(15, len(svd_features) - 1),
            min_dist=0.1,
            metric='cosine',
            low_memory=True
        )
        umap_features = self.umap_reducer.fit_transform(svd_features)
        return umap_features
    def cluster_news(self, features):
        """Кластеризация с помощью HDBSCAN"""
        min_cluster_size = 5
        min_samples = max(3, min_cluster_size // 2)  # min_samples = 3 для min_cluster_size = 5
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.0,
            gen_min_span_tree=True,
            metric='euclidean'
        )
        clusters = self.clusterer.fit_predict(features)
        return clusters
    def assign_cluster_names_centroid_sparse(self, df, clusters, tfidf_matrix):
        """Присвоение названий кластерам на основе центроидов"""
        cluster_labels_names = {}
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:
                continue
            cluster_mask = clusters == cluster_id
            cluster_docs = tfidf_matrix[cluster_mask]
            if cluster_docs.shape[0] == 0:
                continue
            # Вычисляем центроид
            centroid = np.array(cluster_docs.mean(axis=0)).flatten()
            # Вычисляем схожести
            similarities = []
            for i in range(cluster_docs.shape[0]):
                doc_vector = cluster_docs[i].toarray().flatten()
                similarity = cosine_similarity([centroid], [doc_vector])[0][0]
                similarities.append(similarity)
            # Находим ближайший документ
            closest_idx = np.argmax(similarities)
            cluster_indices = np.where(cluster_mask)[0]
            closest_doc_idx = cluster_indices[closest_idx]
            cluster_labels_names[cluster_id] = df.iloc[closest_doc_idx]['title']
        return cluster_labels_names
    def run_pipeline_for_dataframe(self, df):
        """Основной пайплайн обработки данных"""
        if len(df) == 0:
            raise ValueError("Получен пустой DataFrame")
        # Подготовка данных
        df = df.copy()
        df['published'] = pd.to_datetime(df['published'], errors='coerce')
        # Удаляем строки с NaN в критичных полях
        df = df.dropna(subset=['title', 'lead'])
        # Заполняем пропущенные значения
        df['title'] = df['title'].fillna('')
        df['lead'] = df['lead'].fillna('')
        df['full_text'] = df['title'] + ' ' + df['lead']
        df['full_text'] = df['full_text'].str.slice(0, 300)
        df['processed_text'] = df['full_text'].apply(self.preprocess_text)
        # Удаляем пустые тексты после обработки
        df = df[df['processed_text'].str.len() > 10]
        if len(df) == 0:
            raise ValueError("Нет данных для обработки после очистки текстов")
        # Проверяем, что есть достаточно данных для кластеризации
        if len(df) < 5:
            # Если данных мало, возвращаем простую категоризацию
            df['cluster'] = 0
            df['news_feed_label'] = 'Мало данных для кластеризации'
            return df, [('Мало данных для кластеризации', len(df))]
        try:
            # Векторизация, снижение размерности, кластеризация
            tfidf_matrix = self.vectorize_texts(df['processed_text'])
            # Проверяем, что матрица не пустая
            if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
                raise ValueError("Матрица признаков пустая после векторизации")
            umap_features = self.reduce_dimensionality(tfidf_matrix)
            clusters = self.cluster_news(umap_features)
            df['cluster'] = clusters
            # Присвоение названий кластерам
            cluster_names_centroid = self.assign_cluster_names_centroid_sparse(df, clusters, tfidf_matrix)
            df['news_feed_label'] = df['cluster'].apply(lambda x: cluster_names_centroid.get(x, "Без категории") if x != -1 else "Шум")
            # Формирование топа кластеров
            valid_clusters = df[df['cluster'] != -1]
            if len(valid_clusters) > 0:
                cluster_sizes = valid_clusters['cluster'].value_counts()
                top_clusters_info = []
                for cluster_id, size in cluster_sizes.head(10).items():
                    cluster_name = cluster_names_centroid.get(cluster_id, "Без названия")
                    top_clusters_info.append((cluster_name, size))
            else:
                top_clusters_info = [("Не удалось выделить кластеры", len(df))]
            return df, top_clusters_info
        except Exception as e:
            # Fallback: простая категоризация при ошибке ML
            logger.error(f"Ошибка ML обработки: {str(e)}")
            df['cluster'] = 0
            df['news_feed_label'] = 'Ошибка обработки'
            return df, [('Ошибка ML обработки', len(df))]
    def create_excel_bytes(self, df):
        """Создает Excel файл в байтах из DataFrame."""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Clustered News')
        output.seek(0)
        return output.getvalue()
class TelegramBot:
    def __init__(self, telegram_token: str):
        self.telegram_token = telegram_token
        self.clusterer = NewsClusterer()
        self.application = None
        self._stop_event = asyncio.Event()
    async def initialize(self):
        """Асинхронная инициализация бота"""
        try:
            self.application = Application.builder().token(self.telegram_token).build()
            # Регистрация обработчиков команд
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            # Регистрация обработчиков документов (главный функционал)
            self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
            # Регистрация обработчика текстовых сообщений
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
            # Общий обработчик для всех остальных типов сообщений
            self.application.add_handler(MessageHandler(filters.ALL, self.handle_unknown))
            # Обработчик сигналов для graceful shutdown
            loop = asyncio.get_running_loop()
            for sig in [signal.SIGINT, signal.SIGTERM]:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            return True
        except Exception as e:
            logger.error(f"Ошибка при инициализации бота: {str(e)}")
            return False
    async def start_command(self, update: Update, context: CallbackContext):
        """Обработчик команды /start"""
        await update.message.reply_text(
            "Привет! Я бот для кластеризации новостей с использованием машинного обучения.\n"
            "Отправьте мне файл в формате CSV или Excel с данными новостей.\n"
            "Обязательные поля: title, lead"
        )
    async def help_command(self, update: Update, context: CallbackContext):
        """Обработчик команды /help"""
        help_text = (
            "📋 Как пользоваться ботом:\n\n"
            "1. 📎 Отправьте файл в формате CSV или Excel\n"
            "2. ✅ Файл должен содержать обязательные поля:\n"
            "   - title (заголовок новости)\n"
            "   - lead (текст новости или первый абзац)\n"
            "3. ⏳ Я обработаю файл с помощью ML алгоритмов\n"
            "4. 📊 Вы получите кластеризованные данные и топ-10 инфоповодов\n\n"
            "Используемые технологии: TF-IDF, UMAP, HDBSCAN"
        )
        await update.message.reply_text(help_text)
    async def handle_text(self, update: Update, context: CallbackContext):
        """Обработчик текстовых сообщений"""
        user_text = update.message.text.lower()
        # Ответы на частые вопросы
        if any(word in user_text for word in ['привет', 'hello', 'hi', 'здравствуй']):
            await update.message.reply_text(
                "👋 Привет! Я бот для анализа новостей. "
                "Отправьте мне файл с данными для кластеризации."
            )
        elif any(word in user_text for word in ['как дела', 'как ты', 'how are you']):
            await update.message.reply_text(
                "🤖 У меня все отлично! Готов анализировать ваши новостные данные. "
                "Присылайте CSV или Excel файл!"
            )
        elif any(word in user_text for word in ['спасибо', 'thanks', 'thank you']):
            await update.message.reply_text(
                "😊 Пожалуйста! Если нужна помощь с файлом - используйте /help"
            )
        elif any(word in user_text for word in ['файл', 'file', 'данные', 'data']):
            await update.message.reply_text(
                "📎 Для анализа нужен файл в формате CSV или Excel с колонками:\n"
                "• title - заголовок новости\n" 
                "• lead - текст новости\n\n"
                "Просто отправьте мне такой файл как документ!"
            )
        else:
            await update.message.reply_text(
                "📝 Я получил ваше текстовое сообщение!\n"
                "Но я специализируюсь на анализе новостных данных. "
                "Отправьте мне CSV или Excel файл с колонками:\n"
                "• title (заголовок)\n"
                "• lead (текст)\n\n"
                "Нужна помощь? Используйте /help"
            )
            
    async def handle_photo(self, update: Update, context: CallbackContext):
        """Обработчик фотографий"""
        await update.message.reply_text(
            "📸 Я вижу, что вы отправили фото!\n"
            "Но я специализируюсь на анализе новостных данных. "
            "Отправьте мне CSV или Excel файл с новостями для кластеризации.\n"
            "Используйте /help для инструкции."
        )
    async def handle_video(self, update: Update, context: CallbackContext):
        """Обработчик видео"""
        await update.message.reply_text(
            "🎥 Видео принято!\n"
            "Однако мой функционал - анализ текстовых данных новостей. "
            "Пожалуйста, отправьте файл с расширением .csv или .xlsx "
            "с колонками: published, title, lead"
        )
    async def handle_audio(self, update: Update, context: CallbackContext):
        """Обработчик аудио"""
        await update.message.reply_text(
            "🎵 Аудио сообщение получено!\n"
            "Я работаю с текстовыми данными новостей. "
            "Отправьте мне файл для анализа новостей в формате CSV или Excel."
        )
    async def handle_voice(self, update: Update, context: CallbackContext):
        """Обработчик голосовых сообщений"""
        await update.message.reply_text(
            "🎤 Голосовое сообщение!\n"
            "Для работы мне нужны текстовые данные. "
            "Пожалуйста, отправьте файл с новостями (.csv или .xlsx) "
            "с обязательными полями: published, title, lead"
        )
    async def handle_sticker(self, update: Update, context: CallbackContext):
        """Обработчик стикеров"""
        await update.message.reply_text(
            "😊 Забавный стикер!\n"
            "Но я жду от вас файл с новостными данными. "
            "Отправьте CSV или Excel файл для анализа."
        )
    async def handle_location(self, update: Update, context: CallbackContext):
        """Обработчик геолокации"""
        await update.message.reply_text(
            "📍 Геолокация получена!\n"
            "Мой функционал - кластеризация новостей. "
            "Пришлите мне файл с данными для анализа."
        )
    async def handle_contact(self, update: Update, context: CallbackContext):
        """Обработчик контактов"""
        await update.message.reply_text(
            "📞 Контакт сохранен!\n"
            "Теперь отправьте мне файл с новостями в формате CSV или Excel "
            "для проведения анализа и кластеризации."
        )
    async def handle_unknown(self, update: Update, context: CallbackContext):
        """Обработчик неизвестных типов сообщений"""
        await update.message.reply_text(
            "🤔 Я не совсем понимаю, что вы от меня хотите.\n"
            "Я бот для анализа новостных данных. "
            "Отправьте мне файл в формате CSV или Excel с колонками:\n"
            "- title (заголовок новости)\n"
            "- lead (текст новости)\n\n"
            "Используйте /help для подробной инструкции."
        )
    async def handle_document(self, update: Update, context: CallbackContext):
        """Обработчик документов с ML обработкой"""
        try:
            user = update.message.from_user
            logger.info(f"Получен файл от пользователя {user.username}")
            
            # Проверяем тип файла
            document = update.message.document
            file_name = document.file_name.lower()
            
            if not (file_name.endswith('.csv') or file_name.endswith('.xlsx')):
                await update.message.reply_text("❌ Поддерживаются только CSV и Excel файлы")
                return
                
            await update.message.reply_text("⏳ Обрабатываю файл с помощью ML... Это может занять несколько минут")
            
            # Скачиваем файл
            file = await document.get_file()
            file_bytes = await file.download_as_bytearray()
            
            # Читаем файл
            df = self.read_file(file_bytes, file_name)
            
            # Проверяем обязательные поля
            required_columns = ['title', 'lead']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                await update.message.reply_text(
                    f"❌ В файле отсутствуют обязательные поля: {', '.join(missing_columns)}"
                )
                return
            
            # Обрабатываем данные с помощью ML (синхронный вызов в отдельном потоке)
            await update.message.reply_text("🔍 Запускаю ML обработку...")
            
            # Запускаем синхронную ML обработку в отдельном потоке
            loop = asyncio.get_running_loop()
            result_df, top_clusters = await loop.run_in_executor(
                None, self.clusterer.run_pipeline_for_dataframe, df
            )
            
            await update.message.reply_text("✅ ML обработка завершена, формирую результат...")
            
            # Создаем Excel файл с результатами
            result_df = result_df.drop(columns=['full_text', 'processed_text'])
            excel_bytes = self.clusterer.create_excel_bytes(result_df)
            output = io.BytesIO(excel_bytes)
            
            # Отправляем результат
            await update.message.reply_document(
                document=output,
                filename='clustered_news.xlsx',
                caption="✅ Файл обработан!"
            )
            
            # Отправляем топ-10 инфоповодов
            await self.send_top_clusters(top_clusters, update)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке файла: {str(e)}", exc_info=True)
            await update.message.reply_text(f"❌ Произошла ошибка при обработке: {str(e)}")
    def read_file(self, file_bytes: bytes, file_name: str) -> pd.DataFrame:
        """Чтение файла из байтов"""
        try:
            if file_name.endswith('.csv'):
                # Пробуем разные кодировки для CSV
                encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'latin1']
                for encoding in encodings:
                    try:
                        return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Не удалось определить кодировку CSV файла")
                
            elif file_name.endswith('.xlsx'):
                return pd.read_excel(io.BytesIO(file_bytes))
                
        except Exception as e:
            logger.error(f"Ошибка чтения файла: {str(e)}")
            raise ValueError(f"Ошибка чтения файла: {str(e)}")
    async def send_top_clusters(self, top_clusters, update: Update):
        """Отправка топ-10 инфоповодов"""
        try:
            if not top_clusters:
                await update.message.reply_text("📊 Не удалось выделить значимые кластеры")
                return
            
            message = "🏆 Топ-10 инфоповодов:\n\n"
            for i, (cluster_name, count) in enumerate(top_clusters, 1):
                message += f"{i}. {cluster_name} - {count} новостей\n"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            logger.error(f"Ошибка при формировании топа кластеров: {str(e)}")
            await update.message.reply_text("📊 Произошла ошибка при анализе кластеров")
    async def shutdown(self):
        """Graceful shutdown бота"""
        logger.info("Остановка бота...")
        self._stop_event.set()
        if self.application:
            await self.application.shutdown()
        logger.info("Бот остановлен")
    async def run(self):
        """Запуск бота"""
        try:
            if not await self.initialize():
                return
                
            logger.info("Запуск ML бота...")
            
            # Запускаем поллинг
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            # Ждем сигнала остановки
            await self._stop_event.wait()
            
            # Останавливаем бота
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            
        except Exception as e:
            logger.error(f"Ошибка при запуске бота: {str(e)}", exc_info=True)
        finally:
            # Гарантируем cleanup
            if self.application and self.application.running:
                await self.application.stop()
                await self.application.shutdown()
# Основная функция
async def main():
    # Получаем токен бота
    telegram_token = os.environ.get('telegram_token')
    
    if not telegram_token:
        logger.error("Токен бота не найден! Установите переменную окружения telegram_token")
        return
    
    # Создаем и запускаем бота
    bot = TelegramBot(telegram_token)
    await bot.run()
if __name__ == "__main__":
    # Правильный запуск асинхронного кода
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}", exc_info=True)

nest_asyncio.apply()

# Запускаем
await main()
