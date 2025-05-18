# Импортируем необходимые библиотеки для работы бота
import os
import asyncio
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import Command, CommandStart
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from datetime import datetime
import torch
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
import re
from nltk.tokenize import word_tokenize
import nltk

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('instance/bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv('tg_env.env')
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

if not TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN не найден в файле tg_env.env")
    raise ValueError("TELEGRAM_BOT_TOKEN не найден в файле tg_env.env")

# Инициализация бота
bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Инициализация моделей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paraphraser_model = None
paraphraser_tokenizer = None
headliner_model = None
headliner_tokenizer = None


# Состояния бота
class ProcessingStates(StatesGroup):
    waiting_for_text = State()


# Клавиатуры
def get_main_keyboard():
    buttons = [
        [KeyboardButton(text="🤩 Простая версия")],
        [KeyboardButton(text="🔄 Перефразировать")],
        [KeyboardButton(text="✏️ Исправить ошибки")],
        [KeyboardButton(text="📌 Заголовок")]
    ]
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)


def get_cancel_keyboard():
    buttons = [[KeyboardButton(text="❌ Отменить")]]
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)


# Функции обработки текста
def correct_spelling(text, max_length=4000, batch_size=4):
    logger.info(f"Начало исправления орфографии для текста: {text[:100]}...")
    try:
        MODEL_NAME = 'UrukHan/t5-russian-spell'
        tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        model.eval()  # Переводим модель в режим оценки для ускорения

        task_prefix = "Spell correct: "

        # Предварительная загрузка модели в GPU (если доступен)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            dummy_input = tokenizer("test", return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model.generate(**dummy_input, max_length=10)

        # Разбиваем текст на части
        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        results = []

        # Обрабатываем батчами для ускорения
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Токенизация батча
            encoded = tokenizer(
                [task_prefix + chunk for chunk in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            # Генерация с отключенным вычислением градиентов
            with torch.no_grad():
                predicts = model.generate(
                    **encoded,
                    max_length=max_length,
                    num_beams=3,  # Уменьшаем для скорости
                    early_stopping=True
                )

            # Декодируем результаты
            batch_results = tokenizer.batch_decode(
                predicts,
                skip_special_tokens=True
            )
            results.extend(batch_results)

        return " ".join(results)

    except Exception as e:
        logger.error(f"Ошибка при исправлении орфографии: {str(e)}")
        raise

def paraphrase_text(text):
    global paraphraser_model, paraphraser_tokenizer
    logger.info(f"Начало перефразирования текста: {text[:100]}...")

    try:
        temp = 1.7
        top_k = 60
        top_p = 0.92

        inputs = paraphraser_tokenizer(
            text,
            return_tensors="pt",
            max_length=4000,
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = paraphraser_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=4000,
                num_beams=5,
                do_sample=True,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=2.5,
                early_stopping=True
            )

        decoded_output = paraphraser_tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = clean_paraphrase_output(decoded_output)
        logger.info(f"Успешно перефразирован текст. Результат: {result[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Ошибка при перефразировании: {str(e)}")
        raise


def clean_paraphrase_output(text):
    text = re.sub(r'^(перефразируй:|перефразируя:|подробнее:|дополнительно:)\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


def should_generate_headline(text):
    words = word_tokenize(text)
    return len(words) >= 20


def generate_short_headline(text):
    global headliner_model, headliner_tokenizer
    logger.info(f"Генерация заголовка для текста: {text[:100]}...")

    try:
        input_ids = headliner_tokenizer(
            text,
            return_tensors="pt",
            max_length=1000,
            truncation=True
        ).input_ids.to(device)

        output_ids = headliner_model.generate(
            input_ids=input_ids,
            max_length=20,
            min_length=5,
            num_beams=4,
            repetition_penalty=3.0,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        headline = headliner_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        headline = headline.split(".")[0].strip()
        words = [w for w in headline.split() if len(w) > 2][:7]
        result = " ".join(words).capitalize()
        logger.info(f"Успешно сгенерирован заголовок: {result}")
        return result
    except Exception as e:
        logger.error(f"Ошибка при генерации заголовка: {str(e)}")
        raise


# Обработчики команд
@dp.message(CommandStart())
@dp.message(Command("help"))
async def send_welcome(message: Message):
    logger.info(f"Пользователь {message.from_user.id} запустил бота")
    welcome_text = (
        "🤖 *Добро пожаловать в TextMasterBot!*\n\n"
        "Я помогу вам обработать текст:\n"
        "• *Простая версия* — упрощаю сложный текст\n"
        "• *Перефразировать* — переформулирую текст\n"
        "• *Исправить ошибки* — исправляю орфографию и грамматику\n"
        "• *Заголовок* — создаю заголовок для текста\n\n"
        "Выберите действие кнопкой ниже 👇"
    )
    await message.answer(welcome_text, reply_markup=get_main_keyboard())


@dp.message(lambda message: message.text in [
    "🤩 Простая версия",
    "🔄 Перефразировать",
    "✏️ Исправить ошибки",
    "📌 Заголовок"
])
async def process_action(message: Message, state: FSMContext):
    action_map = {
        "🤩 Простая версия": "simplify",
        "🔄 Перефразировать": "paraphrase",
        "✏️ Исправить ошибки": "spellcheck",
        "📌 Заголовок": "headline"
    }
    action = action_map[message.text]
    logger.info(f"Пользователь {message.from_user.id} выбрал действие: {action}")

    await state.set_data({"action": action})
    await message.answer(
        "📝 Отправьте текст для обработки:",
        reply_markup=get_cancel_keyboard()
    )
    await state.set_state(ProcessingStates.waiting_for_text)


@dp.message(ProcessingStates.waiting_for_text, lambda message: message.text == "❌ Отменить")
async def cancel_processing(message: Message, state: FSMContext):
    logger.info(f"Пользователь {message.from_user.id} отменил операцию")
    try:
        await state.clear()
        await message.answer(
            "❌ Действие отменено",
            reply_markup=get_main_keyboard()
        )
        logger.info(f"Состояние успешно очищено для пользователя {message.from_user.id}")
    except Exception as e:
        logger.error(f"Ошибка при отмене операции: {str(e)}")
        await message.answer(
            "⚠️ Произошла ошибка при отмене операции. Попробуйте еще раз.",
            reply_markup=get_main_keyboard()
        )


@dp.message(ProcessingStates.waiting_for_text)
async def process_text(message: Message, state: FSMContext):
    user_id = message.from_user.id
    logger.info(f"Начало обработки текста от пользователя {user_id}")

    try:
        user_data = await state.get_data()
        action = user_data.get("action")
        text = message.text.strip()

        if not text:
            logger.warning(f"Пользователь {user_id} отправил пустой текст")
            await message.answer(
                "❗ Пожалуйста, отправьте непустой текст!",
                reply_markup=get_cancel_keyboard()
            )
            return

        logger.info(f"Пользователь {user_id} отправил текст для обработки ({action}): {text[:100]}...")

        wait_msg = await message.answer("<b>⏳ Ожидайте...</b>", parse_mode=ParseMode.HTML)
        logger.info(f"Отправлено сообщение 'Ожидайте...' пользователю {user_id}")

        try:
            if action == "spellcheck":
                logger.info(f"Начало исправления орфографии для пользователя {user_id}")
                result = correct_spelling(text)
                # Разбиваем результат на части, если он слишком длинный
                max_message_length = 4096  # Максимальная длина сообщения в Telegram
                if len(result) > max_message_length:
                    chunks = [result[i:i+max_message_length] for i in range(0, len(result), max_message_length)]
                    await wait_msg.delete()  # Удаляем сообщение "Ожидайте..."
                    for i, chunk in enumerate(chunks, 1):
                        await message.answer(
                            f"<b>✅ Часть {i} из {len(chunks)}:</b>\n\n{chunk}",
                            parse_mode=ParseMode.HTML
                        )
                else:
                    await wait_msg.edit_text(
                        text=f"<b>✅ Результат:</b>\n\n{result}",
                        parse_mode=ParseMode.HTML
                    )
            elif action == "paraphrase":
                logger.info(f"Начало перефразирования для пользователя {user_id}")
                result = paraphrase_text(text)
                await wait_msg.edit_text(
                    text=f"<b>✅ Результат:</b>\n\n{result}",
                    parse_mode=ParseMode.HTML
                )
            elif action == "simplify":
                logger.info(f"Начало упрощения текста для пользователя {user_id}")
                corrected = correct_spelling(text)
                result = paraphrase_text(corrected)
                await wait_msg.edit_text(
                    text=f"<b>✅ Результат:</b>\n\n{result}",
                    parse_mode=ParseMode.HTML
                )
            elif action == "headline":
                logger.info(f"Проверка возможности генерации заголовка для пользователя {user_id}")
                if should_generate_headline(text):
                    result = generate_short_headline(text)
                    await wait_msg.edit_text(
                        text=f"<b>✅ Заголовок:</b>\n\n{result}",
                        parse_mode=ParseMode.HTML
                    )
                else:
                    await wait_msg.edit_text(
                        text="❗ Текст слишком короткий для создания заголовка (нужно минимум 20 слов).",
                        parse_mode=ParseMode.HTML
                    )
                    result = None

            logger.info(f"Успешно обработан текст для пользователя {user_id}. Результат отправлен.")

            await message.answer(
                "Выберите следующее действие:",
                reply_markup=get_main_keyboard()
            )
            logger.info(f"Отправлено меню выбора действий пользователю {user_id}")

        except Exception as e:
            logger.error(f"Ошибка при обработке текста для пользователя {user_id}: {str(e)}")
            await wait_msg.edit_text(
                text=f"<b>❌ Ошибка:</b> {str(e)}",
                parse_mode=ParseMode.HTML
            )
            await message.answer(
                "Выберите действие заново:",
                reply_markup=get_main_keyboard()
            )

    except Exception as e:
        logger.error(f"Критическая ошибка в обработчике текста для пользователя {user_id}: {str(e)}")
        await message.answer(
            "⚠️ Произошла критическая ошибка. Пожалуйста, попробуйте еще раз.",
            reply_markup=get_main_keyboard()
        )
    finally:
        try:
            await state.clear()
            logger.info(f"Состояние очищено для пользователя {user_id}")
        except Exception as e:
            logger.error(f"Ошибка при очистке состояния для пользователя {user_id}: {str(e)}")


@dp.message()
async def unknown_message(message: Message):
    logger.warning(f"Пользователь {message.from_user.id} отправил неизвестное сообщение: {message.text}")
    await message.answer(
        "❗ Я не понимаю это сообщение. Выберите режим обработки, нажав на кнопку!",
        reply_markup=get_main_keyboard()
    )


# Загрузка моделей
def load_models():
    logger.info("Начало загрузки моделей...")
    global paraphraser_model, paraphraser_tokenizer, headliner_model, headliner_tokenizer

    try:
        # Модель для перефразирования
        MODEL_PATH = "cointegrated/rut5-base-paraphraser"
        paraphraser_tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, legacy=True)
        paraphraser_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

        # Модель для генерации заголовков
        headliner_tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/rut5_base_sum_gazeta", legacy=True)
        headliner_model = AutoModelForSeq2SeqLM.from_pretrained("IlyaGusev/rut5_base_sum_gazeta").to(device)

        # Загрузка NLTK данных
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

        logger.info("Все модели успешно загружены")
    except Exception as e:
        logger.error(f"Ошибка при загрузке моделей: {str(e)}")
        raise


# Запуск бота
async def main():
    logger.info("Запуск бота...")
    try:
        load_models()
        logger.info("Бот готов к работе")
        await dp.start_polling(bot)
    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске бота: {str(e)}")
    finally:
        logger.info("Бот остановлен")


if __name__ == "__main__":
    asyncio.run(main())