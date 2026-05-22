import asyncio
import io
import logging
import sys
import uuid
from pathlib import Path

# Фикс для Windows: ProactorEventLoop плохо работает с VPN-адаптерами
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from aiogram import Bot, Dispatcher, types, html, F
from aiogram.filters.command import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BufferedInputFile

# Добавляем корень проекта в sys.path, чтобы импортировать пакет im2latex
BASE_DIR = Path(__file__).resolve().parent          # tgBot/
PROJECT_ROOT = BASE_DIR.parent                      # MathVision/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_reader import config                    # noqa: E402

import matplotlib
matplotlib.use("Agg")                               # без GUI
import matplotlib.figure

import torch
from PIL import Image

from im2latex.infer import load_model, beam_decode, greedy_decode
from im2latex.data.transforms import build_image_transform
from im2latex.utils import load_tokenizer

# ---------------------------------------------------------------------------
# Настройки модели (соответствуют im2latex/config.py ветки model_v2_d384)
# ---------------------------------------------------------------------------
CKPT_DIR       = PROJECT_ROOT / "checkpoints" / "im2latex_convnext"
CKPT_PATH      = CKPT_DIR / "epoch_4.pt"
TOKENIZER_PATH = CKPT_DIR / "tokenizer.json"

IMAGE_HEIGHT   = 64
MAX_WIDTH      = 512   # увеличено в новой версии
DECODE_MODE    = "beam"
BEAM_SIZE      = 5
REPEAT_PENALTY = 1.0
LENGTH_PENALTY = 0.8
NO_REPEAT_NGRAM_SIZE = 3
MIN_LEN        = 4
MAX_LEN        = 320   # увеличено в новой версии

# Временная папка для входящих фото
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Глобальное состояние модели (загружается один раз при старте)
# ---------------------------------------------------------------------------
_model     = None
_tokenizer = None
_transform = None
_device    = None


def load_inference_pipeline():
    global _model, _tokenizer, _transform, _device

    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    logger.info("Загружаю модель на устройстве: %s", _device)
    logger.info("Чекпоинт: %s", CKPT_PATH)

    _tokenizer = load_tokenizer(str(TOKENIZER_PATH))
    _model     = load_model(str(CKPT_PATH), device=_device)
    _transform = build_image_transform(height=IMAGE_HEIGHT, max_width=MAX_WIDTH)

    logger.info("Модель успешно загружена.")


def run_inference(image_path: Path) -> str:
    img = Image.open(image_path).convert("RGB")
    x   = _transform(img).unsqueeze(0)   # (1, C, H, W)

    bos = _tokenizer.vocab.bos
    eos = _tokenizer.vocab.eos

    if DECODE_MODE == "greedy":
        ids = greedy_decode(_model, x, bos, eos, max_len=MAX_LEN)
    else:
        ids = beam_decode(
            _model, x, bos, eos,
            beam_size=BEAM_SIZE,
            max_len=MAX_LEN,
            repeat_penalty=REPEAT_PENALTY,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            min_len=MIN_LEN,
        )

    return _tokenizer.decode(ids, skip_special=True)


# ---------------------------------------------------------------------------
# Рендер LaTeX → PNG через matplotlib
# ---------------------------------------------------------------------------
def render_latex_to_png(latex: str) -> io.BytesIO | None:
    try:
        fig = matplotlib.figure.Figure(figsize=(8, 2), dpi=150)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_facecolor("white")

        ax.text(
            0.5, 0.5, f"${latex}$",
            fontsize=22,
            ha="center", va="center",
            transform=ax.transAxes,
            color="black",
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="white", dpi=150)
        buf.seek(0)
        return buf

    except Exception as exc:
        logger.warning("Не удалось отрендерить LaTeX в картинку: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Общая функция отправки результата
# ---------------------------------------------------------------------------
async def send_result(message: types.Message, latex: str):
    await message.answer(
        f"<b>LaTeX-код:</b>\n\n"
        f"<code>{latex}</code>"
    )

    png_buf = await asyncio.get_event_loop().run_in_executor(
        None, render_latex_to_png, latex
    )
    if png_buf:
        await message.answer_photo(
            BufferedInputFile(png_buf.read(), filename="formula.png"),
            caption="Как выглядит формула"
        )
    else:
        await message.answer(
            "Не удалось отрисовать формулу — "
            "некоторые команды LaTeX не поддерживаются рендерером."
        )


# ---------------------------------------------------------------------------
# Диспетчер
# ---------------------------------------------------------------------------
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        f"Привет, {html.bold(message.from_user.full_name)}!\n\n"
        "Я конвертирую фотографии с математическими формулами в <b>LaTeX-код</b> "
        "и сразу показываю, как формула выглядит.\n\n"
        "Просто отправь мне фото — и получишь LaTeX и картинку с формулой."
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(
        "<b>Как пользоваться:</b>\n"
        "1. Сфотографируй или найди изображение с формулой.\n"
        "2. Отправь его мне как фото (или файл — тоже работает).\n"
        "3. Получишь LaTeX-код и картинку с отрисованной формулой.\n\n"
        "<b>Команды:</b>\n"
        "/start — приветствие\n"
        "/help  — эта справка\n\n"
        "Лучше всего работает с чёткими, контрастными изображениями формул."
    )


@dp.message(F.photo)
async def handle_photo(message: types.Message, bot: Bot):
    await message.answer("Обрабатываю изображение, подожди секунду...")

    photo      = message.photo[-1]
    tg_file    = await bot.get_file(photo.file_id)
    local_path = IMAGES_DIR / f"{uuid.uuid4().hex}.jpg"

    await bot.download_file(tg_file.file_path, destination=str(local_path))

    try:
        latex = await asyncio.get_event_loop().run_in_executor(
            None, run_inference, local_path
        )
        if latex.strip():
            await send_result(message, latex)
        else:
            await message.answer(
                "Не удалось распознать формулу. "
                "Попробуй более чёткое изображение."
            )
    except Exception as exc:
        logger.exception("Ошибка при инференсе: %s", exc)
        await message.answer(
            "Произошла ошибка при распознавании. "
            "Попробуй ещё раз или пришли другое фото."
        )
    finally:
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass


@dp.message(F.document)
async def handle_document(message: types.Message, bot: Bot):
    doc  = message.document
    mime = doc.mime_type or ""

    if not mime.startswith("image/"):
        await message.answer(
            "Это не изображение.\n"
            "Отправь, пожалуйста, фото или изображение формулы."
        )
        return

    await message.answer("Обрабатываю изображение, подожди секунду...")

    tg_file    = await bot.get_file(doc.file_id)
    suffix     = Path(doc.file_name or "img.png").suffix or ".png"
    local_path = IMAGES_DIR / f"{uuid.uuid4().hex}{suffix}"

    await bot.download_file(tg_file.file_path, destination=str(local_path))

    try:
        latex = await asyncio.get_event_loop().run_in_executor(
            None, run_inference, local_path
        )
        if latex.strip():
            await send_result(message, latex)
        else:
            await message.answer(
                "Не удалось распознать формулу. "
                "Попробуй более чёткое изображение."
            )
    except Exception as exc:
        logger.exception("Ошибка при инференсе: %s", exc)
        await message.answer(
            "Произошла ошибка при распознавании. "
            "Попробуй ещё раз или пришли другое фото."
        )
    finally:
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass


@dp.message(~F.photo & ~F.document)
async def not_photo(message: types.Message):
    await message.answer(
        "Отправь мне фото или изображение с формулой.\n"
        "Напиши /help, если нужна помощь."
    )


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
async def main():
    load_inference_pipeline()

    bot = Bot(
        token=config.bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    logger.info("Бот запущен, начинаю polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
