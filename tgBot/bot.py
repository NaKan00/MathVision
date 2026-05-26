import asyncio
import logging
import subprocess
from pathlib import Path

from aiogram import Bot, Dispatcher, types, html, F
from aiogram.filters.command import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from config_reader import config


logging.basicConfig(level=logging.INFO)

dp = Dispatcher()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

CKPT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "im2latex_full_plus_school_ft"
    / "current_bot_best.pt"
)

TOKENIZER_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "im2latex_full_plus_school_ft"
    / "tokenizer.json"
)


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        f"Hello, {html.bold(message.from_user.full_name)}!\n"
        "Send me a photo with a formula, and I will try to recognize it."
    )


@dp.message(F.photo)
async def handle_photo(message: types.Message, bot: Bot):
    photo = message.photo[-1]

    tg_file = await bot.get_file(photo.file_id)

    local_path = IMAGES_DIR / f"{photo.file_id}.jpg"

    await bot.download_file(
        tg_file.file_path,
        destination=local_path,
    )

    await message.answer("Photo received. Recognizing formula...")

    result = subprocess.run(
        [
            "python",
            "run_infer.py",
            "--ckpt",
            str(CKPT_PATH),
            "--tokenizer",
            str(TOKENIZER_PATH),
            "--image",
            str(local_path),
            "--decode",
            "beam",
            "--beam_size",
            "5",
        ],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        await message.answer(
            "Recognition failed:\n"
            f"<pre>{html.quote(result.stderr[-3000:])}</pre>"
        )
        return

    stdout = result.stdout.strip()
    latex = stdout

    for line in stdout.splitlines():
        if line.startswith("pred:"):
            latex = line.replace("pred:", "", 1).strip()
            break

    await message.answer(
        "Recognized LaTeX:\n"
        f"<code>{html.quote(latex)}</code>"
    )


@dp.message()
async def fallback(message: types.Message):
    await message.answer(
        "Please send a photo with a mathematical formula."
    )


async def main():
    bot = Bot(
        token=config.bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    await bot.delete_webhook(drop_pending_updates=True)

    me = await bot.get_me()
    print(f"BOT STARTED: @{me.username} id={me.id}")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())