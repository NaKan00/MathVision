import asyncio
import logging
from pathlib import Path

from aiogram import Bot, Dispatcher, types, html, F
from aiogram.filters.command import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from config_reader import config

logging.basicConfig(level=logging.INFO)

dp = Dispatcher()

# Путь к папке tgBot/images
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        f"Hello, {html.bold(message.from_user.full_name)}!\n"
        "I'am Bot, which can recognize your text!\n"
        "Send me photo and I will try"
    )


@dp.message(F.photo)
async def handle_photo(message: types.Message, bot: Bot):
    photo = message.photo[-1]
    tg_file = await bot.get_file(photo.file_id)

    local_path = IMAGES_DIR / f"{photo.file_id}.jpg"

    await bot.download_file(tg_file.file_path, destination=local_path)

    await message.answer(
        f"Photo received and saved!\n"
        "Now I will learn to recognize it..."
    )

@dp.message(~F.photo)
async def not_photo(message: types.Message):
    await message.answer(f"It doesn't look like the photo...\n"
                        "Please send a photo")


async def main():
    bot = Bot(
        token=config.bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
    