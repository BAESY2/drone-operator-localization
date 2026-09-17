"""Telegram bot — defensive drone operator localization assistant."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TELEGRAM_BOT_TOKEN  # noqa: E402

# Local handlers.py (avoid clashing with python-telegram-bot package name)
import importlib.util

_handlers_path = Path(__file__).resolve().parent / "handlers.py"
_spec = importlib.util.spec_from_file_location("local_tg_handlers", _handlers_path)
_handlers_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_handlers_mod)
format_result_message = _handlers_mod.format_result_message

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BEARING_INPUT, LOCATION_CONFIRMATION = range(2)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (
        "Drone Operator Localization (defensive use only)\n\n"
        "1) Send bearing degrees (0-359)\n"
        "2) Share your GPS location\n\n"
        "Users assume full legal liability."
    )
    keyboard = [[InlineKeyboardButton("Start", callback_data="start_input")]]
    if update.message:
        await update.message.reply_text(
            text, reply_markup=InlineKeyboardMarkup(keyboard)
        )
    return BEARING_INPUT


async def start_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Enter bearing degrees (0-359):")
    return BEARING_INPUT


async def receive_bearing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        bearing = int(update.message.text.strip())
    except (TypeError, ValueError):
        await update.message.reply_text("Please send a number between 0 and 359.")
        return BEARING_INPUT

    if not (0 <= bearing <= 359):
        await update.message.reply_text("Bearing must be 0-359.")
        return BEARING_INPUT

    context.user_data["bearing"] = bearing
    await update.message.reply_text(
        f"Bearing {bearing}° saved.\nNow share your location (paperclip → Location)."
    )
    return LOCATION_CONFIRMATION


async def receive_location(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    location = update.message.location
    bearing = context.user_data.get("bearing")
    if bearing is None:
        await update.message.reply_text("Send /start and enter bearing first.")
        return ConversationHandler.END

    await update.message.reply_chat_action(ChatAction.TYPING)
    try:
        from src.predict_api import predict

        result = predict(
            latitude=location.latitude,
            longitude=location.longitude,
            bearing_degrees=bearing,
        )
        await update.message.reply_text(format_result_message(result))
    except Exception as exc:  # noqa: BLE001
        logger.exception("predict failed")
        await update.message.reply_text(f"Failed: {exc}")

    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


def main() -> None:
    token = TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "your_token_here":
        raise ValueError("Set TELEGRAM_BOT_TOKEN in .env")

    app = Application.builder().token(token).build()
    conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CallbackQueryHandler(start_callback, pattern="^start_input$"),
        ],
        states={
            BEARING_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_bearing),
                CallbackQueryHandler(start_callback, pattern="^start_input$"),
            ],
            LOCATION_CONFIRMATION: [
                MessageHandler(filters.LOCATION, receive_location)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(conv)
    logger.info("Bot starting (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
