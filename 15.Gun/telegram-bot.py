import logging
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import os
import io

import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC

#from google.cloud import speech
#os.environ['GOOGLE_APPLICATION_CREDENTIALS']= 'pure-coda-429717-e3-81e98c066b52.json'
#client = speech.SpeechClient()

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

processor = AutoProcessor.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-turkish")
model = AutoModelForCTC.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-turkish")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(
        rf"Merhabalar {user.mention_html()}! Ben T3 örnek uygulama botuyum. Tanıştığıma memnun oldum!",
        reply_markup=ForceReply(selective=True),
    )

async def yardim(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Yardım menüsü. Kullanabileceğin komutlar şunlar:")
    await update.message.reply_text("Yardım için (Zaten buradasın (: ): /yardim")
    await update.message.reply_text("Yazdıklarını tekrar etmem için: /yanki")
    await update.message.reply_text("T3 Hakında: /hakkinda")

async def yanki(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Söylediklerin yankı yaptı:",update.message.text)
    
async def hakkinda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Sayfamızı ziyaret edebilirsin: https://www.t3vakfi.org/")
    
async def sesli_komut(update, context):
    await update.message.reply_text("Sesli komut için teşekkür ederim (:")
    fh = await context.bot.get_file(update.message.voice.file_id)
    await fh.download_to_drive("file.ogg")

    voice, sample_rate = torchaudio.load("file.ogg")
    voice_resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(voice)
    with torch.no_grad():
        logits = model(voice_resample).logits

    #argmax'a bak
    output_ids = torch.argmax(logits, dim=-1)
    command = processor.batch_decode(output_ids)[0]

    print(f"command : {command}")
    if (command == "yardım"):
        await yardim(update, context)
    elif (command == "yankı"):
        await yanki(update, context)
    elif (command == "hakkında"):
        await hakkinda(update, context)
    else:
        await update.message.reply_text("Maalesef komutunuzu anlayamadım ):")
    
    #with io.open("file.ogg", "rb") as audio_file:
    #    content = audio_file.read()
    #    audio = speech.RecognitionAudio(content=content)
    #
    #config = speech.RecognitionConfig(
    #sample_rate_hertz=8000,
    #encoding=speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
    #enable_automatic_punctuation=True,
    #audio_channel_count=1,
    #language_code="tr-TR")
    #
    #response = client.recognize(request={"config": config, "audio": audio})

    #for result in response.results:
    #    if (result.alternatives[0].transcript == "yardım"):
    #        yardim(update, context)
    #    elif (result.alternatives[0].transcript == "yankı"):
    #        yanki(update, context)
    #    elif (result.alternatives[0].transcript == "hakkında"):
    #        hakkinda(update, context)
    #    else:
    #        await update.message.reply_text("Maalesef komutunuzu anlayamadım ):")
    
if __name__ == "__main__":
    application = Application.builder().token("<token>").build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("yardim", yardim))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, yanki))
    application.add_handler(CommandHandler("hakkinda", hakkinda))
    application.add_handler(MessageHandler(filters.VOICE, sesli_komut))
    application.run_polling(allowed_updates=Update.ALL_TYPES)
