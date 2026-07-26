import os
import torch
import discord
from discord.ext import commands
from transformers import AutoTokenizer, AutoModelForCausalLM
import nest_asyncio

nest_asyncio.apply()

MODEL_DIR = "play451/r3tard"
MAX_NEW_TOKENS = 60
MIN_NEW_TOKENS = 3
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 60
REPETITION_PENALTY = 1.3
NO_REPEAT_NGRAM_SIZE = 3

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, token=HF_TOKEN).to(device)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

print(f"Model loaded on {device}")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.command(name="chat")
async def chat(ctx, *, prompt: str = ""):
    if not prompt:
        await ctx.send("Give me something to respond to, e.g. `!chat hey what's up`")
        return

    async with ctx.typing():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][prompt_len:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not generated:
            generated = "..."

    await ctx.send(generated[:2000])

if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN environment variable is not set.")
    bot.run(DISCORD_BOT_TOKEN)
