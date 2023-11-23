import discord

heart_emoji = discord.PartialEmoji(name="❤️")
sparkling_heart_emoji = discord.PartialEmoji(name="💖")

note_emoji = discord.PartialEmoji(name="📝")
memory_emoji = discord.PartialEmoji(name="🧠")

class Bot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_channel = None

    async def on_message(self, message: discord.Message):
        # create a direct message channel on first message
        if isinstance(message.channel, discord.DMChannel) and self.message_channel is None:
            self.message_channel = await message.author.create_dm()

        if message.author == self.user:
            return

        if message.content.startswith('$hello'):
            await message.reply("Hey!", mention_author=True)

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        msg = None
        if self.message_channel:
            msg = await self.message_channel.fetch_message(payload.message_id)
        if payload.emoji == heart_emoji and msg:
            await msg.add_reaction(sparkling_heart_emoji)

