import discord
from libs.chat import Chat

heart_emoji = discord.PartialEmoji(name="❤️")
sparkling_heart_emoji = discord.PartialEmoji(name="💖")

note_emoji = discord.PartialEmoji(name="📝")
memory_emoji = discord.PartialEmoji(name="🧠")

class Bot(discord.Client):
    def __init__(self, chat: Chat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat = chat
        self.message_channel = None

    async def on_message(self, message: discord.Message):
        # create a direct message channel on first message
        if isinstance(message.channel, discord.DMChannel) and self.message_channel == None:
            self.message_channel = await message.author.create_dm()

        if message.author == self.user:
            return

        # chat_response = await self.chat.message(message.content)
        chat_response = await self.chat.test(input=message.content)
        await message.reply(chat_response, mention_author=True)

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        msg = None
        if self.message_channel:
            msg = await self.message_channel.fetch_message(payload.message_id)
        if msg and payload.emoji == heart_emoji:
            await msg.add_reaction(sparkling_heart_emoji)

