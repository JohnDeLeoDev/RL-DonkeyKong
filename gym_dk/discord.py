from discord_webhook import DiscordWebhook

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1229458443233132645/AgJnu4XQR1_IFzRCGm_4KgKxnaP_JanJk4EvyN55CaOxaPjZyxh5j7gPgOt6C1cwnUZx"


def send_message(message) -> None:
    webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL, content=message)
    response = webhook.execute()
