import requests
import telebot

TOKEN = ''
API_URL = 'http://localhost:8000/predict'

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def welcome(message):
    bot.send_message(message.chat.id, "Привет! Пришли число — я верну его квадрат")

@bot.message_handler(func=lambda msg: True)
def handle_input(message):
    try:
        x = float(message.text.strip())
        response = requests.post(API_URL, json={"x": x})
        result = response.json().get("predicted", None)
        if result is not None:
            bot.send_message(message.chat.id, f"{x}² = {result:.4f}")
        else:
            bot.send_message(message.chat.id, "Ошибка данных")
    except ValueError:
        bot.send_message(message.chat.id, "Нужно ввести число.")
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка: {e}")

bot.polling()
