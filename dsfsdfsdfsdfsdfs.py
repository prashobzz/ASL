from googletrans import Translator
translator = Translator()
sentence="cat"
translate_word = translator.translate(sentence, dest="ml")
print(f'Translation: {translate_word.text}')