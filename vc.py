import pyttsx3

converter = pyttsx3.init()
converter.setProperty('rate', 150)
converter.setProperty('volume', 0.9)
converter.say("hai")

converter.runAndWait()