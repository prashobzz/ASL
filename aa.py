import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import speech_recognition as sr

# Global variables
from adodbapi.schema_table import names
from mediapipe.python.solutions import hands

curr_char = None
prev_char = None
word = ""
sentence = ""
threshold = 0.95
cap = None

# Initialize SpeechRecognition recognizer
recognizer = sr.Recognizer()


# Function to recognize voice commands
def recognize_voice():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        # Recognize speech using Google Speech Recognition
        recognized_text = recognizer.recognize_google(audio)
        print("Recognized:", recognized_text)
        return recognized_text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return None


# Function to process voice commands
def process_voice_command():
    global sentence
    voice_command = recognize_voice()
    if voice_command:
        sentence += voice_command + " "
        sent_entrybox.delete(0, 'end')
        sent_entrybox.insert('end', sentence)


# Function to process video frames
def frame_video_stream(names, curr_char, prev_char, word, sentence, vid_label, hands, th_entrybox, cc_entrybox,
                       ow_entrybox, cw_entrybox, sent_entrybox):
    ret, frame = cap.read()
    if not ret:
        return

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for point in hand_landmarks.landmark:
                # Process hand landmarks
                pass

    # Display the frame
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    vid_label.imgtk = imgtk
    vid_label.configure(image=imgtk)
    vid_label.after(10, frame_video_stream)


# Function to exit the application
def exit_app(gui, cap):
    if cap is not None:
        cap.release()
    gui.root.destroy()


# Function to start the GUI
def start_gui(title, size):
    gui = GUI(title, size)
    vid_label = tk.Label(gui.root)
    vid_label.pack()
    return gui, vid_label


# GUI class
class GUI:
    def __init__(self, title, size):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(size)

    def create_labels(self, num, labels, direction, x, y, y_spacing=0.06, create_entrybox_per_label=False):
        entryboxes = {}
        for i in range(num):
            label = labels[i]
            label_name = label.lower().replace(" ", "_") + "_entrybox"
            if create_entrybox_per_label:
                entrybox = tk.Entry(self.root)
                entrybox.grid(column=x + 1, row=y + i, pady=2)
                entryboxes[label_name] = entrybox
            tk.Label(self.root, text=label).grid(column=x, row=y + i, sticky=direction, pady=2)
        return labels, entryboxes

    def create_buttons(self, num, button_texts, direction, x, y, command=None):
        buttons = []
        for i in range(num):
            button = tk.Button(self.root, text=button_texts[i], command=command)
            button.grid(column=x, row=y + i, pady=5)
            buttons.append(button)
        return buttons


# Initialize the GUI
title = "Sign Language Recognition GUI"
size = "1100x1100"
gui, vid_label = start_gui(title, size)

# Open the webcam
cap = cv2.VideoCapture(0)

# Start the main loop
while True:
    # Process video frames
    frame_video_stream(names, curr_char, prev_char, word, sentence, vid_label,
                       hands, th_entrybox, cc_entrybox, ow_entrybox, cw_entrybox, sent_entrybox)

    # Process voice commands
    process_voice_command()

    gui.root.update()
