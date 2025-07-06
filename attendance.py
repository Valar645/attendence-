import tkinter as tk
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
from tkinter import messagebox
import cv2
print(cv2.__version__)
print(dir(cv2))
import cv2

# Check if OpenCV face module is available
if hasattr(cv2, 'face'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
else:
    print("Error: OpenCV face module not found! Make sure you installed 'opencv-contrib-python'.")

def TakeImages():
    Id = txt.get().strip()
    name = txt2.get().strip()
    
    if not Id or not name:
        message.configure(text="Please enter both ID and Name")
        return

    if not is_number(Id) or not name.isalpha():
        message.configure(text="Enter a numeric ID and alphabetical Name")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    sampleNum = 0
    training_path = "TrainingImage"
    student_details_path = "StudentDetails"
    student_file = os.path.join(student_details_path, "StudentDetails.csv")

    # Ensure the directories exist
    os.makedirs(training_path, exist_ok=True)
    os.makedirs(student_details_path, exist_ok=True)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite(os.path.join(training_path, f"{name}.{Id}.{sampleNum}.jpg"), gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('frame', img)

        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 60:
            break

    cam.release()
    cv2.destroyAllWindows()

    # Create the CSV file if it does not exist
    if not os.path.exists(student_file):
        with open(student_file, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Id", "Name"])  # Add header row

    # Append new student details
    with open(student_file, 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([Id, name])

    message.configure(text=f"Images Saved for ID: {Id}, Name: {name}")
    

    def TrackImages():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("Trainer.yml")

    # Define faceCascade before using it
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    df = pd.read_csv("StudentDetails/StudentDetails.csv")

    cam = cv2.VideoCapture(0)
    attendance = pd.DataFrame(columns=['Id', 'Name', 'Date', 'Time'])

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)  # Make sure faceCascade is used here

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                name = df.loc[df['Id'] == Id, 'Name'].values[0]  # Fix indexing issue

                attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                cv2.putText(im, f"{Id}-{name}", (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Attendance', im)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # Show attendance details in the message2 label
    if not attendance.empty:
        attendance_str = attendance.to_string(index=False)
        message2.configure(text=attendance_str)
    else:
        message2.configure(text="No attendance recorded.")

    message.configure(text="Attendance Taken Successfully!")











# Initialize the main window
window = tk.Tk()
window.title("Attendance System")
window.geometry("1500x1500")
window.configure(background='#CADCFC')

# UI elements
message = tk.Label(window, text="ATTENDANCE MANAGEMENT SYSTEM", bg="#CADCFC", fg="#00246B",
                    width=40, height=1, font=('Times New Roman', 35, 'bold underline'))
message.place(x=200, y=20)

tk.Label(window, text="College ID", width=20, height=2, fg="#00246B", bg="#CADCFC",
          font=('Times New Roman', 25, 'bold')).place(x=125, y=180)

txt = tk.Entry(window, width=30, bg="#CADCFC", fg="blue", font=('Times New Roman', 15, 'bold'))
txt.place(x=175, y=280)

tk.Label(window, text="Candidate Name", width=20, height=2, fg="#00246B", bg="#CADCFC",
          font=('Times New Roman', 25, 'bold')).place(x=525, y=180)

txt2 = tk.Entry(window, width=30, bg="#CADCFC", fg="blue", font=('Times New Roman', 15, 'bold'))
txt2.place(x=575, y=280)

tk.Label(window, text="Notification", width=20, height=2, fg="#00246B", bg="#CADCFC",
          font=('Times New Roman', 25, 'bold')).place(x=985, y=180)

message = tk.Label(window, text="", bg="#CADCFC", fg="blue", width=30, height=1,
                    font=('Times New Roman', 15, 'bold'))
message.place(x=1000, y=280)

tk.Label(window, text="Attendance Details", width=20, height=2, fg="#00246B", bg="#CADCFC",
          font=('Times New Roman', 30, 'bold')).place(x=120, y=570)

message2 = tk.Label(window, text="", fg="red", bg="#CADCFC", width=60, height=4,
                     font=('times', 15, 'bold'))
message2.place(x=700, y=570)

# Functions
def clear1():
    txt.delete(0, 'end')
    message.configure(text="")

def clear2():
    txt2.delete(0, 'end')
    message.configure(text="")

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def TakeImages():
    Id = txt.get().strip()
    name = txt2.get().strip()
    
    if not Id or not name:
        message.configure(text="Please enter both ID and Name")
        return

    if not is_number(Id) or not name.isalpha():
        message.configure(text="Enter a numeric ID and alphabetical Name")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    sampleNum = 0
    training_path = "TrainingImage"

    os.makedirs(training_path, exist_ok=True)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite(os.path.join(training_path, f"{name}.{Id}.{sampleNum}.jpg"), gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('frame', img)

        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 60:
            break

    cam.release()
    cv2.destroyAllWindows()

    with open('StudentDetails/StudentDetails.csv', 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([Id, name])

    message.configure(text=f"Images Saved for ID: {Id}, Name: {name}")

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Ids = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Ids))
    recognizer.save("Trainer.yml")

    message.configure(text="Model Trained Successfully!")
    messagebox.showinfo("Success", "Model trained successfully!")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces, Ids = [], []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces.append(imageNp)
        Ids.append(Id)

    return faces, Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    df = pd.read_csv("StudentDetails/StudentDetails.csv")

    cam = cv2.VideoCapture(0)
    attendance = pd.DataFrame(columns=['Id', 'Name', 'Date', 'Time'])

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                name = str(df.loc[df['Id'] == Id]['Name'].values[0])

                attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                cv2.putText(im, f"{Id}-{name}", (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Attendance', im)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    message.configure(text="Attendance Taken Successfully!")

# Buttons
tk.Button(window, text="IMAGE CAPTURE", command=TakeImages, font=('Times New Roman', 15, 'bold')).place(x=245, y=425)
tk.Button(window, text="MODEL TRAINING", command=TrainImages, font=('Times New Roman', 15, 'bold')).place(x=645, y=425)

window.mainloop()
