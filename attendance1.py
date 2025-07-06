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




def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Load student details
    df = pd.read_csv(student_file)
    
    cam = cv2.VideoCapture(0)
    attendance = pd.DataFrame(columns=['Id', 'Name', 'Date', 'Time'])

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            # ✅ Check if ID exists before accessing the name
            if Id in df['Id'].values:
                name = df.loc[df['Id'] == Id, 'Name'].values[0]
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                cv2.putText(im, f"{Id}-{name}", (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                name = "Unknown"
                cv2.putText(im, "Unknown", (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Attendance', im)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    message.configure(text="Attendance Taken Successfully!")





# Initialize Face Recognizer
if hasattr(cv2, 'face'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
else:
    print("Error: OpenCV face module not found! Make sure you installed 'opencv-contrib-python'.")

# Ensure directories exist
training_path = "TrainingImage"
student_details_path = "StudentDetails"
student_file = os.path.join(student_details_path, "StudentDetails.csv")
os.makedirs(training_path, exist_ok=True)
os.makedirs(student_details_path, exist_ok=True)

# Create student details CSV file if not exists
def initialize_csv():
    if not os.path.exists(student_file):
        with open(student_file, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Id", "Name"])

initialize_csv()

# Function to clear input fields
def clear1():
    txt.delete(0, 'end')
    message.configure(text="")

def clear2():
    txt2.delete(0, 'end')
    message.configure(text="")

# Function to check if input is a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Function to capture images and store student details
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

    # Update Student Details (overwrite file)
    with open(student_file, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["Id", "Name"])
        writer.writerow([Id, name])

    message.configure(text=f"Images Saved for ID: {Id}, Name: {name}")

# Function to train images
def TrainImages():
    faces, Ids = getImagesAndLabels(training_path)
    recognizer.train(faces, np.array(Ids))
    recognizer.save("Trainer.yml")

    message.configure(text="Model Trained Successfully!")
    messagebox.showinfo("Success", "Model trained successfully!")

# Function to get images and labels
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

# Function to track images and take attendance
def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    df = pd.read_csv(student_file)

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

# Initialize GUI
window = tk.Tk()
window.title("Attendance System")
window.geometry("1500x1500")
window.configure(background='#CADCFC')

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

# Buttons
tk.Button(window, text="IMAGE CAPTURE", command=TakeImages, font=('Times New Roman', 15, 'bold')).place(x=245, y=425)
tk.Button(window, text="MODEL TRAINING", command=TrainImages, font=('Times New Roman', 15, 'bold')).place(x=645, y=425)
tk.Button(window, text="TRACK ATTENDANCE", command=TrackImages, font=('Times New Roman', 15, 'bold')).place(x=1045, y=425)

window.mainloop()
