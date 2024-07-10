from flask import Flask, render_template, request, redirect, url_for , session
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from gazee.gaze import gazee , stop_event
import threading
from keras.models import load_model
import numpy as np 
from PIL import Image
from werkzeug.serving import run_simple

# from tasks import flask_app, long_running_task

import pickle
import time
import multiprocessing
import cv2
import matplotlib.pyplot as plt
import datetime
import fastai 
from fastai.vision import *
import os
# from gaze_tracking import GazeTracking
import seaborn as sns
from gazee.gaze_tracking import GazeTracking


start_gazee = False
sns.set(style="ticks", context="talk")
# plt.style.use("dark_background")
# plt.figure(figsize=(10, 10), dpi = 150/10)
# 
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = "fafsdfs"




# Load the dataset from the CSV file
df = pd.read_csv('adhd_questionnaire.csv')

# Drop rows with missing target values (NaN in 'ADHD' column)
df.dropna(subset=['ADHD'], inplace=True)

# Convert 'ADHD' column to binary labels (0 for 'No' and 1 for 'Yes')
df['ADHD'] = df['ADHD'].map({'No': 0, 'Yes': 1})

# Separate features (X) and target variable (y)
X = df.drop(columns=['ADHD'])
y = df['ADHD']

# Train decision tree model
model = DecisionTreeClassifier()
model.fit(X, y)

model2 = load_model('keras_Model.h5')

def thread_test():
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    # plt.figure(figsize=(10, 10), dpi = 150/10)
    global gaze
    global start_gazee
    
    print(start_gazee)
    while not start_gazee:

        pass 
    while start_gazee:
        # We get a new frame from the webcam
        _, frame = webcam.read()

        print("here")
   
        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)
    
        frame = gaze.annotated_frame()
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        if(left_pupil == (0, 0) or right_pupil == (0, 0)):
          pass
        else:
          plt.plot(left_pupil, right_pupil)
        # cv2.imshow("Demo", frame)
    plt.savefig('2.png', transparent = False)
    print("gaze ended")

thread2 = None




# Define a route to handle form submission

# thread.daemon = True

@app.route('/submit', methods=['POST'])
def submit():
    # Get user input from the form
    choices = []
    answers = []
    for i in range(1, 14):  # Adjust the range based on the number of questions
        choice = request.form.get(f'q{i}')
        choices.append(choice)

    point_mapping = {
        "never": 1,
        "rarely": 2,
        "sometimes": 3,
        "often": 4,
    }

    for choice in choices:
        answers.append(point_mapping[choice])

    # Make prediction using the trained model
    prediction = model.predict([answers])[0]

    # Convert prediction to human-readable format
    result = True if prediction == 1 else False

    print(result)
    session['result1'] = result

    # Redirect to the intermediate page with the result
    return redirect(url_for('intermediate', result=result))

# Define a route to handle intermediate page


@app.route('/intermediate')
def intermediate():
    message = "You have completed the questionnaire. Click below to start the test."
    return render_template('inter.html', message=message)

# Define a route to render the HTML form

@app.route("/start_gaze" , methods = ["GET", "POST"])
def start_gaze():
    
    # thread = threading.Thread( target=gazee , args=(stop_event,))
    # thread.start()

    global start_gazee
    start_gazee = True
    global thread2
    thread2 = threading.Thread( target=thread_test)
    thread2.start()
    
    return ( '' , 200)

@app.route("/stop_gaze", methods = ["GET", "POST"])
def stop_gaze():
    # stop_event.set()
    
    global start_gazee
    start_gazee =  False 
    thread2.join()
    print("reached here")
    

    image_path = '2.png'
    image = Image.open(image_path)
    image = image.resize((150, 150))
    image_array = np.array(image)


    if image_array.shape[-1] == 4:  # to remove alpha channel
        image_array = image_array[:, :, :3]

    image_array = np.expand_dims(image_array, axis=0)

    print(image_array.shape)
    print(image_array.dtype)

    result = model2.predict(image_array)

    if result[0][0] == 1:
        prediction = True
    else:
        prediction = False  

    print(prediction)
    
    session['result2'] = prediction
    return ('', 200)
   
@app.route("/result")
def result():
    result1 = session.get("result1")
    result2 =  session.get("result2")

    outcome = ""

    if result1 and result2:
        outcome = "most likely"
    elif result1 or result2:
        outcome = "somewhat likely"
    else:
        outcome = "not likely"

    return render_template("result.html", result = outcome)


@app.route('/')
def index():
    return render_template('quest.html')


if __name__ == '__main__':
    # thread2.start()
    # process.start()
    app.run(debug=True , threaded=True)
    # run_simple('localhost', 5000, app )
