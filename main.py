import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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

# Function to get user input and make prediction


def predict_adhd():
    # Get user input for each question
    l = []
    answers = []
    for i in range(5):
        print("Hi")
        a = input("enter the value:")
        l.append(a)

    # for i in range(1, 14):
    #    print(i)  # Adjust the range based on the number of questions
    #    answer = int(input(
    #        "Question: How often do you experience this? (Enter a number from 1 to 5): "))
    #    answers.append(answer)

    # Make prediction using the trained model
    answers = [1, 1, 1, 3, 4, 5, 1, 3, 1, 1, 5, 1, 1]
    prediction = model.predict([answers])[0]

    # Convert prediction to human-readable format
    result = "Yes" if prediction == 1 else "No"

    # Print prediction result
    print(
        f"Based on your responses, it is likely that you have ADHD: {result}")


# Call the function to predict ADHD based on user input
predict_adhd()
