# Predicting-ADHD-through-Eye-Gaze:A-Machine-Learning-Approach

This project aims to detect Attention-Deficit/Hyperactivity Disorder (ADHD) using a combination of a questionnaire-based decision tree algorithm and an eye gaze tracking mechanism. The eye gaze data is processed using OpenCV and plotted using Matplotlib, then analyzed with a Convolutional Neural Network (CNN) model. The final result is based on the combined outputs of both the questionnaire and eye gaze tracking.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methods](#methods)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

ADHD is a prevalent mental health condition affecting various age groups. Accurate and timely detection of ADHD can lead to better management and treatment. This project combines a decision tree algorithm for a questionnaire and a CNN model for eye movement tracking to provide a comprehensive approach to ADHD detection.

## Dataset

The dataset used in this project includes:
- Responses from a questionnaire with various ADHD-related questions.
- Eye movement data captured using a webcam and processed for gaze tracking.

## Methods

### Questionnaire-Based Detection (Decision Tree)
- Implemented a decision tree algorithm to analyze questionnaire responses.
- The decision tree classifies responses into ADHD and non-ADHD categories based on predefined patterns.

### Eye Movement Tracking
- Used OpenCV to capture eye movement data through a webcam.
- Plotted eye coordinates using Matplotlib for visualization.
- Processed the eye movement data as input for the CNN model.

### Convolutional Neural Network (CNN)
- Developed a CNN model to analyze the eye movement data.
- Trained the CNN model on the processed eye movement data to detect ADHD.

### Combined Results
- The final result is determined based on the outputs of both the questionnaire and eye tracking:
  - If both methods indicate ADHD, the result is "More Likely".
  - If either method indicates ADHD, the result is "Somewhat Likely".
  - If neither method indicates ADHD, the result is "Not Likely".

## Results

The combination of the decision tree algorithm for the questionnaire and the CNN model for eye movement tracking provides a comprehensive and accurate method for detecting ADHD. The final result is displayed on the output page based on the combined outputs of both methods.

## Conclusion

This project demonstrates an innovative approach to ADHD detection by integrating machine learning techniques with traditional methods. The combination of a questionnaire-based decision tree algorithm and a CNN model for eye movement tracking offers a reliable and accurate diagnostic tool.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Minha-ak/Predicting-ADHD-through-Eye-Gaze-A-Machine-Learning-Approach.git
    cd adhd-detection
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Prepare the dataset (ensure it's in the appropriate format and path).

2. Run the Flask application:
  sh
  Copy code
  python pyflask.py
  This will direct you to the questionnaire page.

3. Fill in the questionnaire, click 'end'.
   
4. Now click 'begin test' a passage will appear for you to readwhile your eye movements are tracked.

5. Once you have finished reading, click "End Test" to get the final output.

6. The result will be displayed as "More Likely", "Somewhat Likely", or "Not Likely" based on the combined outputs of the questionnaire and eye tracking.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

