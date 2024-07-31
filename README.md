# Diabetes Prediction using KNN Algorithm

This project demonstrates the use of the K-Nearest Neighbors (KNN) algorithm to predict diabetes based on medical diagnostic measurements.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Diabetes is a chronic disease characterized by high levels of sugar in the blood. Early detection and management are crucial to prevent complications. This project uses the KNN algorithm to build a predictive model for diabetes diagnosis based on features such as glucose levels, blood pressure, and body mass index (BMI).

## Dataset

The dataset used in this project is the Pima Indians Diabetes Database, which is available on Kaggle. It contains 768 records with 9 features:

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age
9. Outcome (0 for non-diabetic, 1 for diabetic)

## Libraries Used

The following libraries are used in this project:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib`: For data visualization.
- `scikit-learn`: For machine learning algorithms and tools.

## Installation

To run this project, you need to have Python installed. Follow the steps below to set up the environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/diabetes-prediction-knn.git
    ```
2. Navigate to the project directory:
    ```bash
    cd diabetes-prediction-knn
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Requirements

The project requires the following Python libraries, which can be installed using the provided `requirements.txt` file:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

## Usage

1. Make sure you have the `diabetes.csv` file in the project directory. If not, download it from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database) and place it in the directory.
2. Run the script:
    ```bash
    python DiabetesPrediction.py
    ```
3. The script will:
    - Load and display the dataset.
    - Split the dataset into training and test sets.
    - Train a KNN model with varying numbers of neighbors (from 1 to 10).
    - Plot the training and test accuracy for different numbers of neighbors.
    - Display the accuracy of the KNN classifier with the optimal number of neighbors.

## Results

The script will output the training and test accuracy for different numbers of neighbors and display a plot comparing these accuracies. It will also show the accuracy of the KNN classifier on the training and test sets with the optimal number of neighbors.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
