"""Linear regression on 170 years of Lake Mendota (Madison, WI) ice-cover records.

Fits days-of-ice ~ year two ways (closed-form normal equation and batch gradient
descent on min-max normalized years), then extrapolates the trend.

Usage: python ice_regression.py ice_data.csv <learning_rate> <iterations>
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename):
    data = pd.read_csv(filename)
    years = data.iloc[:, 0].values
    frozen_days = data.iloc[:, 1].values
    return years, frozen_days


def perform_normalization(x):
    min = np.min(x)
    max = np.max(x)
    norm_x = (x - min) / (max - min)
    return norm_x, min, max


def compute_optimal_weights(X_normalized, y):
    X_transpose_X = np.dot(X_normalized.T, X_normalized)
    X_transpose_y = np.dot(X_normalized.T, y)
    return np.linalg.inv(X_transpose_X).dot(X_transpose_y)


def run_gradient_descent(X_normalized, y, learning_rate, iterations):
    n = len(y)
    weights = np.array([0.0, 0.0])
    loss_values = []
    print("Gradient-descent weights (every 10 iterations):")
    for t in range(iterations):
        predictions = np.dot(X_normalized, weights)
        gradient = (1 / n) * np.dot(X_normalized.T, (predictions - y))
        if t % 10 == 0:
            print(weights)
        weights -= learning_rate * gradient
        current_loss = np.sum((predictions - y) ** 2) / (2 * n)
        loss_values.append(current_loss)

    return weights, loss_values

def main():
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])

    years, ice_days = load_data(filename)
    # Raw data plot
    plt.figure()
    plt.plot(years,ice_days, linestyle='-', color='blue')
    plt.xlabel("Year")
    plt.ylabel("Number of Frozen Days")
    plt.title("Lake Mendota Number Of Frozen Days")
    plt.xticks(years, labels=[str(year) for year in years])
    plt.savefig("data_plot.jpg")
    plt.close()

    # Min-max normalize years and add a bias column
    normalized_years, min_year, max_year = perform_normalization(years)
    X_normalized = np.column_stack((normalized_years, np.ones_like(normalized_years)))
    print("Normalized design matrix:")
    print(X_normalized)
    # Closed-form (normal equation) solution
    optimal_weights = compute_optimal_weights(X_normalized, ice_days)
    print("Closed-form weights [w, b]:")
    print(optimal_weights)

    # Gradient descent
    final_weights, loss_history = run_gradient_descent(X_normalized, ice_days, learning_rate, iterations)
    print("Learning rate:", learning_rate)
    print("Iterations:", iterations)

    plt.figure()
    plt.plot(range(iterations), loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.jpg")
    plt.close()
    # Predict the number of ice days in 2023
    predict_year = 2023
    normalized_predict = (predict_year - min_year) / (max_year - min_year)
    prediction = optimal_weights[0] * normalized_predict + optimal_weights[1]
    print("Predicted ice days in 2023:", prediction)

    # Interpret the slope
    weight_sign = ">" if optimal_weights[0] > 0 else "<" if optimal_weights[0] < 0 else "="
    print("Sign of slope w:", weight_sign)


    if optimal_weights[0] != 0:
        no_freeze_year = min_year - (optimal_weights[1] * (max_year - min_year)) / optimal_weights[0]
    else:
        no_freeze_year = float('inf')
    print("Predicted first ice-free year:", no_freeze_year)



if __name__ == '__main__':
    main()