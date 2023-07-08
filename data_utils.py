import matplotlib.pyplot as plt
import numpy as np

def calculate_yellow_scores(data_row, num_vals=9):

    # Get the RGB values from the data row
    rgb_vals = data_row[num_vals:]

    # Reshape the RGB values into a 3x9 array
    rgb_vals = rgb_vals.reshape((3, num_vals))

    # Calculate the yellow score for each numerical value
    # ugly slow implementation
    yellow_scores = []
    for i in range(num_vals):
        red_val = rgb_vals[0][i]
        green_val = rgb_vals[1][i]
        blue_val = rgb_vals[2][i]
        yellow_score = (red_val + green_val - blue_val) / 2.0
        yellow_scores.append(yellow_score)

    return yellow_scores

def sum_two_most_yellow(data, num_vals=9):
    yellow_scores = [calculate_yellow_scores(value, num_vals=num_vals) for value in data]
    sorted_indices = np.argsort(yellow_scores) # sort in descending order
    top_indices = sorted_indices[:,-2:] # get the indices of the two most yellow values
    most_yellow = data[np.arange(len(data))[:, None], top_indices]
    return np.sum(most_yellow, axis=1)


def plot_data_row(data_row, num_vals=9):
    # Extract the numerical and RGB values from the data row
    numerical_vals = data_row[:num_vals]
    rgb_vals = data_row[num_vals:]

    # Reshape the RGB values into a 3x9 array
    rgb_vals = rgb_vals.reshape((3, num_vals))

    # Calculate the yellow scores for each numerical value
    yellow_scores = calculate_yellow_scores(data_row, num_vals=num_vals)

    # Determine the two most yellow values
    top_indices = np.argsort(yellow_scores)[-2:]

    # Define the font colors based on the RGB values
    font_colors = rgb_vals.T

    # Create a 3x3 grid of subplots with reduced spacing
    fig, axs = plt.subplots(3, 3, figsize=(2, 2),
                            gridspec_kw={'hspace': 0, 'wspace': 0})

    # Iterate over the numerical values and font colors, and add each value to a subplot
    for i, (val, color) in enumerate(zip(numerical_vals, font_colors)):

        row = i // 3
        col = i % 3
        axs[row, col].text(0.5, 0.5, f'{val:.1f}',
                           horizontalalignment='center',
                           verticalalignment='center',
                           fontsize=18, color=color)
        axs[row, col].axis('off')
        if i in top_indices:
            circle = plt.Circle((0.5, 0.5), radius=.4, color='blue', fill=False, alpha=0.1)
            axs[row, col].add_artist(circle)

    plt.show()
    return fig