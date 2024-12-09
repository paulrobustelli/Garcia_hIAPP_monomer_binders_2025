import matplotlib.pyplot as plt

def set_custom_formats():
    """Set custom matplotlib formats for consistent styling."""
    plt.rcParams.update({
        'xtick.labelsize': 10,  # Font size for x-tick labels
        'ytick.labelsize': 10,  # Font size for y-tick labels
        'axes.labelsize': 12,   # Font size for x and y axis labels
        'legend.fontsize': 8,   # Font size for legend labels
        'figure.figsize': (4, 3),    # Default figure size (width, height) in inches
    })

# Call the function to set the formats when this file is imported
set_custom_formats()
