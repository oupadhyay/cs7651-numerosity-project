import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import argparse

# --- Helper functions for calculating effects ---
def calculate_effects(results):
    distances = []
    sizes = []
    ratios = []
    similarities_distance = []
    similarities_size = []
    similarities_ratio = []

    for n1, n2, similarity in results:
        distances.append(abs(n1 - n2))
        sizes.append((n1 + n2) / 2)
        ratios.append(max(n1, n2) / min(n1, n2))
        similarities_distance.append(similarity)
        similarities_size.append(similarity)
        similarities_ratio.append(similarity)

    return (
        distances,
        sizes,
        ratios,
        similarities_distance,
        similarities_size,
        similarities_ratio,
    )

def sigmoid_function(x, lower, upper, inflection, rate):
    return lower + (-lower + upper) / (1 + np.exp((inflection - x) / rate))


# --- Plotting and correlation functions ---
def plot_effects(
    distances,
    sizes,
    ratios,
    similarities_distance,
    similarities_size,
    similarities_ratio,
    epoch,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

    # Distance Effect
    correlation_distance, _ = pearsonr(distances, similarities_distance)
    axes[0].scatter(distances, similarities_distance)
    axes[0].set_xlabel("Distance |n1 - n2|")
    axes[0].set_ylabel("Average Cosine Similarity")
    axes[0].set_title(f"Distance Effect (Epoch {epoch}), r={correlation_distance:.2f}")

    # Size Effect
    correlation_size, _ = pearsonr(sizes, similarities_size)
    axes[1].scatter(sizes, similarities_size)
    axes[1].set_xlabel("Size (n1 + n2) / 2")
    axes[1].set_ylabel("Average Cosine Similarity")
    axes[1].set_title(f"Size Effect (Epoch {epoch}), r={correlation_size:.2f}")

    # Ratio Effect
    try:
        popt, _ = scipy.optimize.curve_fit(
            sigmoid_function,
            ratios,
            similarities_ratio,
            p0=[0.5, 1.0, 1.5, 1.0],
        )  # Fit exponential
        y_pred = sigmoid_function(np.array(ratios), *popt)
        r2 = r2_score(similarities_ratio, y_pred)

        axes[2].scatter(ratios, similarities_ratio)
        x_fit = np.linspace(min(ratios), max(ratios), 100)
        y_fit = popt[0] * np.exp(-popt[1] * x_fit) + popt[2]
        axes[2].plot(x_fit, y_fit, "r-", label="Fitted Sigmoid")  # Plot fitted line

        axes[2].set_xlabel("Ratio max(n1, n2) / min(n1, n2)")
        axes[2].set_ylabel("Average Cosine Similarity")
        axes[2].set_title(f"Ratio Effect (Epoch {epoch}), R²={r2:.2f}")
        axes[2].legend()

    except RuntimeError:
        print(f"No fit for Epoch {epoch} ratio effect.")
        axes[2].scatter(ratios, similarities_ratio)
        axes[2].set_xlabel("Ratio max(n1, n2) / min(n1, n2)")
        axes[2].set_ylabel("Average Cosine Similarity")
        axes[2].set_title(f"Ratio Effect (Epoch {epoch}) - No Fit")

    fig.tight_layout()  # Adjust subplot parameters for a tight layout
    plt.savefig(f"Combined_Effects_Epoch_{epoch}.png")  # Save as PNG
    plt.close(fig)

def plot_correlations(all_correlations, all_r2):
    epochs = list(all_correlations.keys())
    distance_correlations = [all_correlations[epoch][0] for epoch in epochs]
    size_correlations = [all_correlations[epoch][1] for epoch in epochs]
    ratio_r2 = [
        all_r2[epoch] for epoch in epochs if epoch in all_r2
    ]  # Handle cases where ratio fit fails

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].plot(epochs, distance_correlations, marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Pearson Correlation (r)")
    axes[0].set_title("Distance Effect Correlation Across Epochs")
    axes[0].grid(True)

    axes[1].plot(epochs, size_correlations, marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Pearson Correlation (r)")
    axes[1].set_title("Size Effect Correlation Across Epochs")
    axes[1].grid(True)

    axes[2].plot(
        epochs[: len(ratio_r2)], ratio_r2, marker="o"
    )  # Use the correct number of epochs.
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R²")
    axes[2].set_title("Ratio Effect R² Across Epochs")
    axes[2].grid(True)

    fig.tight_layout()
    plt.savefig("Correlation_and_R2_Across_Epochs.png")
    plt.close(fig)

# --- Main effects analysis ---
def main(results_file):
    all_results = torch.load(results_file)
    all_correlations = {}
    all_r2 = {}

    for epoch, results in all_results.items():
        (
            distances,
            sizes,
            ratios,
            similarities_distance,
            similarities_size,
            similarities_ratio,
        ) = calculate_effects(results)
        plot_effects(
            distances,
            sizes,
            ratios,
            similarities_distance,
            similarities_size,
            similarities_ratio,
            epoch,
        )

        correlation_distance, _ = pearsonr(distances, similarities_distance)
        correlation_size, _ = pearsonr(sizes, similarities_size)
        all_correlations[epoch] = (correlation_distance, correlation_size)

        try:
            popt, _ = scipy.optimize.curve_fit(
                lambda x, a, b, c: a * np.exp(-b * x) + c,
                ratios,
                similarities_ratio,
                p0=[1, 1, 0],
            )
            y_pred = popt[0] * np.exp(-popt[1] * np.array(ratios)) + popt[2]
            r2 = r2_score(similarities_ratio, y_pred)
            all_r2[epoch] = r2
        except RuntimeError:
            print(f"No fit for Epoch {epoch} ratio effect.")

    plot_correlations(all_correlations, all_r2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze numerosity effects from results file."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the .pth file containing the results.",
    )
    args = parser.parse_args()
    main(args.results_file)
