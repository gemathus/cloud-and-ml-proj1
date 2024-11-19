import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Create a roofline plot from NSight CSV data')
parser.add_argument('csv_file', help='Path to the CSV file containing NSight profiling data')
args = parser.parse_args()

# Load the dataset from the provided CSV file
df = pd.read_csv(args.csv_file)

# Convert gpu__time_duration.sum to numeric, coercing errors to NaN
df["gpu__time_duration.sum"] = pd.to_numeric(df["gpu__time_duration.sum"], errors='coerce')

# Compute derived metrics
df["Memory Traffic (Bytes)"] = df["dram__bytes_read.sum"] + df["dram__bytes_write.sum"]
df["FLOPs"] = (
    df["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"]
    + df["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"] * 2
    + df["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"]
)
df["Time (Seconds)"] = df["gpu__time_duration.sum"] / 1e9  # Convert nanoseconds to seconds

# Convert FLOPs and Memory Traffic to numeric, coercing errors to NaN
df["FLOPs"] = pd.to_numeric(df["FLOPs"], errors='coerce')
df["Memory Traffic (Bytes)"] = pd.to_numeric(df["Memory Traffic (Bytes)"], errors='coerce')

# Operational Intensity (FLOPs / Bytes)
df["Operational Intensity"] = df["FLOPs"] / df["Memory Traffic (Bytes)"]

# Performance (FLOPs / Seconds)
df["Performance (GFLOPs)"] = df["FLOPs"] / df["Time (Seconds)"] / 1e9

# System-specific roofline constraints
peak_memory_bandwidth = 1500  # GB/s (example value)
peak_computational_performance = 20  # TFLOPs (example value)

# Create the roofline plot
# Set dark style with custom colors
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1C1C1C')
ax.set_facecolor('#1C1C1C')

# Plot the roofline lines with gradient
oi = np.logspace(-2, 2, 200)  # More points for smoother lines
memory_bound = oi * peak_memory_bandwidth
compute_bound = [peak_computational_performance] * len(oi)

# Create gradient colors
colors = plt.cm.plasma(np.linspace(0, 1, len(oi)))
for i in range(len(oi)-1):
    ax.plot(oi[i:i+2], memory_bound[i:i+2], color=colors[i], linewidth=2)
    ax.plot(oi[i:i+2], compute_bound[i:i+2], color=colors[i], linewidth=2)

# Plot data points with custom markers and sizes
performance = df["Performance (GFLOPs)"]
intensity = df["Operational Intensity"]
sizes = np.log10(performance) * 100  # Size based on performance
ax.scatter(intensity, performance, c='#FF69B4', s=sizes, alpha=0.6, 
          marker='s', label="Compute Kernels", edgecolor='white')

# Get CSV filename without extension and convert to title case
title = args.csv_file.replace(".csv", "").replace("-", " ").title()

# Customize labels and appearance
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Operational Intensity (FLOPs/Byte)", fontsize=14, color='white')
ax.set_ylabel("Performance (GFLOPs)", fontsize=14, color='white')
ax.set_title(f"GPU Performance Roofline Analysis\n{title}", fontsize=16, color='white', pad=20)

# Custom legend
legend_elements = [
    plt.Line2D([0], [0], color='#FF00FF', label='Memory Bandwidth Limit'),
    plt.Line2D([0], [0], color='#00FFFF', label='Peak Compute Limit'),
    plt.scatter([0], [0], c='#FF69B4', marker='s', s=100, 
                label='Compute Kernels', edgecolor='white')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=12, 
          facecolor='#1C1C1C', edgecolor='white')

# Customize grid
ax.grid(True, which="both", linestyle=':', alpha=0.3, color='gray')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white') 
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(colors='white')

# Show plot
plt.tight_layout()


# Save the plot to a file with the same name as the CSV file but with a .png extension
plt.savefig("./" + args.csv_file.replace(".csv", ".png"))
