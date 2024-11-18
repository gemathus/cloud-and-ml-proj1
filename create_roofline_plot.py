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
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the roofline lines
oi = np.logspace(-2, 2, 100)  # Operational intensity range
memory_bound = oi * peak_memory_bandwidth
compute_bound = [peak_computational_performance] * len(oi)

ax.plot(oi, memory_bound, label="Memory Bandwidth (1500 GB/s)", linestyle="--", color="blue")
ax.plot(oi, compute_bound, label="Compute Peak (20 TFLOPs)", linestyle="--", color="red")

# Plot data points
ax.scatter(df["Operational Intensity"], df["Performance (GFLOPs)"], color="green", label="Kernels")

# Labels and legend
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Operational Intensity (FLOPs/Byte)", fontsize=12)
ax.set_ylabel("Performance (GFLOPs)", fontsize=12)
ax.set_title("Roofline Model", fontsize=14)
ax.legend()
ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show plot
plt.tight_layout()


# Save the plot to a file with the same name as the CSV file but with a .png extension
plt.savefig("./" + args.csv_file.replace(".csv", ".png"))
