import numpy as np
import matplotlib.pyplot as plt

# Input data
llama_0 = [0.58, 0.42, 0.50, 0.75, 0.83]
qwen_0 = [0.58, 0.45, 0.63, 0.75, 0.92]  # 0.96 should be treated as an outlier
mistral_0 = [0.63, 0.54, 0.75, 0.71, 0.79]

# fold 1
llama_1 = [0.50, 0.63, 0.42, 0.71, 0.63]
qwen_1 = [0.83, 0.21, 0.54, 0.58, 1.0]
mistral_1 = [0.29, 0.29, 0.67, 0.71, 0.96]

# fold 2
llama_2 = [0.58, 0.38, 0.63, 0.17, 0.67]
qwen_2 = [0.63, 0.99, 0.46, 0.46, 1.0]  # Contains outliers (1.0)
mistral_2 = [0.63, 0.42, 0.54, 0.75, 1.0]

# X-axis: Trigger Number
trigger_number = [10, 50, 200, 400, 600]

# Mark outliers, values of 1.0 and 0.96 are marked as NaN
qwen_2_clean = [np.nan if value == 0.99 else value for value in qwen_2]
qwen_0_clean = [np.nan if value == 0.96 else value for value in qwen_0]

# Calculate mean and standard deviation (ignore NaN)
llama_mean = np.mean([llama_0, llama_1, llama_2], axis=0)
qwen_mean = np.nanmean([qwen_0_clean, qwen_1, qwen_2_clean], axis=0)  
mistral_mean = np.mean([mistral_0, mistral_1, mistral_2], axis=0)

llama_std = np.std([llama_0, llama_1, llama_2], axis=0)
qwen_std = np.nanstd([qwen_0_clean, qwen_1, qwen_2_clean], axis=0)  
mistral_std = np.std([mistral_0, mistral_1, mistral_2], axis=0)

# Colors
colors = {
    "llama": "#1f77b4",  # Blue
    "qwen": "#ff7f0e",   # Orange
    "mistral": "#2ca02c" # Green
}

# Plot line chart
plt.figure(figsize=(8, 6))

# Llama
plt.plot(trigger_number, llama_mean, label="Llama", color=colors["llama"], marker="o")
plt.fill_between(trigger_number, llama_mean - llama_std, llama_mean + llama_std, color=colors["llama"], alpha=0.2)

# Qwen
plt.plot(trigger_number, qwen_mean, label="Qwen", color=colors["qwen"], marker="s")
plt.fill_between(trigger_number, qwen_mean - qwen_std, qwen_mean + qwen_std, color=colors["qwen"], alpha=0.2)

# Mistral
plt.plot(trigger_number, mistral_mean, label="Mistral", color=colors["mistral"], marker="^")
plt.fill_between(trigger_number, mistral_mean - mistral_std, mistral_mean + mistral_std, color=colors["mistral"], alpha=0.2)

# Mark outliers
for i, value in enumerate(qwen_0):
    if value == 0.96:
        plt.scatter(trigger_number[i], value, color=colors["qwen"], edgecolors="black", marker="x", s=100, label="Outlier" if i == 0 else "")

for i, value in enumerate(qwen_2):
    if value == 1.0:
        plt.scatter(50, value, color=colors["qwen"], edgecolors="black", marker="x", s=100, label="Outlier" if i == 0 else "")

# Set legend, title, and ticks
plt.xlabel("Trigger Number", fontsize=14)
plt.ylabel("IP-ROC", fontsize=14)
plt.title("The Effect of Trigger Numbers", fontsize=16)
plt.xticks([10, 50, 200, 400, 600], fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12, loc='lower right')

# Save figure
plt.savefig("roc.pdf", bbox_inches="tight")
plt.show()