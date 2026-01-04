import numpy as np
import matplotlib.pyplot as plt
import os

# Membership functions
def triangular_mf(x, a, b, c):
    y = np.zeros_like(x, dtype=float)
    idx = (a < x) & (x < b)
    y[idx] = (x[idx] - a) / (b - a)
    y[x == b] = 1.0
    idx = (b < x) & (x < c)
    y[idx] = (c - x[idx]) / (c - b)
    return np.clip(y, 0, 1)

def trapezoidal_mf(x, a, b, c, d):
    y = np.zeros_like(x, dtype=float)
    idx = (a < x) & (x < b)
    y[idx] = (x[idx] - a) / (b - a)
    idx = (b <= x) & (x <= c)
    y[idx] = 1.0
    idx = (c < x) & (x < d)
    y[idx] = (d - x[idx]) / (d - c)
    return np.clip(y, 0, 1)

# Universe of discourse (wind speed in km/h)
x = np.linspace(0, 60, 1000)

# Linguistic terms
very_low  = trapezoidal_mf(x, 0, 0, 5, 12)
low       = triangular_mf(x, 8, 15, 22)
medium    = triangular_mf(x, 18, 28, 38)
high      = triangular_mf(x, 32, 42, 52)
very_high = trapezoidal_mf(x, 48, 55, 60, 60)

# Crisp input
x0 = 20.0  # km/h

# Membership evaluation
def mu_at(mf, x, x0):
    return float(np.interp(x0, x, mf))

mu_values = {
    "Very Low":  mu_at(very_low, x, x0),
    "Low":       mu_at(low, x, x0),
    "Medium":    mu_at(medium, x, x0),
    "High":      mu_at(high, x, x0),
    "Very High": mu_at(very_high, x, x0),
}

# Get base file name for saving figures
file_name = os.path.splitext(__file__)[0]  # Get filename without extension

# ===== Figure 1: Membership Functions Plot =====
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(x, very_low,  label="Very Low")
ax1.plot(x, low,       label="Low")
ax1.plot(x, medium,    label="Medium")
ax1.plot(x, high,      label="High")
ax1.plot(x, very_high, label="Very High")
ax1.axvline(x0, linestyle="--", label=f"x = {x0:.0f} km/h")

ax1.set_xlabel("Wind Speed (km/h)")
ax1.set_ylabel("Membership Degree")
ax1.set_title("Membership Functions and Fuzzification of a Single Input Value")
ax1.set_ylim(-0.05, 1.05)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(alpha=0.3)

# Annotate memberships outside the plot area
mu_text = "\n".join([f"μ_{label.replace(' ', '')}({x0:.0f}) = {mu:.2f}" 
                     for label, mu in mu_values.items()])
ax1.text(1.02, 0.5, mu_text, transform=ax1.transAxes, 
        verticalalignment='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Make room for the text on the right
plt.savefig(f"{file_name} (1).png", dpi=300, bbox_inches='tight')
plt.show()

# ===== Figure 2: Bar Chart =====
labels = ["Very Low", "Low", "Medium", "High", "Very High"]
values = [
    mu_at(very_low, x, x0),
    mu_at(low, x, x0),
    mu_at(medium, x, x0),
    mu_at(high, x, x0),
    mu_at(very_high, x, x0),
]

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.bar(labels, values)
ax2.set_ylabel("Membership Degree")
ax2.set_xlabel("Linguistic Term")
ax2.set_title(f"Membership Degrees Produced by Fuzzification (x = {x0:.0f} km/h)")
ax2.set_ylim(0, 1.05)
ax2.set_xticklabels(labels, rotation=20, ha="right")
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{file_name} (2).png", dpi=300, bbox_inches='tight')
plt.show()

# Print numerical values
print(mu_values)
