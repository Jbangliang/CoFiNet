import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"model_predictions\aflow__ael_bulk_modulus_vrh_test_output.csv")

y_true = df["target"].values
y_pred = df["pred-0"].values
unc = df["uncertainty"].values

plt.figure(figsize=(6, 6))
sc = plt.scatter(
    y_true, y_pred,
    c=unc,
    cmap="viridis",
    s=12,
    alpha=0.7
)

min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)

plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title("Regression Parity Plot with Uncertainty")

cbar = plt.colorbar(sc)
cbar.set_label("Predictive Uncertainty")

plt.tight_layout()
plt.show()
