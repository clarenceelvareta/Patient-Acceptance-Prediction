# Loads the saved model and evaluates it on the held-out
# validation set with the following metrics:
#
#   1. Accuracy          — % of referrals correctly classified
#   2. Per-class F1      — F1 for Accepted / Rejected / Withdrawn
#   3. Confusion Matrix  — where the model makes mistakes
#   4. Log-Likelihood    — how confident and correct predictions are
#   5. AIC (final model) — complexity-penalised fit score
#   6. Calibration       — are predicted probabilities trustworthy?
#   7. Trajectory QA     — do latent states evolve smoothly?
# =============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, log_loss, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1) LOAD SAVED ARTEFACTS FROM TRAINING SCRIPT

print("Step 1: Loading saved model and data...")

with open('dkf_config.pkl', 'rb') as f:
    config = pickle.load(f)

INPUT_DIM   = config['input_dim']
LATENT_DIM  = config['latent_dim']
HIDDEN_DIM  = config['hidden_dim']
NUM_CLASSES = config['num_classes']

with open('dkf_val_indices.pkl', 'rb') as f:
    idx_val = pickle.load(f)

with open('dkf_patient_ids.pkl', 'rb') as f:
    patient_ids = pickle.load(f)

X_all = np.load('dkf_X_all.npy')
Y_all = np.load('dkf_Y_all.npy')
L_all = np.load('dkf_L_all.npy')

X_val = X_all[idx_val]
Y_val = Y_all[idx_val]
L_val = L_all[idx_val]

print(f"  Model config : latent_dim={LATENT_DIM}, hidden_dim={HIDDEN_DIM}, input_dim={INPUT_DIM}")
print(f"  Val patients : {len(idx_val):,}")

# 2) REBUILD MODEL & LOAD WEIGHTS

class DeepKalmanFilter(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2,
                          batch_first=True, dropout=0.2)
        self.infer_mean   = nn.Linear(hidden_dim, latent_dim)
        self.infer_logvar = nn.Linear(hidden_dim, latent_dim)
        self.transition   = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.emission = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden_dim, num_classes),
        )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reparameterise(self, mean, logvar):
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def forward(self, x, lengths):
        gru_out, _ = self.gru(x)
        z_mean     = self.infer_mean(gru_out)
        z_logvar   = self.infer_logvar(gru_out)
        z_seq      = self.reparameterise(z_mean, z_logvar)
        z_prior    = torch.zeros_like(z_mean)
        z_prior[:, 1:, :] = self.transition(z_seq[:, :-1, :])
        kl_loss = -0.5 * torch.mean(
            1 + z_logvar - (z_mean - z_prior).pow(2) - z_logvar.exp()
        )
        return self.emission(z_seq), z_seq, kl_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = DeepKalmanFilter(INPUT_DIM, LATENT_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
model.load_state_dict(torch.load('dkf_best_model.pt', map_location=device))
model.eval()
print(f"  Model loaded  : {model.count_parameters():,} trainable parameters")

# STEP 3: RUN INFERENCE ON VALIDATION SET
# Collect all predictions and true labels, ignoring padding

print("\nStep 2: Running inference on validation set...")

val_dl = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(Y_val), torch.tensor(L_val)),
    batch_size=64
)

all_probs, all_preds, all_true = [], [], []
all_z = []  # latent states for trajectory QA

with torch.no_grad():
    for X_b, Y_b, L_b in val_dl:
        X_b = X_b.to(device)
        logits, z_seq, _ = model(X_b, L_b)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [B, T, 3]
        preds = np.argmax(probs, axis=-1)                    # [B, T]
        Y_np  = Y_b.numpy()                                  # [B, T]

        for b in range(Y_np.shape[0]):
            mask = Y_np[b] != -1   # ignore padding
            all_probs.extend(probs[b][mask])
            all_preds.extend(preds[b][mask])
            all_true.extend(Y_np[b][mask])
            all_z.extend(z_seq[b].cpu().numpy()[mask])

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_true  = np.array(all_true)
all_z     = np.array(all_z)

CLASS_NAMES = ['Accepted', 'Rejected', 'Withdrawn']


# 4) CLASSIFICATION METRICS

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

accuracy = accuracy_score(all_true, all_preds)
print(f"\n[1] Accuracy: {accuracy:.4f}  ({accuracy*100:.2f}%)")

print("\n[2] Per-class Precision / Recall / F1:")
report = classification_report(
    all_true, all_preds,
    target_names=CLASS_NAMES,
    digits=4
)
print(report)

# 5) CONFUSION MATRIX
# Rows = actual outcome, Columns = predicted outcome
# Off-diagonal cells show where the model gets confused

print("[3] Confusion Matrix:")
cm = confusion_matrix(all_true, all_preds)
print(f"     {'':>12}  " + "  ".join(f"{n:>10}" for n in CLASS_NAMES))
for i, row in enumerate(cm):
    print(f"  {CLASS_NAMES[i]:>12}  " + "  ".join(f"{v:>10}" for v in row))

fig, ax = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title('DKF Confusion Matrix — Validation Set')
plt.tight_layout()
plt.savefig('dkf_confusion_matrix.png', dpi=150)
plt.close()
print("  → Saved: dkf_confusion_matrix.png")


# 6) LOG-LIKELIHOOD & AIC ON FINAL MODEL
# Log-likelihood measures how well the model's predicted
# probabilities match the actual outcomes.

print("\n[4] Log-Likelihood & AIC (final trained model):")

# sklearn log_loss = mean negative log-likelihood
nll_per_token = log_loss(all_true, all_probs, labels=[0, 1, 2])
total_nll     = nll_per_token * len(all_true)
log_likelihood = -total_nll

k   = model.count_parameters()
aic = 2 * k - 2 * log_likelihood

print(f"  Total tokens evaluated  : {len(all_true):,}")
print(f"  NLL per token           : {nll_per_token:.6f}")
print(f"  Total log-likelihood    : {log_likelihood:.2f}")
print(f"  Num parameters (k)      : {k:,}")
print(f"  AIC (final model)       : {aic:.2f}")
print()
print("  Interpretation:")
print("  AIC rewards models that fit well WITHOUT being overly complex.")
print("  Compare this value against the AIC comparison table in")
print("  dkf_aic_comparison.csv to confirm the best latent dim was chosen.")

# Also load and display the AIC comparison table
try:
    aic_table = pd.read_csv('dkf_aic_comparison.csv')
    print("\n  AIC Comparison (from model selection):")
    print(aic_table.to_string(index=False))
except FileNotFoundError:
    pass


# 7) CALIBRATION CHECK
# A well-calibrated model should have predicted probability ~0.7
# for outcomes that occur ~70% of the time.
# We bin predictions into deciles and compare predicted vs actual.

print("\n[5] Calibration Check (Accepted class):")

accepted_probs = all_probs[:, 0]  # predicted P(Accepted)
accepted_true  = (all_true == 0).astype(int)

bins = np.linspace(0, 1, 11)
bin_indices = np.digitize(accepted_probs, bins) - 1
bin_indices = np.clip(bin_indices, 0, 9)

cal_rows = []
for b in range(10):
    mask = bin_indices == b
    if mask.sum() == 0:
        continue
    mean_pred   = accepted_probs[mask].mean()
    mean_actual = accepted_true[mask].mean()
    cal_rows.append({
        'Predicted_Prob_Bin': f"{bins[b]:.1f}–{bins[b+1]:.1f}",
        'Mean_Predicted':     round(mean_pred, 4),
        'Actual_Rate':        round(mean_actual, 4),
        'Gap':                round(abs(mean_pred - mean_actual), 4),
        'N':                  int(mask.sum()),
    })

cal_df = pd.DataFrame(cal_rows)
print(cal_df.to_string(index=False))
print("\n  A small 'Gap' means the model's probabilities are trustworthy.")
print("  A large gap means predictions are over/under confident.")

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.scatter(cal_df['Mean_Predicted'], cal_df['Actual_Rate'],
           s=cal_df['N'] / cal_df['N'].max() * 300,
           alpha=0.7, color='steelblue', label='Model')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Actual Acceptance Rate')
ax.set_title('Calibration Plot — P(Accepted)')
ax.legend()
plt.tight_layout()
plt.savefig('dkf_calibration_plot.png', dpi=150)
plt.close()
print("  → Saved: dkf_calibration_plot.png")

# 8) LATENT TRAJECTORY QUALITY CHECK
# If the DKF is learning meaningful dynamics, the latent state
# norm should change across a patient's referrals, so patients
# moving toward acceptance should show a directional shift.

print("\n[6] Latent Trajectory Quality:")

traj_df = pd.read_csv('dkf_trajectory.csv')
mean_by_step = traj_df.groupby('Referral_Number')['Latent_State_Norm'].mean()
print(mean_by_step.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
mean_by_step.plot(ax=ax, marker='o', color='darkorange')
ax.set_xlabel('Referral Number (chronological)')
ax.set_ylabel('Mean Latent State Norm')
ax.set_title('Mean Latent State Norm by Referral Step')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dkf_latent_trajectory.png', dpi=150)
plt.close()
print("  → Saved: dkf_latent_trajectory.png")


# 9) SAVE FULL EVALUATION REPORT

eval_summary = {
    'Metric': [
        'Accuracy', 'NLL per token', 'Total Log-Likelihood',
        'Num Parameters (k)', 'AIC (final model)', 'Latent Dim Used',
    ],
    'Value': [
        round(accuracy, 4),
        round(nll_per_token, 6),
        round(log_likelihood, 2),
        k,
        round(aic, 2),
        LATENT_DIM,
    ]
}
eval_df = pd.DataFrame(eval_summary)
eval_df.to_csv('dkf_evaluation_report.csv', index=False)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
print("Files saved:")
print("  → dkf_evaluation_report.csv    (key metrics summary)")
print("  → dkf_confusion_matrix.png     (confusion matrix plot)")
print("  → dkf_calibration_plot.png     (calibration plot)")
print("  → dkf_latent_trajectory.png    (latent state norm by step)")