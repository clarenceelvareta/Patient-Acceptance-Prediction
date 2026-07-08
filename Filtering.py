# Deep Kalman Filter


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle, warnings

#CLOSE ALL FILES BEFORE RUNNING NO EXCEL IS TO BE OPEN

# Remove nonsense (API change, deprecation notice, etc)
# and make output easier to read
warnings.filterwarnings('ignore')
# Model has randomness included due to accounting for noise, due to data
# lack, we have to account for natural human error via randomness.

random_seed = 2018
torch.manual_seed(2018)
np.random.seed(2018)


#1) LOAD DATA

#Make reading output easier
print("Step 1: Loading data...")
df = pd.read_csv('combined_repeat_patients.csv')
print(f"  Rows: {len(df):,}  |  Patients: {df['Pseudo NRIC'].nunique():,}")


#2) DATETIME FEATURE SELECTION

# Neural networks need actual time series instead of datetime columns
# Make reading output easier
print("Step 2: Engineering features...")

# Run through datetime columns, setting error to coerce(read pandas docu)
# Changes unparseable values into NaT instead of exploding
for col in ['RS Submit Datetime', 'RS Last Status Datetime']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
if 'DOB' in df.columns:
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')

# Processing_Days is the number of days between current submission and last
# status update. with clip(lower=0) to prevent negative in case people stupid
# and key in last_status < submit_date
if 'RS Submit Datetime' in df.columns and 'RS Last Status Datetime' in df.columns:
    df['Processing_Days'] = (
        df['RS Last Status Datetime'] - df['RS Submit Datetime']
    ).dt.days.fillna(0).clip(lower=0)
else:
    df['Processing_Days'] = 0

# take year and month from datetime. If datetime column is
# missing, we take from app_id
if 'RS Submit Datetime' in df.columns:
    df['Submit_Year']  = df['RS Submit Datetime'].dt.year.fillna(2020).astype(int)
    df['Submit_Month'] = df['RS Submit Datetime'].dt.month.fillna(1).astype(int)
else:
    df['Submit_Year']  = df['Application ID'].astype(str).str.extract(r'^(\d{4})').astype(int)
    df['Submit_Month'] = 1


# Sort each referral by time before computing the time since the first referral
# Essential to DKF as DKF processes by time order defined by us.
df = df.sort_values(['Pseudo NRIC', 'Submit_Year', 'Submit_Month']).reset_index(drop=True)


# For each row, see how many days have passed since first referral
if 'RS Submit Datetime' in df.columns:
    df['First_Submit'] = df.groupby('Pseudo NRIC')['RS Submit Datetime'].transform('min')
    df['Days_Since_First_Referral'] = (
        df['RS Submit Datetime'] - df['First_Submit']
    ).dt.days.fillna(0).clip(lower=0)
else:
    df['Days_Since_First_Referral'] = 0


#3) OUTCOME MAPPING
# Map the 6 outcomes to 3 integer values. We can set withdrawn groups all
# together since withdrawals are all patient/service initiated

outcome_map = {
    'admit':                                        0,
    'reject by agency':                             1,
    'pre assign withdrawn':                         2,
    'pre assign withdrawn-bpr':                     2,
    'withdrawn':                                    2,
    'withdrawn and case closed by referral source': 2,
    'withdrawn from service provider waiting list': 2,
}

df['Outcome_Code'] = (
    df['RS Outcome Item Desc']
    .astype(str).str.strip().str.lower()
    .map({k.lower(): v for k, v in outcome_map.items()})
)

# Removes null values or values outside mapping
df = df[df['Outcome_Code'].notna()].copy()
df['Outcome_Code'] = df['Outcome_Code'].astype(int)

# 4) FEATURE CATEGORISATION

# Array for text Columns
CATEGORICAL_COLS = [
    'Referral Source', 'AIC Log Patient Curr Location', 'RS Svc Type Desc',
    'Gender', 'Race', 'Citizenship', 'Age Group', 'During Covid?',
    'Religion', 'Living Arrangement', 'Lift Landing Ind',
    'Accom Status', 'Accom Summary',
    'RS Referral Status',
]

#Array for number columns
NUMERICAL_COLS = [
    'Age AO Application', 'Num of Caregiver',
    'AIC Log Level of Assitance', 'AIC Log Feeding', 'AIC Log Mobility',
    'AIC Log Toileting', 'AIC Log Transfer', 'AIC Log Hearing Impairment',
    'AIC Log Mental Status', 'AIC Log Activity Tolerance',
    'AIC Log Visual Impairment', 'AIC Log Wound Care',
    'AIC Log Aids Needed', 'AIC Log Respiratory Care',
    'ct_brotherfamilyrelationship', 'ct_daughter', 'ct_daughter-in-law',
    'ct_neighbour', 'ct_other relatives', 'ct_otherrelative',
    'ct_parent', 'ct_sibling', 'ct_sisterfamilyrelationship',
    'ct_son', 'ct_son-in-law', 'ct_spouse',
    'Referral Source_freq', 'SP Full Name_freq',
    'Referral Source_is_rare', 'SP Full Name_is_rare',
    'duplicate_count', 'Submit_Year', 'Submit_Month',
    'Days_Since_First_Referral',
]

# Fill in missing values before encoding, numerical blanks/error set to 0, will be
# scaled later, text blanks/errors set to unknown, categorise later
for col in NUMERICAL_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
for col in CATEGORICAL_COLS:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown').astype(str)

# takes only text columns that actually exist in dataset, prevents columns
# being renamed or missing
cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
df_encoded = pd.get_dummies(df, columns=cat_cols_present)

# collect last list of feature column includes the binary column and original
# number column
feature_cols = (
    [c for c in df_encoded.columns if any(c.startswith(cat) for cat in cat_cols_present)]
    + [c for c in NUMERICAL_COLS if c in df_encoded.columns]
)

# fit the scaler on training data and transform all features, then find mean
# and s.d of each column, then subtracts mean and divides by s.d for each value
scaler = StandardScaler()
df_encoded[feature_cols] = scaler.fit_transform(df_encoded[feature_cols].fillna(0))
INPUT_DIM = len(feature_cols)
print(f"  Feature dimensionality: {INPUT_DIM}")


# save scaler and feature list so we can apply the same transformation to new
# data without refitting, stops data leakage
with open('dkf_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('dkf_feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)


# 5) BUILD PATIENT SEQUENCES
# Final set up for data to put into DKF
# set sequence to be (x,y) where x is array for feature matrix for patient
# one row per referral in chronological order and y is array for the outcome
# for each referral
# Need neural network for data processing. Do in batches and all sequence
# must be same length. For shorter sequence, we put 0 for x and -1 for y
# preventing the filler from touching the gradient

print("Step 3: Building patient sequences...")

sequences, patient_ids = [], []
for nric, group in df_encoded.groupby('Pseudo NRIC'):
    if len(group) < 2:
        continue
    sequences.append((
        group[feature_cols].values.astype(np.float32),
        group['Outcome_Code'].values.astype(np.int64),
    ))
    patient_ids.append(nric)

# Cap sequence length at 25 referrals, most have lesser than 25, just to limit
# extreme outliers
print(f"  Sequences built: {len(sequences):,}")
MAX_SEQ_LEN = min(25, max(x.shape[0] for x, _ in sequences))

def pad_sequences(seqs, max_len):
    X_out, Y_out, L_out = [], [], []
    for X, Y in seqs:
        T = X.shape[0]
        pad = max_len - T
        X_out.append(np.pad(X, ((0, pad), (0, 0))))
        Y_out.append(np.pad(Y, (0, pad), constant_values=-1))
        L_out.append(T)
    return (np.array(X_out, dtype=np.float32),
            np.array(Y_out, dtype=np.int64),
            np.array(L_out, dtype=np.int64))

X_all, Y_all, L_all = pad_sequences(sequences, MAX_SEQ_LEN)

#I HATE CODING AHHHHHHHH
# Dataset for PyTorch use standard wrapper so dataloader can see
class ReferralDataset(Dataset):
    def __init__(self, X, Y, L):
        self.X, self.Y, self.L = map(torch.tensor, (X, Y, L))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i], self.L[i]

idx_tr, idx_val = train_test_split(range(len(sequences)), test_size=0.2, random_state=42)

# Persist split & arrays for testing script, 80 for training, 20 for validation
# test and training both use validation dataset, drop out
# if you need to ask me why
with open('dkf_val_indices.pkl', 'wb') as f: pickle.dump(idx_val, f)
with open('dkf_patient_ids.pkl', 'wb') as f: pickle.dump(patient_ids, f)
np.save('dkf_X_all.npy', X_all)
np.save('dkf_Y_all.npy', Y_all)
np.save('dkf_L_all.npy', L_all)

# go read PyTorch documentation u fuckwit

train_dl = DataLoader(ReferralDataset(X_all[idx_tr],  Y_all[idx_tr],  L_all[idx_tr]),  batch_size=64, shuffle=True)
val_dl   = DataLoader(ReferralDataset(X_all[idx_val], Y_all[idx_val], L_all[idx_val]), batch_size=64)


# 6) MODEL DEFINITION
# Use PyTorch nn.Module class. nn stands for neural network, u figure out why
# we use nn to build neural network
# it uses 3 sub-networks- GRU Inference Network, Transition Network, and
# Emission Network. I not explaning this here go read up deep kalman filter
# on your own. Or just put the code in claude or use your brain for once u
# fucking 3rd party thinkers


# Hyperparameters for models
HIDDEN_DIM  = 64 #size of all intermediary hidden layer
NUM_CLASSES = 3 #integer classifications

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


# 7) AIC- not the company. the criterion
# We test 4 latent dimensions for lowest aic.
# AIC penalises models that gain only tiny improvements in fit
# at the cost of many more parameters. Cost outweigh benefit.
# So lowest aic means better model.

print("\nStep 4: AIC model selection across latent dimensions...")
print("  Testing latent dims: [8, 16, 32, 64]")
print("  (Trains 4 quick models — takes a few minutes)\n")
# so yall know something is happening and your com isn't just lagging

KL_WEIGHT    = 0.1 # this means prediction loss is 10x more important than
# trajectory smoothness
# Please use your gpu, or don't and wait a long long time for ur cpu to have fun
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using device: {device}")

# sums loss across all tokens to give a better aic calc, ignores padded pos
ce_sum       = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

#count non-padding tokens in validation set to normalise loss during training
n_val_tokens = int((Y_all[idx_val] != -1).sum())

aic_results  = []

for latent_dim in [8, 16, 32, 64]:
    # New model for each config
    model = DeepKalmanFilter(INPUT_DIM, latent_dim, HIDDEN_DIM, NUM_CLASSES).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    k     = model.count_parameters() # aic parameters

# Quick training run: 20 plas-45 for representative fits
    for epoch in range(20):
        model.train()
        for X_b, Y_b, L_b in train_dl:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            opt.zero_grad()
            logits, _, kl = model(X_b, L_b)
            # Normalise by n_val_tokens so loss scale is comparable across
            # models with different numbers of different VALID timesteps
            loss = ce_sum(logits.view(-1, NUM_CLASSES), Y_b.view(-1)) / n_val_tokens + KL_WEIGHT * kl
            loss.backward() # backpropagation for gradient
            # Stops clipping and the gradient 爆炸 by capping norm at 1
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step() # update model weight

    # Evaluate NLL on the validation set (no need find gradient)
    model.eval()
    total_nll = 0.0
    with torch.no_grad():
        for X_b, Y_b, L_b in val_dl:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            logits, _, _ = model(X_b, L_b)
            total_nll += ce_sum(logits.view(-1, NUM_CLASSES), Y_b.view(-1)).item()

    aic = 2 * k - 2 * (-total_nll)
    aic_results.append({'latent_dim': latent_dim, 'n_parameters': k,
                        'val_nll': round(total_nll, 2), 'AIC': round(aic, 2)})
    print(f"  Latent {latent_dim:3d} | Params: {k:,} | NLL: {total_nll:.1f} | AIC: {aic:.1f}")

# lowest score is best model
aic_df = pd.DataFrame(aic_results).sort_values('AIC')
aic_df.to_csv('dkf_aic_comparison.csv', index=False)

best_latent_dim = int(aic_df.iloc[0]['latent_dim'])
print(f"\n  Best latent dim by AIC: {best_latent_dim}")


# 8) FULL TRAINING OF BEST MODEL
# Run through all 30 epochs based off of AIC's selection of the optimal latent dim
# Training phase then validation phase
# Save model weights whenever the validation loss improves, so if model overfits
# we keep best generalisation. Final model is loaded by testing script so it
# will be lowest validation loss isntead of last epoch.

print(f"\nStep 5: Training best model (latent_dim={best_latent_dim}) for 30 epochs...")

# learning rate of 1e-3 is standard, we set weight decay to 1e-4 to proportionally
# discourage model from relying too heavily on a single feature
model     = DeepKalmanFilter(INPUT_DIM, best_latent_dim, HIDDEN_DIM, NUM_CLASSES).to(device)
optimiser = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# mean reduction for training loss display
ce_mean   = nn.CrossEntropyLoss(ignore_index=-1)
best_val  = float('inf')


for epoch in range(1, 31):
    #training phase
    model.train() #enables dropout during training, disabled otherwise
    tr_loss = 0
    for X_b, Y_b, L_b in train_dl:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        optimiser.zero_grad() # pytorch is dumb and keeps old gradients so must
        #clear or your model will pass away
        logits, _, kl = model(X_b, L_b)
        loss = ce_mean(logits.view(-1, NUM_CLASSES), Y_b.view(-1)) + KL_WEIGHT * kl
        # total loss = prediction accuracy loss + trajectory smoothness loss
        # kl term penalises the inference network for producing trajectories that
        # deviate from the transition network

        loss.backward() #backpropagation for gradient

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)# once again,
        # stops gradient clipping, refer to above on why this is bad
        optimiser.step()# apply computed gradients to update weights
        tr_loss += loss.item()
#validation phase
    model.eval()#disables dropout so evaluation is deterministic
    vl_loss = 0
    with torch.no_grad(): #stops my ram from exploding, disable this if u rich
        for X_b, Y_b, L_b in val_dl:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            logits, _, kl = model(X_b, L_b)
            vl_loss += (ce_mean(logits.view(-1, NUM_CLASSES), Y_b.view(-1)) + KL_WEIGHT * kl).item()

    avg_tr, avg_vl = tr_loss / len(train_dl), vl_loss / len(val_dl)
    # save model if this epoch achieved new loss
    if avg_vl < best_val:
        best_val = avg_vl
        torch.save(model.state_dict(), 'dkf_best_model.pt')

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/30  |  Train: {avg_tr:.4f}  |  Val: {avg_vl:.4f}")
# saves model config so test can rebuild the same network
with open('dkf_config.pkl', 'wb') as f:
    pickle.dump({'latent_dim': best_latent_dim, 'hidden_dim': HIDDEN_DIM,
                 'input_dim': INPUT_DIM, 'num_classes': NUM_CLASSES}, f)

# 9) EXTRACT TRAJECTORIES
print("\nStep 6: Extracting trajectories...")

#load best saved model weights
model.load_state_dict(torch.load('dkf_best_model.pt', map_location=device))
model.eval()
#use submission datetime for sorting first, cannot then app id
sort_col = 'RS Submit Datetime' if 'RS Submit Datetime' in df.columns else 'Application ID'

all_rows = []
with torch.no_grad():
    for nric, (X_seq, Y_seq) in zip(patient_ids, sequences):
        T   = X_seq.shape[0]

        #add a batch dimension, unsqueeze changes [t,f] to [1,t,f]
        X_t = torch.tensor(X_seq).unsqueeze(0).to(device)
        logits, z_seq, _ = model(X_t, torch.tensor([T]))
        #softmax converts raw logits to probabilities summing to 1
        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        z     = z_seq[0].cpu().numpy()
        #get the original rows for this patient in chronological order
        patient_rows = df[df['Pseudo NRIC'] == nric].sort_values(sort_col)
        for t in range(T):
            row = patient_rows.iloc[t]
            all_rows.append({
                'Pseudo NRIC':       nric,
                'Application ID':    row['Application ID'],
                'Submit_Year':       row.get('Submit_Year', ''),
                'Submit_Month':      row.get('Submit_Month', ''),
                'Actual_Outcome':    row['RS Outcome Item Desc'],
                'Referral_Number':   t + 1,
                'Total_Referrals':   T,
                'Prob_Accepted':     round(float(probs[t, 0]), 4),
                'Prob_Rejected':     round(float(probs[t, 1]), 4),
                'Prob_Withdrawn':    round(float(probs[t, 2]), 4),
                'Predicted_Outcome': ['Accepted','Rejected','Withdrawn'][np.argmax(probs[t])],
                #L2 norm of the latent vector: a single number summarising how far the patient's
                #state from the origin of the latent space, change in this value means model
                #detetced a shift in state
                'Latent_State_Norm': round(float(np.linalg.norm(z[t])), 4),
                # we store each dimension of the latent vector individually
                **{f'z_{j}': round(float(z[t, j]), 4) for j in range(best_latent_dim)},
            })

trajectory_df = pd.DataFrame(all_rows)

#per patient summary
summary = []
for nric, grp in trajectory_df.groupby('Pseudo NRIC'):
    grp = grp.sort_values('Referral_Number')
    fp, lp = grp['Prob_Accepted'].iloc[0], grp['Prob_Accepted'].iloc[-1]
    delta, n = lp - fp, len(grp)
    summary.append({
        'Pseudo NRIC':           nric,
        'Total_Referrals':       n,
        'First_Acceptance_Prob': round(fp, 4),
        'Final_Acceptance_Prob': round(lp, 4),
        'Prob_Delta':            round(delta, 4),
        'Trajectory_Trend':      'Improving' if delta > 0.1 else ('Worsening' if delta < -0.1 else 'Stable'),
        'Max_Acceptance_Prob':   round(grp['Prob_Accepted'].max(), 4),
        #fuck you iosif, you rat bastard
        'Is_Cycling_Risk':       (n >= 3) and (grp['Prob_Accepted'].max() < 0.5),
    })

summary_df = pd.DataFrame(summary)
trajectory_df.to_csv('dkf_trajectory.csv',      index=False)
summary_df.to_csv   ('dkf_patient_summary.csv', index=False)

# 10) FINAL SUMMARY
# I CAN SEE THE END OF THE HORIZON, I'M COMING HOME, HATSUNE MIKU????!!!!
print("\n" + "=" * 55)
print("TRAINING COMPLETE")
print("=" * 55)
print(f"Best latent dim (AIC)        : {best_latent_dim}")
print(f"Repeat patients modelled     : {len(summary_df):,}")
print(f"Patients flagged as cycling  : {summary_df['Is_Cycling_Risk'].sum():,}")
print("\nTrajectory Trend Distribution:")
print(summary_df['Trajectory_Trend'].value_counts().to_string())
print("\nFiles saved:")
print("  → dkf_best_model.pt         (trained weights)")
print("  → dkf_aic_comparison.csv    (AIC scores per latent dim)")
print("  → dkf_trajectory.csv        (per-referral latent states)")
print("  → dkf_patient_summary.csv   (per-patient trend + cycling flag)")
print("  → dkf_config.pkl / dkf_scaler.pkl / dkf_feature_cols.pkl  (for test script)")