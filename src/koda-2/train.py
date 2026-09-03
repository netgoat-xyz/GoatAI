import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILES  = [PROJECT_ROOT / 'synthetic_ddos_dataset.csv', PROJECT_ROOT / 'cicddos2019_dataset.csv']
TEST_FILE    = PROJECT_ROOT / 'cicddos2019_test.csv'  # unseen real-world evaluation

CHUNK_SIZE       = 200_000
BATCH_SIZE       = 512
STEPS_PER_EPOCH  = 2000
EPOCHS           = 2 # how lond the model is being trained

CANONICAL_FEATURES_FILE = PROJECT_ROOT / 'canonical_features.json'
SCALER_FILE             = PROJECT_ROOT / 'scaler.pkl'
MODEL_FILE              = PROJECT_ROOT / 'model.keras'

# Discover canonical features from the first training file
# (legacy schema — inference code never needs to change)
def discover_canonical_features(filename):
    print(f"--- Discovering canonical features from: {filename} ---")
    sample = pd.read_csv(filename, nrows=1)
    features = [c for c in sample.columns if c != 'Label']
    print(f"Found {len(features)} canonical features: {features}")
    return features


def align_to_schema(df, canonical_features):
    """
    Force any dataframe into the canonical feature schema:
      - Missing columns → filled with 0.0 (min of [0,1] scaled range, neutral)
      - Extra columns   → dropped
      - Column order    → enforced
    Label column is preserved separately.
    """
    label = df['Label'].values if 'Label' in df.columns else None

    for col in canonical_features:
        if col not in df.columns:
            df[col] = 0.0

    df = df[canonical_features].copy()

    if label is not None:
        df['Label'] = label

    return df

# Fit scaler ONLY on training data (no test leakage)
def get_scaler(filenames, canonical_features, sample_limit=1_000_000):
    print(f"\n--- Phase 0: Calibrating Scaler on TRAINING data only ---")
    scaler = MinMaxScaler()
    total_seen = 0

    for filename in filenames:
        if not filename.exists():
            print(f"  [SKIP] {filename} not found.")
            continue
        print(f"  Fitting on: {filename}")
        for chunk in pd.read_csv(filename, chunksize=CHUNK_SIZE):
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.dropna(inplace=True)
            chunk = align_to_schema(chunk, canonical_features)

            scaler.partial_fit(chunk.drop('Label', axis=1))
            total_seen += len(chunk)

            if total_seen >= sample_limit:
                break
        if total_seen >= sample_limit:
            break

    print(f"Scaler calibrated on {total_seen} samples.")
    return scaler

# Generator streams benign-only training batches
def train_generator(filenames, canonical_features, scaler, batch_size):
    while True:
        for filename in filenames:
            if not filename.exists():
                continue
            for chunk in pd.read_csv(filename, chunksize=CHUNK_SIZE):
                chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                chunk.dropna(inplace=True)
                chunk = align_to_schema(chunk, canonical_features)

                benign = chunk[chunk['Label'] == 0].drop('Label', axis=1)
                if len(benign) == 0:
                    continue

                X_scaled = scaler.transform(benign)
                num_batches = len(X_scaled) // batch_size
                for i in range(num_batches):
                    batch = X_scaled[i * batch_size:(i + 1) * batch_size]
                    yield (batch, batch)

# STEP 4: Evaluate on a chunk from a given file
def evaluate(autoencoder, scaler, canonical_features, filename, threshold=None):
    print(f"\n  Loading evaluation chunk from: {filename}")
    chunk = next(pd.read_csv(filename, chunksize=CHUNK_SIZE))
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.dropna(inplace=True)
    chunk = align_to_schema(chunk, canonical_features)

    X = chunk.drop('Label', axis=1)
    y = chunk['Label']

    # ── Normalize labels to 0/1 regardless of source dataset ──────
    if y.dtype == object or y.dtype.name == 'category':
        # CIC uses strings: "BENIGN" → 0, everything else → 1
        unique = y.unique()
        print(f"  Detected string labels: {unique}")
        benign_str = [v for v in unique if 'benign' in str(v).lower()]
        if not benign_str:
            raise ValueError(f"Can't identify benign class from labels: {unique}")
        y = y.apply(lambda v: 0 if str(v).lower() == 'benign' else 1)
    else:
        y = y.astype(int)
    # ──────────────────────────────────────────────────────────────

    X_scaled = scaler.transform(X)

    if threshold is None:
        benign_scaled = X_scaled[y == 0]
        recon = autoencoder.predict(benign_scaled, verbose=0)
        loss  = tf.keras.losses.mse(recon, benign_scaled).numpy()
        threshold = np.mean(loss) + 2 * np.std(loss)
        print(f"  Derived threshold from this chunk: {threshold:.6f}")

    all_recon = autoencoder.predict(X_scaled, verbose=1)
    all_loss  = tf.keras.losses.mse(all_recon, X_scaled).numpy()
    predictions = (all_loss > threshold).astype(int)

    acc  = accuracy_score(y, predictions)
    prec = precision_score(y, predictions, zero_division=0)
    rec  = recall_score(y, predictions, zero_division=0)
    cm   = confusion_matrix(y, predictions)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}  (of flagged traffic, how much was real attack?)")
    print(f"  Recall:    {rec:.4f}  (of all attacks, how many did we catch?)")
    print(f"\n  Confusion Matrix:")
    print( "                   Predicted Benign | Predicted Attack")
    print(f"  Actual Benign:   {cm[0][0]:<16} | {cm[0][1]}")
    print(f"  Actual Attack:   {cm[1][0]:<16} | {cm[1][1]}")

    return threshold, acc, prec, rec

# MAIN
if __name__ == '__main__':

    # ── Canonical schema (driven by legacy/synthetic dataset) ──────
    canonical_features = discover_canonical_features(TRAIN_FILES[0])
    with open(CANONICAL_FEATURES_FILE, 'w') as f:
        json.dump(canonical_features, f)
    print(f"Canonical features saved to {CANONICAL_FEATURES_FILE}")

    # ── Scaler (train data only) ───────────────────────────────────
    scaler = get_scaler(TRAIN_FILES, canonical_features)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler saved to {SCALER_FILE}")

    # ── Build model ────────────────────────────────────────────────
    print("\n--- Phase 1: Building Autoencoder ---")
    input_dim = len(canonical_features)

    input_layer = Input(shape=(input_dim,))

    x = Dense(64)(input_layer)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.2)(x)

    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Dense(8)(x)
    x = BatchNormalization()(x)
    encoded = Activation('linear')(x)

    x = Dense(32)(encoded)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.2)(x)

    decoded = Dense(input_dim, activation='sigmoid')(x)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    autoencoder.summary()

    # ── Train ──────────────────────────────────────────────────────
    print(f"\n--- Phase 2: Training on synthetic data ({STEPS_PER_EPOCH} steps/epoch) ---")
    gen = train_generator(TRAIN_FILES, canonical_features, scaler, BATCH_SIZE)
    autoencoder.fit(gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1)

    # ── Evaluate: synthetic (sanity check) ────────────────────────
    print("\n--- Phase 3a: Sanity Check — Evaluating on SYNTHETIC (train distribution) ---")
    threshold, _, _, _ = evaluate(autoencoder, scaler, canonical_features, TRAIN_FILES[0])

    if TEST_FILE.exists():
        print("\n--- Phase 3b: Real Test — Evaluating on CIC-DDoS2019 (unseen real traffic) ---")
        evaluate(autoencoder, scaler, canonical_features, TEST_FILE, threshold=threshold)
    else:
        print(f"\n[WARNING] {TEST_FILE} not found — skipping real-world evaluation.")
        print("Download it via: kaggle datasets download -d dhoogla/cicddos2019 --unzip")

    autoencoder.save(MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
