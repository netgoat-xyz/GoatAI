import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os

DATA_FILE = 'dataset.csv'
CHUNK_SIZE = 200_000 
BATCH_SIZE = 512
STEPS_PER_EPOCH = 2000 
EPOCHS = 5

def get_scaler(filename, sample_limit=1_000_000):
    print(f"--- Phase 0: Calibrating Scaler (First {sample_limit} rows) ---")
    scaler = MinMaxScaler()
    total_seen = 0
    
    try:
        for chunk in pd.read_csv(filename, chunksize=CHUNK_SIZE):
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.dropna(inplace=True)
            
            features = chunk.drop('Label', axis=1)
            scaler.partial_fit(features)
            
            total_seen += len(chunk)
            if total_seen >= sample_limit:
                break
        print(f"Scaler calibrated on {total_seen} samples.")
        return scaler
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return None

def train_generator(filename, scaler, batch_size):
    while True:
        for chunk in pd.read_csv(filename, chunksize=CHUNK_SIZE):
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.dropna(inplace=True)
            
            benign_chunk = chunk[chunk['Label'] == 0].drop('Label', axis=1)
            
            if len(benign_chunk) == 0:
                continue
                
            X_scaled = scaler.transform(benign_chunk)
            
            num_batches = len(X_scaled) // batch_size
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                batch = X_scaled[start:end]
                
                yield (batch, batch)

scaler = get_scaler(DATA_FILE)

if scaler:
    print("\n--- Phase 1: Building The STRONG (Autoencoder) ---")

    dummy_chunk = pd.read_csv(DATA_FILE, nrows=1)
    input_dim = len(dummy_chunk.columns) - 1 # Exclude 'Label'
    
    input_layer = Input(shape=(input_dim,))
    
    # --- ENCODER (Compression) ---
    # The model tries to learn non-linear relationships between features.
    # e.g., "High Packet Count" usually correlates with "High Duration".
    x = Dense(64)(input_layer)
    x = BatchNormalization()(x) # Stabilizes learning, allows higher learning rates
    x = Activation('elu')(x)    # ELU handles negative internal values better than ReLU here
    x = Dropout(0.2)(x)         # Prevents memorizing specific rows
    
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    
    # --- THE BOTTLENECK (Latent Space) ---
    # This is the critical component. We force the 64+ inputs into just 8 neurons.
    # To survive this compression, the model MUST learn the "Essence" of benign traffic.
    # Noise and anomalies (like DDoS patterns) cannot be compressed easily.
    x = Dense(8)(x)
    x = BatchNormalization()(x)
    encoded = Activation('linear')(x) 
    
    # --- DECODER (Reconstruction) ---
    # The model attempts to rebuild the original traffic profile from the essence.
    x = Dense(32)(encoded)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.2)(x)
    
    # Output Layer
    # We use Sigmoid because our input data was scaled to [0, 1].
    decoded = Dense(input_dim, activation='sigmoid')(x)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    
    opt = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
    
    autoencoder.summary()
    
    print(f"\n--- Phase 2: Streaming Training ({STEPS_PER_EPOCH} steps/epoch) ---")
    gen = train_generator(DATA_FILE, scaler, BATCH_SIZE)
    
    autoencoder.fit(
        gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        verbose=1
    )
    
    print("\n--- Phase 3: Evaluation & Threshold Setting ---")
    print("Loading a validation chunk containing Mixed traffic (Benign + DDoS)...")
    
    val_chunk = next(pd.read_csv(DATA_FILE, chunksize=200_000))
    val_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    val_chunk.dropna(inplace=True)
    
    X_val = val_chunk.drop('Label', axis=1)
    y_val = val_chunk['Label']
    X_val_scaled = scaler.transform(X_val)
    
    benign_val = X_val_scaled[y_val == 0]
    reconstructions = autoencoder.predict(benign_val)
    train_loss = tf.keras.losses.mse(reconstructions, benign_val)
    
    threshold = np.mean(train_loss) + 2 * np.std(train_loss)
    
    print(f"Calculated Anomaly Threshold: {threshold:.6f}")
    
    all_reconstructions = autoencoder.predict(X_val_scaled)
    all_loss = tf.keras.losses.mse(all_reconstructions, X_val_scaled)
    
    predictions = tf.math.greater(all_loss, threshold)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    
    print("\nResults on Validation Chunk:")
    print(f"Accuracy:  {accuracy_score(y_val, predictions):.4f}")
    print(f"Precision: {precision_score(y_val, predictions):.4f} (How many detected attacks were real?)")
    print(f"Recall:    {recall_score(y_val, predictions):.4f} (How many attacks did we catch?)")
    
    cm = confusion_matrix(y_val, predictions)
    print("\nConfusion Matrix:")
    print("                 Predicted Benign | Predicted Attack")
    print(f"Actual Benign:   {cm[0][0]:<16} | {cm[0][1]}")
    print(f"Actual Attack:   {cm[1][0]:<16} | {cm[1][1]}")
    
    autoencoder.save('ddos_detector_model.keras')
    print("\nModel saved.")
