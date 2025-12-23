import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_samples=20000):
    
    n_benign = n_samples // 2
    n_attack = n_samples - n_benign
    benign_duration = np.random.randint(50000, 60000000, n_benign)
    benign_fwd_pkts = np.random.randint(10, 100, n_benign)
    benign_bwd_pkts = benign_fwd_pkts + np.random.randint(5, 50, n_benign)
    benign_pkt_len = np.random.normal(loc=500, scale=200, size=n_benign)
    benign_pkt_len = np.abs(benign_pkt_len) # Ensure length is non-negative
    benign_iat = np.random.normal(loc=100000, scale=50000, size=n_benign)
    benign_iat = np.abs(benign_iat)
    benign_syn = np.random.choice([0, 1], size=n_benign, p=[0.95, 0.05])
    attack_duration = np.random.randint(100, 10000, n_attack)
    attack_fwd_pkts = np.random.randint(500, 50000, n_attack)
    attack_bwd_pkts = np.random.randint(0, 5, n_attack)
    attack_pkt_len = np.random.normal(loc=1200, scale=10, size=n_attack)
    attack_iat = np.random.exponential(scale=100, size=n_attack) # microseconds
    attack_syn = np.random.choice([0, 1], size=n_attack, p=[0.1, 0.9])

    df_benign = pd.DataFrame({
        'Flow Duration': benign_duration,
        'Total Fwd Packets': benign_fwd_pkts,
        'Total Backward Packets': benign_bwd_pkts,
        'Packet Length Mean': benign_pkt_len,
        'Flow IAT Mean': benign_iat,
        'Fwd Flag Count': benign_syn, 
        'Label': 0 # Label 0 indicates Benign traffic
    })
    
    df_attack = pd.DataFrame({
        'Flow Duration': attack_duration,
        'Total Fwd Packets': attack_fwd_pkts,
        'Total Backward Packets': attack_bwd_pkts,
        'Packet Length Mean': attack_pkt_len,
        'Flow IAT Mean': attack_iat,
        'Fwd Flag Count': attack_syn,
        'Label': 1 # Label 1 indicates DDoS traffic
    })
    
    df = pd.concat([df_benign, df_attack], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    target_samples = 10_000_000 # Define the total dataset size we want
    chunk_size = 1_000_000      # Generate in chunks to prevent RAM overflow
    filename = "dataset.csv"
    
    print(f"Generating {target_samples} synthetic DDoS samples...")
    print(f"Using chunk size of {chunk_size} to manage memory.")
    
    if os.path.exists(filename):
        os.remove(filename)

    total_generated = 0
    
    while total_generated < target_samples:
        current_batch = min(chunk_size, target_samples - total_generated)
        
        df_chunk = generate_synthetic_data(current_batch)
        
        dirty_count = max(1, int(current_batch * 0.001))
        indices = np.random.choice(df_chunk.index, dirty_count, replace=False)
        df_chunk.loc[indices, 'Flow Duration'] = np.nan
        
        mode = 'w' if total_generated == 0 else 'a'
        header = True if total_generated == 0 else False
        
        df_chunk.to_csv(filename, index=False, mode=mode, header=header)
        
        total_generated += current_batch
        print(f"Progress: {total_generated:,} / {target_samples:,} samples saved.")

    print(f"\nSuccess! Full dataset saved as '{filename}'")
