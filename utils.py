import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def generate_sample_data(num_customers=1000, seq_length=10, fraud_rate=0.05):
    
    """
    
    Sample data generation for time series data for fraud detection
    based on customer transactions
    
    """
    data = []
    customer_ids = range(num_customers)
    
    for customer_id in customer_ids:
        # Generate typical behavior pattern for this customer
        typical_merchant = np.random.randint(0, 10)
        typical_transaction_type = np.random.randint(0, 5)
        typical_amount = np.random.gamma(shape=5, scale=20)  # Mean around $100
        typical_time_between = np.random.gamma(shape=2, scale=12)  # Mean around 24 hours
        
        # Generate sequence of transactions
        for seq_id in range(seq_length + 1):  # +1 for the label transaction
            # Normal transaction with some noise
            merchant = typical_merchant if np.random.random() > 0.2 else np.random.randint(0, 10)
            trans_type = typical_transaction_type if np.random.random() > 0.2 else np.random.randint(0, 5)
            amount = np.random.normal(typical_amount, typical_amount * 0.1)
            time_gap = np.random.normal(typical_time_between, typical_time_between * 0.2)
            
            # For some transactions, make them fraudulent
            is_fraud = 0
            if seq_id == seq_length and np.random.random() < fraud_rate:
                is_fraud = 1
                # Fraudulent transactions often have different patterns
                if np.random.random() > 0.5:
                    merchant = np.random.randint(0, 10)
                if np.random.random() > 0.5:
                    trans_type = np.random.randint(0, 5)
                if np.random.random() > 0.7:
                    amount = amount * np.random.uniform(2, 5)  # Much larger amount
                if np.random.random() > 0.7:
                    time_gap = time_gap * np.random.uniform(0.1, 0.5)  # Shorter time gap
            
            data.append([
                customer_id, 
                seq_id, 
                merchant, 
                trans_type, 
                max(0.01, amount),  # Ensure positive amount
                max(0.1, time_gap),  # Ensure positive time gap
                is_fraud
            ])
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['customer_id', 'sequence_id', 'merchant_category', 
                                    'transaction_type', 'amount', 'time_gap', 'fraud'])
    return df


def encode_categorical_features(df, objcols=None):

    # if the user doesnt specify objcols
    if not objcols:
        objcols = df.select_dtypes(include=['object']).columns.to_list()
    
    l_encoders = {}
    for col in objcols:
        encoder = LabelEncoder()
        df[col] = df[col].astype(str)  # Convert all values to string to prevent NaN issues
        df[col] = df[col].replace('nan', 'Missing')  # Replace NaN with a placeholder
        df[col] = encoder.fit_transform(df[col])
        l_encoders[col] = dict(zip(encoder.classes_, range(len(encoder.classes_))))  # Store mapping
    
    return df, l_encoders


def scale_cols(df, scalecols=None):

    if not scalecols:
        scalecols = df[df.columns[df.max(axis=0) > 1]].columns
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[scale_cols])
    
    return scaled_data, scaler