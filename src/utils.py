import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def pc(df1, pca_model_filename="pca_model.pkl"):
    """Applies Principal Component Analysis (PCA) to the DataFrame."""

    pca1 = None  # Initialize pca1 outside the try block
    try:
        with open(pca_model_filename, 'rb') as file:
            pca1 = pickle.load(file)  # Load the pre-fitted PCA model
        print("Loaded pre-fitted PCA model.")
        df = pca1.transform(df1)
        df = pd.DataFrame(df)
    except FileNotFoundError:
        print("No pre-fitted PCA model found. Fitting a new one.")
        pca = PCA()
        pca.fit(df1)
        r = pca.explained_variance_ratio_.cumsum()
        n = 0
        for i in range(len(r)):
            if r[i] <= 0.995:
                n = n + 1
            else:
                break
        pca1 = PCA(n_components=n)
        df = pca1.fit_transform(df1)
        df = pd.DataFrame(df)
    except (EOFError, FileExistsError, OSError) as e:
        print(f"Error loading PCA model from {pca_model_filename}: {e}")
        print("Re-fitting PCA model.")
        pca = PCA()
        pca.fit(df1)
        r = pca.explained_variance_ratio_.cumsum()
        n = 0
        for i in range(len(r)):
            if r[i] <= 0.995:
                n = n + 1
            else:
                break
        pca1 = PCA(n_components=n)
        df = pca1.fit_transform(df1)
        df = pd.DataFrame(df)
    finally:
        if pca1 is not None:
            try:
                # Save the fitted PCA model to a file (overwrite if exists)
                with open(pca_model_filename, 'wb') as file:
                    pickle.dump(pca1, file)
                print("Saved the new PCA model.")
            except (OSError, FileExistsError) as e:
                print(f"Error saving PCA model to {pca_model_filename}: {e}")
        else:
            print("PCA model was not fitted. Not saving.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    g = df.shape[1]
    df = scaler.fit_transform(df.iloc[:, :g])
    df = pd.DataFrame(df)
    df['Close'] = df1['Close']
    cols = ['Close']
    for i in df.columns:
        if (i == 'Close'):
            break
        else:
            cols.append(i)
    df = df.reindex(cols, axis=1)
    df = np.array(df)
    return df
