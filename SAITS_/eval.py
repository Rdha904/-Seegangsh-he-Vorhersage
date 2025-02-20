import torch as nn
from pypots.imputation import SAITS
import numpy as np
import pandas as pd
from pypots.data.utils import turn_data_into_specified_dtype ,sliding_window
from sklearn.preprocessing import StandardScaler


# Schritt 2: Bereite deinen Datensatz vor
# Angenommen, df ist dein DataFrame mit den fehlenden Werten
df = pd.read_csv('/pfad/datensatz/', sep=',')
print(df.columns)

n_steps=48
# Bestimme die Anzahl der Zeitschritte
total_time_steps = len(df['Time'].unique())
print("Anzahl der Zeitschritte:", total_time_steps)
time_column = df['Time']  # Speichere die Time-Spalte separat nn


# Entferne ungewollte Spalten (RecordID, Time)
data= df.drop(['Time'], axis=1)
# Konvertiere die restlichen Spalten in ein NumPy-Array
data_numerical = data.to_numpy()
#Konvertiere in den gewünschten Datentyp (float32)
data_numerical = data_numerical.astype(np.float32)
# Überprüfe die Größen der Teilmengen
scaler = StandardScaler()
data_numerical = scaler.fit_transform(data_numerical)    
data_numerical = sliding_window(data_numerical, window_len=n_steps, sliding_len=None)
print(data_numerical.shape)
# assemble the final processed data into a dictionary
processed_dataset = {
    "n_steps": n_steps,
    "n_features": data_numerical.shape[-1],
    "data_numerical": data_numerical,
    }


print(processed_dataset.keys())
from pypots.imputation.saits import SAITS

n_steps=processed_dataset['n_steps']
n_features=processed_dataset['n_features']
n_layers=1
d_model=256
d_ffn=128
n_heads=4
d_k=64
d_v=64
# Initialisiere das Modell mit denselben Parametern
model = SAITS(n_steps, n_features, n_layers, d_model, n_heads, d_k, d_v, d_ffn)

# Lade das vortrainierte Modell
model_path = '/home/elounita/SAITS_/model_4_neu.pypots'  # Pfad zu deinem Modell
model.load(model_path)

# Setze das Modell in den Evaluationsmodus
test_data = {
    "X": processed_dataset['data_numerical']  # Testdaten als Dictionary mit Schlüssel 'X'
}
# Bereite die Testdaten vor (ersetze dies mit deinen echten Daten)
#test_data = processed_dataset['data_numerical']  # Beispiel für Testdaten
has_nan = np.isnan(test_data['X']).any()
if has_nan:
    print("Es gibt noch NaN-Werte im imputierten Datensatz.")
else:
    print("Keine NaN-Werte im imputierten Datensatz.")
# Impute die fehlenden Werte
imputed_data = model.impute(test_data, file_type='array-like')
# Überprüfe auf NaN-Werte im NumPy-Array
has_nan = np.isnan(imputed_data).any()
if has_nan:
    print("Es gibt noch NaN-Werte im imputierten Datensatz.")
else:
    print("Keine NaN-Werte im imputierten Datensatz.")


print(imputed_data.shape)
# Form von (Num_samples, n_steps, n_features) zu (Num_samples * n_steps, n_features) umwandeln
imputed_data_2d = imputed_data.reshape(-1, imputed_data.shape[-1])
print(f"Imputed Data Shape (2D): {imputed_data_2d.shape}")

# Erstelle einen DataFrame aus dem 2D-Array
df_imputed = pd.DataFrame(imputed_data_2d, columns=data.columns)

# Schritt 12: Die ursprüngliche `Time`-Spalte wieder einfügen
# Stelle sicher, dass die Länge der `time_column` der Anzahl der Zeilen in `df_imputed` entspricht
df_imputed.insert(0, 'Time', time_column.iloc[:len(df_imputed)].values)
# Speichere den DataFrame als CSV
output_path = "/home/elounita/SAITS_/imputed_data_2_4_MODEL.csv"
data= df_imputed.to_csv(output_path, index=False, sep=';')
# Falls timesteps ein DataFrame ist

print(f"Die imputierten Daten wurden erfolgreich als CSV gespeichert unter: {output_path}")

# Jetzt kannst du die imputed_data verwenden
# print(imputed_data)
# Schritt 3: Imputiere fehlende Werte
# Schritt 4: Konvertiere die imputierten Daten zurück in ein DataFrame (optional)
# df_imputed = pd.DataFrame(imputed_data, columns=df.columns)

# Jetzt enthält df_imputed die imputierten Werte
