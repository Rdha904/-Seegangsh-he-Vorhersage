import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Schritt 1: Lade deinen Datensatz (ersetze den Pfad entsprechend)
df = pd.read_csv('/home/elounita/SAITS_/imputed_data_2.csv', sep=';')

# Schritt 2: Konvertiere die 'Time'-Spalte in das Datetime-Format
df['Time'] = pd.to_datetime(df['Time'])

# Schritt 3: Filtere die Daten für den Zeitraum von Januar 2016 bis Januar 2017
start_date = '2016-01-21'
end_date = '2016-05-15'
df_filtered = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)]

# Schritt 4: Wähle die zu visualisierenden Spalten aus (z.B. 'Time' und 'Feature1')
feature_column = 'VHM0'  # Ersetze dies durch den tatsächlichen Namen deiner Feature-Spalte

# Schritt 5: Erstelle den Lineplot mit Seaborn
plt.figure(figsize=(14, 6))  # Größe des Diagramms
sns.lineplot(data=df_filtered, x='Time', y=feature_column, color='blue')

# Schritt 6: Formatierung des Plots
plt.xlabel('Zeit')
plt.ylabel('Feature-Wert')
plt.xticks(rotation=45)  # Drehung der x-Achsen-Beschriftung, falls nötig
plt.grid(True)

# Schritt 7: Speichere den Plot als Bilddatei
plt.savefig('/home/elounita/SAITS_/_____saits.png')  # Hier den gewünschten Pfad angeben

# Schritt 8: Zeige den Plot an
plt.show()
