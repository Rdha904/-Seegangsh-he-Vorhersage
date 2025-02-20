# Seegangshöhe Vorhersagen  🌊
Dieses Projekt beschäftigt sich mit der Vorhersage der Seegangshöhe basierend auf historischen Zeitreihendaten. Mithilfe eines Seq2seq LSTM-Modells mit Luong Attention werden Muster in den Daten analysiert, um präzise Prognosen über die Meeresbedingungen zu erstellen.
## 📊 Datenquelle
Die für dieses Projekt verwendeten Daten umfassen historische Messungen von Wellenhöhe, Windgeschwindigkeit und anderen relevanten Parametern. Leider kann der Datensatz aus rechtlichen Gründen nicht in diesem Repository geteilt werden.  

## 🔍 Ziel des Projekts:


Dieses Projekt hat zum Ziel, eine präzise Seegangsvorhersage zu entwickeln. Der Prozess umfasst zwei wesentliche Schritte:

1. **Datenvorverarbeitung**  
   - Umfassende Datenvorbereitung, einschließlich **GAP-Filling** zur Behandlung von fehlenden Werten.

2. **Modelltraining**  
   - Training eines **Seq2Seq LSTM-Modells mit Luong Attention**, um präzise Vorhersagen auf Basis von historischen Zeitreihendaten zu treffen.

## 📈 Ergebnisse 
Die Ergebnisse dieses Projekts, einschließlich der Vorhersagen und Evaluierungen, sind nicht in diesem Repository enthalten. Du kannst die Ergebnisse jedoch selbst generieren, indem du den Code ausführst und den bereitgestellten Datensatz verwendest.  
Für weitere Informationen oder spezifische Fragen zu den Ergebnissen kannst du mich gerne kontaktieren.
## Deskription
Der Code ist wie folgt strukturiert:

SAITS_:
1. saits.py
   - Diese Datei enthält das Training des **Gap-Filling-Modells (SAITS)** auf den Datensatz, um fehlende Werte zu ergänzen. Das Modell wird auf den vorverarbeiteten Daten trainiert und nutzt spezifische Techniken zur Schätzung und Auffüllung der Lücken in den Zeitreihendaten.


2. eval.py  
   - Diese Datei bewertet das **trainierte Gap-Filling-Modell**, um die Konsistenz und Genauigkeit der verwendeten Methode zur Auffüllung fehlender Werte zu überprüfen. 

     
3. visualisierung.py  
   - Diese Datei enthält Funktionen zur **Visualisierung der gefüllten Gaps** im Datensatz. Sie zeigt grafisch, wie das Gap-Filling-Modell fehlende Werte in den Zeitreihendaten ergänzt hat. Mithilfe von Diagrammen und Plots wird die Konsistenz der Gap-Füllung und die Auswirkungen auf die Daten visuell dargestellt.

Model.py: Diese Datei enthält die Implementierung des **Seq2Seq LSTM-Modells mit Luong Attention**. Sie definiert die Architektur des Modells, einschließlich der Encoder- und Decoder-Komponenten sowie des Attention-Mechanismus, der es dem Modell ermöglicht, wichtige Teile der Eingabesequenzen zu fokussieren.

train.py:  Dieses Skript wird verwendet, um das Modell zu trainieren. Es lädt die vorverarbeiteten Daten, konfiguriert das Modell, und führt das Training durch. Am Ende wird mit dem Testdatensatz evaluiert.
