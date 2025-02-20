import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Model import Seq2Seq
import numpy as np
import pandas as pd
import argparse
import wandb 
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_percentage_error



def create_sequences(dataset, input_size, target_size, start_index, end_index, history_size, target_size_steps):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size_steps

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        
        # Eingabesequenz: 48 Zeitschritte mit 2 Features
        data.append(dataset[indices, :input_size])  # Hier nimmst du 48 Zeitschritte und 2 Features

        # Zielsequenz: Die nächsten 24 Zeitschritte mit 1 Zielwert (z.B. Wellenhöhe)
        labels.append(dataset[i:i + target_size_steps, -target_size:])  # Hier nimmst du 24 Zeitschritte und 1 Zielwert

    # Rückgabe des kompletten Datensatzes (ohne Batch-Aufteilung)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)



# Funktion zur Berechnung von RMSE
def calculate_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):

    best_val_loss = float('inf')  # Setze den besten Validierungsverlust auf unendlich
    best_model_state = None  # Hier werden die besten Gewichte gespeichert

    for epoch in range(epochs):
        # Setze das Modell in den Trainingsmodus
        model.train()
        total_train_loss = 0
        total_train_mape = 0
        total_train_r2= 0
        for batch in train_loader:
            src, trg = batch
            optimizer.zero_grad()
            output = model(src, trg)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            
            # Berechne loss für den Trainingsbatch
            total_train_loss += loss.item()

             # Berechne R² für den Trainingsbatch
            r2 = r2_score(trg, output)
            total_train_r2 += r2.item()
            # Berechne MAPE für den Trainingsbatch
            mape = mean_absolute_percentage_error(trg.cpu().numpy(), output.detach().cpu().numpy())
            total_train_mape += mape
  
        # Durchschnittlicher Trainingsverlust und MAPE, R2
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_mape = total_train_mape / len(train_loader)
        avg_train_r2 = total_train_r2 / len(train_loader)


        # Nach jeder Epoche Validierungsverlust und MAPE berechnen
        model.eval()
        total_val_loss = 0
        total_val_mape = 0
        total_val_r2= 0

        with torch.no_grad():
            for val_batch in val_loader:
                val_src, val_trg = val_batch
                val_output = model(val_src, val_trg)
                val_loss = criterion(val_output, val_trg)

                # Berechne loss für den Validierungsbatch

                total_val_loss += val_loss.item()
                 # Berechne R² für den Validierungsbatch
                r2 = r2_score(trg, output)
                total_val_r2 += r2.item()

                # Berechne MAPE für den Validierungsbatch
                val_mape = mean_absolute_percentage_error(val_trg.cpu().numpy(), val_output.cpu().numpy())
                total_val_mape += val_mape.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_mape = total_val_mape / len(val_loader)
        avg_val_r2 = total_val_r2 /  len(train_loader)


        # Modell speichern
        torch.save(model.state_dict(), f"trained_model_epoch_{epoch + 1}.pth")
        
        # Ausgabe des Fortschritts
        print(f"Epoch {epoch + 1}/{epochs}, Train_Loss: {avg_train_loss},Train_MAPE: {avg_train_mape}%, Train_R² : {avg_train_r2},Val_Loss: {avg_val_loss},Val_MAPE: {avg_val_mape}%, Val_R² : {avg_val_r2}")

        # Logge Trainings- und Validierungsverlust und MAPE in Weights & Biases
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                   "train_mape": avg_train_mape, "val_mape": avg_val_mape,
                   "train_R²":avg_train_r2,"val_R²":avg_val_r2})
        # Speichere das beste Modell basierend auf dem Validierungsverlust
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()  # Speichere das beste Modell im RAM

    # Lade das beste Modell in den aktuellen Modellzustand
    model.load_state_dict(best_model_state)

    # Speichere das beste Modell auf der Festplatte
    torch.save(best_model_state, "best_model.pth")
    print("Bestes Modell gespeichert unter 'best_model.pth' und bereit für die Evaluierung")




# Evaluierungsfunktion
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    total_r2 = 0
    total_mape = 0
    total_rmse = 0
    
    with torch.no_grad():  # Kein Gradienten-Tracking während der Evaluierung
        for batch in test_loader:
            src, trg = batch
            output = model(src)
            
            # Berechne den Loss (MSE)
            loss = criterion(output, trg)
            total_loss += loss.item()

            # Berechne R²
            r2 = r2_score(trg, output)
            total_r2 += r2.item()

            # Berechne MAPE
            mape = mean_absolute_percentage_error(trg.cpu().numpy(), output.cpu().numpy())
            total_mape += mape.item()

            # Berechne RMSE
            rmse = calculate_rmse(trg, output)
            total_rmse += rmse.item()

    # Durchschnittlicher Evaluierungsverlust, R², MAPE und RMSE
    avg_loss = total_loss / len(test_loader)
    avg_r2 = total_r2 / len(test_loader)
    avg_mape = total_mape / len(test_loader)
    avg_rmse = total_rmse / len(test_loader)

    # Ausgabe des Fortschritts
    print(f'Evaluation Loss: {avg_loss}, R²: {avg_r2}, MAPE: {avg_mape}, RMSE: {avg_rmse}')
    
    # Verlust, R², MAPE und RMSE in Weights & Biases protokollieren
    wandb.log({"evaluation_loss": avg_loss, "evaluation_r2": avg_r2, "evaluation_mape": avg_mape, "evaluation_rmse": avg_rmse})

 #4. Main-Funktion
def main(args):

    #filled_data= preprocessor.SAITS_()
    data_filled = pd.read_csv(args.path, sep=',')
    data_filled = data_filled.replace(',', ';')
    # Dataset-Aufteilung
    train_size = int(args.train_weight * len(data_filled))  # 70% für Training
    val_size = int(args.val_weight * len(data_filled))   # 15% für Validierung
    # Definiere die Grenzen der verschiedenen Splits
    train_end = train_size
    val_end = train_size + val_size

   

    # Erstelle Trainings-, Validierungs- und Testdaten
    x_train, y_train = create_sequences(data_filled, args.input_features, args.target_feature, 0, train_end, args.history_size, args.target_size)
    x_val, y_val = create_sequences(data_filled, args.input_features, args.target_feature, train_end, val_end, args.history_size, args.target_size)
    x_test, y_test = create_sequences(data_filled, args.input_features, args.target_feature, val_end, None, args.history_size, args.target_size)

    # Ausgabe der Formen der Daten
    print(f"x_train shape: {x_train.shape}")  # Erwartet: (Anzahl Trainingssequenzen, 48, 2)
    print(f"y_train shape: {y_train.shape}")  # Erwartet: (Anzahl Trainingssequenzen, 24)
    print(f"x_val shape: {x_val.shape}")      # Erwartet: (Anzahl Validierungssequenzen, 48, 2)
    print(f"y_val shape: {y_val.shape}")      # Erwartet: (Anzahl Validierungssequenzen, 24)
    print(f"x_test shape: {x_test.shape}")    # Erwartet: (Anzahl Testsequenzen, 48, 2)
    print(f"y_test shape: {y_test.shape}")    # Erwartet: (Anzahl Testsequenzen, 24)

    # Dataloader
    train_loader = DataLoader((x_train, y_train), batch_size= args.batch_size, shuffle=False)
    val_loader = DataLoader((x_val, y_val), batch_size= args.batch_size, shuffle=False)
    test_loader = DataLoader((x_test, y_test), batch_size= args.batch_size, shuffle=False)


    # Modell, Kriterium und Optimierer
    model = Seq2Seq(args.input_size, args.output_size, args.hidden_size,args.n_layers, args.dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # Training
    train_model(model, train_loader, val_loader, criterion, optimizer,args.epochs)

    # Evaluation
    evaluate_model(model, test_loader, criterion)

if __name__ == "__main__":
 # Parameter fuer die Sequenzen
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--input_size', type=int, default=2, help='Anzahl der Eingabefeatures')
    parser.add_argument('--output_size', type=int, default=1, help='Anzahl der Zielwerte')
    parser.add_argument('--history_size', type=int, default=48, help='Anzahl der Zeitschritte in der Vergangenheit')
    parser.add_argument('--target_size', type=int, default=24, help='Anzahl der Zeitschritte in der Zukunft, die vorhergesagt werden sollen.')
    parser.add_argument('--train_weight', type=float, default=0.7)
    parser.add_argument('--val_weight', type=float, default=0.15)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=int, default=3e-4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=0)

    args = parser.parse_args()

    print(args)
    wandb.init()
    main(args)
    wandb.finish()
