# -*- coding: utf-8 -*-

# Data preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
import pandas as pd

import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.data.utils import turn_data_into_specified_dtype ,sliding_window
import pandas as pd
import numpy as np

# Model training
from pypots.utils.metrics import calc_mae
from sklearn.model_selection import train_test_split
from pypots.utils.metrics import calc_mse, calc_mae
from pypots.optim import Adam
from pypots.imputation import SAITS
import torch as nn
import optuna
#function to define the training and test data loaders
def wave_data(data,n_steps,rate):

    # Bestimme die Anzahl der Zeitschritte
    total_time_steps = len(data['Time'].unique())
    print("Anzahl der Zeitschritte:", total_time_steps)
    # Entferne ungewollte Spalten (RecordID, Time)
    data= data.drop(['Time'], axis=1)
    # Konvertiere die restlichen Spalten in ein NumPy-Array
    data_numerical = data.to_numpy()
    #Konvertiere in den gewünschten Datentyp (float32)
    data_numerical = data_numerical.astype(np.float32)
    # Zuerst 80% der Daten für Training und Validierung und 20% für das Testset aufteilen
    train_val, test = train_test_split(data_numerical, test_size=0.2, random_state=42)
    # Jetzt 70% von den Trainings- und Validierungsdaten für das Training (0.7 / 0.8 = 0.875),
    # # und 30% der Trainings- und Validierungsdaten für die Validierung aufteilen
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    # Überprüfe die Größen der Teilmengen
    print(f"Trainingsdaten: {len(train)}, Validierungsdaten: {len(val)}, Testdaten: {len(test)}")
    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train)
    val_set_X = scaler.transform(val)
    test_set_X = scaler.transform(test)
    train_X = sliding_window(train_set_X, window_len=n_steps, sliding_len=None)
    val_X = sliding_window(val_set_X, window_len=n_steps, sliding_len=None)
    test_X = sliding_window(test_set_X, window_len=n_steps, sliding_len=None)
    # assemble the final processed data into a dictionary
    processed_dataset = {
        # general info
        "n_steps": n_steps,
        "n_features": train_X.shape[-1],
        "scaler": scaler,
        # train set
        "train_X": train_X,
        # val set
        "val_X": val_X,
        # test set
        "test_X": test_X,
        }
    # hold out ground truth in the original data for evaluation
    train_X_ori = train_X
    val_X_ori = val_X
    test_X_ori = test_X
    # mask values in the train set to keep the same with below validation and test sets
    train_X = mcar(train_X, rate)
    # mask values in the validation set as ground truth
    val_X = mcar(val_X, rate)
    # mask values in the test set as ground truth
    test_X = mcar(test_X, rate)

    processed_dataset["train_X"] = train_X
    processed_dataset["train_X_ori"] = train_X_ori
    processed_dataset["val_X"] = val_X
    processed_dataset["val_X_ori"] = val_X_ori
    processed_dataset["test_X"] = test_X
    processed_dataset["test_X_ori"] = test_X_ori

    print(processed_dataset.keys())
    # assemble the datasets for training
    dataset_for_training = {
        "X": processed_dataset['train_X'],
        "X_ori": processed_dataset['train_X']

    }
    # assemble the datasets for validation
    dataset_for_validating = {
        "X": processed_dataset['val_X'],
        "X_ori": processed_dataset['val_X_ori'],
    }
    # assemble the datasets for test
    dataset_for_testing = {
        "X": processed_dataset['test_X'],
    }
    ## calculate the mask to indicate the ground truth positions in test_X_ori, will be used by metric funcs to evaluate models
    test_X_indicating_mask = np.isnan(processed_dataset['test_X_ori']) ^ np.isnan(processed_dataset['test_X'])
    test_X_ori = np.nan_to_num(processed_dataset['test_X_ori'])  # metric functions do not accpet input with NaNs, hence fill NaNs with 0
    return processed_dataset, dataset_for_training, dataset_for_validating,dataset_for_testing,test_X_indicating_mask,test_X_ori

# initialize the model
def SAITS_model(processed_dataset, n_layers, batch_size,lr):
    saits = SAITS(
        n_steps=processed_dataset['n_steps'],
        n_features=processed_dataset['n_features'],
        n_layers=n_layers,
        d_model=256,
        d_ffn=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        ORT_weight=1,  # you can adjust the weight values of arguments ORT_weight
        # and MIT_weight to make the SAITS model focus more on one task. Usually you can just leave them to the default values, i.e. 1.
        MIT_weight=1,
        batch_size=batch_size,
        # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
        epochs=10000,
        # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
        # You can leave it to defualt as None to disable early stopping.
        patience=None,
        # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
        # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
        optimizer=Adam(lr),
        # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
        # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
        # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
        num_workers=0,
        # just leave it to default as None, PyPOTS will automatically assign the best device for you.
        # Set it as 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices, even parallelly on ['cuda:0', 'cuda:1']
        device=None,
        # set the path for saving tensorboard and trained model files
        saving_path="tutorial_results/imputation/saits",
        # only save the best model after training finished.
        # You can also set it as "better" to save models performing better ever during training.
        model_saving_strategy="best",
    )
    return saits

def train_(model,dataset_for_training,dataset_for_validating,dataset_for_testing,test_X_indicating_mask,test_X_ori):
    
    #print(f"---------------------- Start Training{trial.number} ----------------------")
    # train the model on the training set, and validate it on the validating set to select the best model for testing in the next step
    model.fit(train_set=dataset_for_training, val_set=dataset_for_validating)
    # Modell speichern
    model_path = '/home/elounita/SAITS_/model_4_neu.pypots'

    # Öffne die Datei im Schreibmodus
    with open(model_path, 'wb') as f:
        saits.save(model_path,f)  # übergebe die Datei an die save-Methode
    # the testing stage, impute the originally-missing values and artificially-missing values in the test set
    saits_results = model.predict(dataset_for_testing)
  
    saits_imputation = saits_results["imputation"]

    # calculate mean absolute error on the ground truth (artificially-missing values)
    testing_mse = calc_mse(
        saits_imputation,
        test_X_ori,
        test_X_indicating_mask,
    )
    testing_mae = calc_mae(
        saits_imputation,
        test_X_ori,
        test_X_indicating_mask,
    )
    print(f"Testing mean absolute error: {testing_mae:.4f}")
    print(f"Testing mean absolute error: {testing_mse:.4f}")
    return testing_mae

#def objective(trial):
     #define the configurations
 #   cfg = {
#      'batch_size': trial.suggest_int("batch_size", 32, 128, step=32),
#        'lr': trial.suggest_float("lr", 1e-3, 2e-3, log=True),
 #       'n_steps': trial.suggest_int("n_steps", 48, 144, step=48),
  #      'rate': 0.1,
   #     'n_layers': trial.suggest_int("n_layers", 1, 3),
    #    'device': "cuda" if nn.cuda.is_available() else "cpu"
    #}
    
    #data = pd.read_csv('/home/elounita/SAITS_/FN1_2010-2022_Trkkk_214734.csv', sep=';')
    #processed_data,dataset_for_training,dataset_for_validating,dataset_for_testing,test_X_indicating_mask,test_X_ori=wave_data(data,cfg['n_steps'],cfg['rate'])
    #saits=SAITS_model(processed_data,cfg['n_layers'],cfg['batch_size'],cfg['lr'])
    #train_(trial,saits,dataset_for_training,dataset_for_validating)
    #testing_mae= eval_(trial,saits,dataset_for_testing,test_X_indicating_mask,test_X_ori)
    #return testing_mae

if __name__ == "__main__":
    #creating study with optuna
    #storage = "sqlite:///db.sqlite3" #define the storage
    #sampler = optuna.samplers.TPESampler() #define the sampler
    #study = optuna.create_study(study_name="saitsss", direction='minimize', storage=storage, sampler=sampler, load_if_exists= True)
    #study.optimize(objective, n_trials=8) #define the number of trials
    #print(f"Parameters of the best trial: {study.best_trial}\nValue of the best trial: {study.best_value}")
    lr=1e-3
    batch_size= 32
    n_layers=3
    n_steps=4320
    rate=0.1

    #model 4 144 n_steps n_layers 2
    #model 5 48 n_steps n_layers 2
    #model 6 1440 n_steps n_layers 4
    #model_2_neu
    #lr=1e-3
    #batch_size= 32
    #n_layers=1
    #n_steps=1440
    #rate=0.1


    #model_3_neu
    #lr=1e-3
    #batch_size= 32
    #n_layers=2
    #n_steps=48
    #rate=0.1

    #model_4_neu
    #lr=1e-3
    #batch_size= 64
    #n_layers=4
    #n_steps=48
    #rate=0.1



    data = pd.read_csv('/home/elounita/SAITS_/dein_dateiname.csv', sep=',')
    data = data.replace(',', ';')
    processed_data,dataset_for_training,dataset_for_validating,dataset_for_testing,test_X_indicating_mask,test_X_ori= wave_data(data,n_steps,rate)
    saits=SAITS_model(processed_data,n_layers,batch_size,lr)
    testing_mae= train_(saits,dataset_for_training,dataset_for_validating,dataset_for_testing,test_X_indicating_mask,test_X_ori)

    









