'''
L'objectif de ce code est de construire un réseau de neurones Auto-
encodeur via TensorFlow-Keras. Ce réseau doit être construit de façon
à ce qu'on puisse le réutiliser en y incluant nos données et en
y redéfinissant ses paramètres et son objet de sortie souhaitée.

Il est composé des portions suivantes:
1) Importation des données (et échantillonnage)
    1a) Importation des modules
    1b) Établissement des variables d'extraction et de sauvegarde des données
    1c) Détermination des réductions de composantes souhaitées
    1d) Importation et assignation du jeu de données en jeux de tests et d'entraînement
    1e) Échantillonnage (90 000 rangées) des variables d'essaie et de test
2) Construction du réseau de neurones 

Historique:
Code initial: 2019/29/10: Création du code par Guillaume Giroux
Modification 1:
'''

#%%
# 1) Importation des données (et échantillonnage) ================================================================================================
# 1a) Importation des modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt


# 1b) Établissement des variables d'extraction et de sauvegarde des données
sourceFilename = r'K:\27000\_Projet_Information_de_gestion\Analytique\Intelligence Artificielle\ScaledDatabaseOHE-robustSc-20191025.pkl'
networkSavePath = r'K:\27000\_Projet_Information_de_gestion\Analytique\Intelligence Artificielle\AutoencoderNetwork'


# 1c) Détermination des réductions de composantes souhaitées et paramètres
# du réseau neuronal
composantesReduites = [2, 4, 8, 12, 18, 24, 36, 42, 83, 166]
batch_size = 10 # Mettre une puissance de 2 optimiser sur GPU
epochs = 400
loss = 'mean_squared_error'


#%%
# 1d) Importation et assignation du jeu de données en jeux de tests et 
# d'entraînement
f = open(sourceFilename, 'rb')
Xs_train, Xs_test, Y_train, Y_test, scaler, Info = pickle.load(f)
f.close()


#%%
# 1e) Échantillonnage (180 000 rangées) des variables d'essaie et de test
Xs_train = Xs_train.sample(n=1800, random_state=42)
Y_train = Y_train.sample(n=1800, random_state=42)
Xs_test = Xs_test.sample(n=1800, random_state=42)
Y_test = Y_test.sample(n=180000, random_state=42)


print('Fin partie 1')


#%%
# 2) Construction du réseau de neurones ===========================================================================================================================================
# 2a) Construction du réseau de neurones
import time
tic = time.time()
def get_compiled_model(dimensions=100):
    model = keras.Sequential([
        keras.layers.Dense(len(Xs_train.columns), input_shape=(len(Xs_train.columns),), activation='relu'),
        keras.layers.Dense(dimensions, activation='relu'),
        keras.layers.Dense(len(Xs_train.columns))
    ])

    model.compile(optimizer='adam',
                loss=loss,
                metrics=['acc'])
    return model


# 2b) Application du réseau sur chaque choix de réduction de composante
for middleLayer in composantesReduites:
    model = get_compiled_model(dimensions=middleLayer)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, verbose=1, patience=20)
    savePath = networkSavePath + str(middleLayer) + 'dimensions.h5'
    mc = keras.callbacks.ModelCheckpoint(savePath, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    history = model.fit(Xs_train, Xs_train, validation_split=0.15, batch_size=batch_size, epochs=epochs, callbacks=[es, mc])

    toc = time.time()
    print(toc - tic)

    # evaluate the model
    _, train_acc = model.evaluate(Xs_train, Xs_train, verbose=0)
    _, test_acc = model.evaluate(Xs_test, Xs_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot training history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


# %%
