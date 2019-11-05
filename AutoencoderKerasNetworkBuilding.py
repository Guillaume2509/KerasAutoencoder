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
2) Préprocession du jeu de données en tenseurs de bonne forme pour le réseau de neuronnes 
3) Construction du réseau de neurones 

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
sourceFilename = r'C:\Users\u3782\Documents\perDiem-Slave\analytique\resultats\ScaledDatabaseOHE-robustSc-20191025.pkl'
networkSavePath = r'C:\Users\u3782\Documents'


# 1c) Détermination des réductions de composantes souhaitées et paramètres
# du réseau neuronal
composantesReduites = 24
batch_size = 30000
epochs = 1
loss = 'mean_squared_error'


#%%
# 1d) Importation et assignation du jeu de données en jeux de tests et 
# d'entraînement
f = open(sourceFilename, 'rb')
Xs_train, Xs_test, Y_train, Y_test, scaler, Info = pickle.load(f)
f.close()


#%%
# 1e) Échantillonnage (90 000 rangées) des variables d'essaie et de test
Xs_train = Xs_train.sample(n=90000, random_state=42)
Y_train = Y_train.sample(n=90000, random_state=42)
Xs_test = Xs_test.sample(n=90000, random_state=42)
Y_test = Y_test.sample(n=90000, random_state=42)


print('Fin partie 1')


#%%
# 2) Préprocession du jeu de données en tenseurs de bonne forme pour le réseau de neuronnes ===================================================================================
'''dataset = tf.data.Dataset.from_tensor_slices((Xs_train.values, Y_train.values))
train_dataset = dataset'''

print('Fin partie 2')


#%%
# 3) Construction du réseau de neurones ===========================================================================================================================================
epochs=20

def get_compiled_model(dimensions=100):
    model = keras.Sequential([
        keras.layers.Dense(len(Xs_train.columns), input_shape=(len(Xs_train.columns),), activation='relu'),
        keras.layers.Dense(dimensions, activation='relu'),
        keras.layers.Dense(len(Xs_train.columns))
    ])

    model.compile(optimizer='adam',
                loss=loss,
                metrics=['accuracy'])
    return model


model = get_compiled_model(dimensions=composantesReduites)
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = keras.callbacks.ModelCheckpoint(networkSavePath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
history = model.fit(Xs_train, Xs_train, validation_split=0.15, batch_size=batch_size, epochs=epochs, callbacks=[es, mc])


#%%
# evaluate the model
_, train_acc = model.evaluate(Xs_train, Xs_train, verbose=0)
_, test_acc = model.evaluate(Xs_test, Xs_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()



#%%
MSEloss = history.history['loss']
accuracy = history.history['accuracy']

print("Test loss" + str(MSEloss))
print("Test accuracy" + str(accuracy))

print('Fin partie 3')

'''# %%
plt.style.use('ggplot')

plt.scatter(composantesReduites, MSEloss[0], color='slateblue')

plt.title('MSE en fonction du nombre de dimensions retenues à epochs=1')
plt.xlabel('Nombre de dimensions retenues')
plt.ylabel('MSE')



plt.show()'''

# %%