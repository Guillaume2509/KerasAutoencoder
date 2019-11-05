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

# %%
plt.style.use('ggplot')

plt.scatter(composantesReduites, MSEloss[0], color='slateblue')

plt.title('MSE en fonction du nombre de dimensions retenues à epochs=1')
plt.xlabel('Nombre de dimensions retenues')
plt.ylabel('MSE')



plt.show()

# %%