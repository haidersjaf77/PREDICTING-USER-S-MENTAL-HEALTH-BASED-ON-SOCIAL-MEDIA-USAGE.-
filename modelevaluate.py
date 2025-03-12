from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Nadam, Adamax
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

'''
evaluates different combinations of optimizers, loss functions, and activation functions for a neural network, 
training it on the data and calculating performance metrics (Accuracy, Precision, Recall, F1 Score) for each combination.
'''

optimizers = ['adam', 'sgd', 'rmsprop', 'nadam', 'adamax']
loss_functions = ['categorical_crossentropy']
activation_functions = ['relu', 'tanh', LeakyReLU(alpha=0.01), 'sigmoid', 'elu']

results_df = pd.DataFrame(columns=['Optimizer', 'Loss Function', 'Activation Function', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

for optimizer in optimizers:
    for loss_func in loss_functions:
        for activation_func in activation_functions:
            model = Sequential([
                Dense(64, activation=activation_func, input_dim=x_train.shape[1]),
                Dense(32, activation=activation_func),
                Dense(target_encoded.shape[1], activation='softmax')  
            ])

            model.compile(
                optimizer=optimizer,
                loss=loss_func,
                metrics=['accuracy']
            )

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test), verbose=0, callbacks=[early_stopping])

            y_pred_prob = model.predict(x_test)
            y_pred = y_pred_prob.argmax(axis=1)  
            y_true = y_test.argmax(axis=1)  

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            results_df = pd.concat([results_df, pd.DataFrame([{
                'Optimizer': optimizer,
                'Loss Function': loss_func,
                'Activation Function': str(activation_func),  
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }])], ignore_index=True)

results_df
