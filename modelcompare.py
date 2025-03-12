'''
Evaluate and compare the performance of different activation functions (ReLU, Tanh, Sigmoid, Softmax) in the neural network by calculating classification metrics (Accuracy, Precision, Recall, F1 Score).
'''

activation_functions = ['relu', 'tanh', 'sigmoid', 'softmax']

results_df = pd.DataFrame(columns=['Activation Function', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

for activation_func in activation_functions:
    model = Sequential([
        Dense(64, input_dim=x_train.shape[1], activation=activation_func),
        Dense(32, activation=activation_func),
        Dense(target_encoded.shape[1], activation='softmax')  
    ])

    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',  
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
        'Activation Function': activation_func,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }])], ignore_index=True)

results_df
