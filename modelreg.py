'''
Train a regularized neural network model with dropout and L2 regularization, and evaluate its performance on the test set using classification metrics.
'''

model = Sequential([
    Dense(64, activation='relu', input_dim=x_train.shape[1], kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(target_encoded.shape[1], activation='softmax')  
])

model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',  
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test), verbose=2)

y_pred_prob = model.predict(x_test)
y_pred = y_pred_prob.argmax(axis=1)  
y_true = y_test.argmax(axis=1)  

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

results = pd.DataFrame([{
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}])

results



'''
Train a neural network with early stopping to prevent overfitting, and evaluate its performance using classification metrics on the test set.
'''

model = Sequential([
    Dense(64, activation='relu', input_dim=x_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(target_encoded.shape[1], activation='softmax')  
])

model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',  
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test), verbose=2, callbacks=[early_stopping])

y_pred_prob = model.predict(x_test)
y_pred = y_pred_prob.argmax(axis=1)  
y_true = y_test.argmax(axis=1)  

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

results = pd.DataFrame([{
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}])

results