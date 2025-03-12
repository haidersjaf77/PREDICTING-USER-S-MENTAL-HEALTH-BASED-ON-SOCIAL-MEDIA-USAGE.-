'''
Build and train a neural network model for multi-class classification, evaluating its performance using accuracy, precision, recall, and F1 score metrics.
'''

encoder = OneHotEncoder(sparse_output=False)
target_encoded = encoder.fit_transform(target.reshape(-1, 1))

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

x_train, x_test, y_train, y_test = train_test_split(data_scaled, target_encoded, test_size=0.2, random_state=1)
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
