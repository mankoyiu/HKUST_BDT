import os
os.environ['PYSPARK_PYTHON'] = 'python3.9'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3.9'
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

first_data = pd.read_csv('firstData.txt', delim_whitespace=True, header=None)
first_data.columns = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing',
                      'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea',
                      'Age', 'Gender', 'Country', 'Infected']

second_data = pd.read_csv('secondData.txt', delim_whitespace=True, header=None)
second_data.columns = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing',
                       'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea',
                       'Age', 'Gender', 'Country']

def preprocess_data(df, is_train=True):
    df_processed = df.copy()
    categorical_features = ['Age', 'Gender', 'Country']
    df_processed['Age'] = df_processed['Age'].astype(str)
    df_processed['Gender'] = df_processed['Gender'].astype(str)
    df_processed['Country'] = df_processed['Country'].astype(str)
    df_processed = pd.get_dummies(df_processed, columns=categorical_features)
    if is_train:
        df_processed['Infected'] = df_processed['Infected'].astype(int)
    return df_processed

train_data = preprocess_data(first_data)
test_data = preprocess_data(second_data, is_train=False)

missing_cols = set(train_data.columns) - set(test_data.columns)
missing_cols.discard('Infected')
for col in missing_cols:
    test_data[col] = 0

test_data = test_data[train_data.columns.drop('Infected')]

X_train = train_data.drop('Infected', axis=1)
y_train = train_data['Infected']

X_test = test_data

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

output_files = ['predicted1.txt', 'predicted2.txt', 'predicted3.txt', 'predicted4.txt', 'predicted5.txt']

models = []

model1 = Sequential()
model1.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model1.add(Dense(32, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))
models.append(model1)

model2 = Sequential()
model2.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
models.append(model2)

model3 = Sequential()
model3.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='tanh'))
model3.add(Dense(32, activation='tanh'))
model3.add(Dense(1, activation='sigmoid'))
models.append(model3)

model4 = Sequential()
model4.add(Dense(256, input_dim=X_train_scaled.shape[1], activation='relu'))
model4.add(Dense(128, activation='relu'))
model4.add(Dense(1, activation='sigmoid'))
models.append(model4)

model5 = Sequential()
model5.add(Dense(16, input_dim=X_train_scaled.shape[1], activation='relu'))
model5.add(Dense(1, activation='sigmoid'))
models.append(model5)

for i, model in enumerate(models):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)
    predictions = model.predict(X_test_scaled)
    binary_predictions = (predictions > 0.5).astype(int)
    with open(output_files[i], 'w') as f:
        for prediction in binary_predictions:
            f.write(f"{prediction[0]}\n")