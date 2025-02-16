from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
def lstm_model(x_train, x_validation, y_train, y_validation, lag, filepath='checkpoints/'):
    """Defines and trains the LSTM model."""
    checkpoint_filepath = os.path.join(filepath, 'lstm_model.keras')
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True)
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    callbacks = [earlystopping, checkpoint]
    lstm = Sequential()
    lstm.add(LSTM(512, activation='relu', return_sequences=True, input_shape=(lag, x_train.shape[2])))
    lstm.add(LSTM(512, activation='relu', return_sequences=True))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(256, activation='relu', return_sequences=True))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(128, activation='relu', return_sequences=True))
    lstm.add(LSTM(64, activation='relu'))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    res = lstm.fit(x_train, y_train, epochs=50, batch_size=8, validation_data=(x_validation, y_validation),
                   callbacks=callbacks)

    return res, lstm
