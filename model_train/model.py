import pandas as pd
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Sequential
from keras.layers import Dense

def read_data(path):
    df = pd.read_csv(path)
    X, y = df.drop(['label'],axis=1), df['label']

    X = X / 255.0 
    X = X.to_numpy().reshape(X.shape[0], 32, 32)

    return X, y

#Model definition
def create_model():
    model = Sequential([
        Input(shape=(32,32,1), name='Input_layer'),
        # Reshape((28,28,1),input_shape=(784,)),
        Conv2D(32, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu',padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(128,activation='relu'),
        BatchNormalization(),
        Dense(25,activation='softmax')
        
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model