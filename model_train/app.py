import numpy as np
import pandas as pd
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from prometheus_client import start_http_server

import model
import metrics

start_http_server(5000)

path = './dataset.csv'

model_cnn = model.create_model()

X, y = model.read_data(path)

train_X, test_X, train_y, test_y = train_test_split(X,y,random_state=42)

metrics_callback = metrics.ModelMetricsCallback(validation_data=(test_X,test_y))

epochs = 30

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    model_cnn.fit(train_X, train_y, epochs=1, validation_data=(test_X, test_y), callbacks=[metrics_callback])
    metrics.collect_metrics_cpu_memory()
    time.sleep(1)

model_cnn.save('/app/models/model.keras')

