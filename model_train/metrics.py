from prometheus_client import Gauge
from sklearn.metrics import roc_auc_score, f1_score
from keras.callbacks import Callback
import psutil
import time

accuracy_gauge = Gauge('model_accuracy', 'Model Accuracy')
loss_gauge = Gauge('model_loss', 'Model Loss')
auc_gauge = Gauge('model_auc', 'Model AUC')
f1_gauge = Gauge('model_f1', 'Model F1')

cpu_usage = Gauge('cpu_usage', 'CPU usage percentage')
memory_usage = Gauge('memory_usage', 'Memory usage in bytes')


class ModelMetricsCallback(Callback):
    def __init__(self, validation_data):
        super(ModelMetricsCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
    
        y_pred_proba = self.model.predict(x_val)
        
        auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
        
        y_pred = y_pred_proba.argmax(axis=1)  

        accuracy = logs.get('accuracy')
        loss = logs.get('loss')
        f1 = f1_score(y_val, y_pred, average='weighted')

        accuracy_gauge.set(accuracy)
        auc_gauge.set(auc)
        loss_gauge.set(loss)
        f1_gauge.set(f1)

def collect_metrics_cpu_memory():
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)
    time.sleep(5)
