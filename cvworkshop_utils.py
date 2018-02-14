from __future__ import print_function
import os, sys, time, base64, requests, itertools
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import zipfile, shutil
from sys import platform


# Download data
try:
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve

datasets_path = "dataset"

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def save(url, filename):
    urlretrieve(url, filename, reporthook)
    
def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_unless_exists(url, filename, max_retries=3):
    '''Download the file unless it already exists, with retry. Throws if all retries fail.'''
    if os.path.exists(filename):
        print('Reusing locally cached: ', filename)
    else:
        print('Starting download of {} to {}'.format(url, filename))
        retry_cnt = 0
        while True:
            try:
                save(url, filename)
                print('Download completed.')
                return
            except:
                retry_cnt += 1
                if retry_cnt == max_retries:
                    print('Exceeded maximum retry count, aborting.')
                    raise
                print('Failed to download, retrying.')
                time.sleep(np.random.randint(1,10))

def download_animals_dataset(dataset_root = os.path.join(datasets_path, 'Animals')):
    ensure_exists(dataset_root)
    animals_uri = 'https://www.cntk.ai/DataSets/Animals/Animals.zip'
    animals_file = os.path.join(dataset_root, 'Animals.zip')
    download_unless_exists(animals_uri, animals_file)
    if not os.path.exists(os.path.join(dataset_root, 'Test')):
        with zipfile.ZipFile(animals_file) as animals_zip:
            print('Extracting {} to {}'.format(animals_file, dataset_root))
            animals_zip.extractall(path=os.path.join(dataset_root, '..'))
            print('Extraction completed.')
    else:
        print('Reusing previously extracted Animals data.')
        
    return {
        'training_folder': os.path.join(dataset_root, 'Train'),
        'testing_folder': os.path.join(dataset_root, 'Test')
    }
                
def download_resnet_model(model_root = 'models/resnet'):
    ensure_exists(model_root)
    resnet18_model_uri = 'https://www.cntk.ai/Models/ResNet/ResNet_18.model'
    resnet18_model_local = os.path.join(model_root, 'ResNet_18.model')
    download_unless_exists(resnet18_model_uri, resnet18_model_local)
    return resnet18_model_local
    
def download_beverage_models():
    base_folder = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_folder, "models")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if not os.path.exists(os.path.join(model_folder, "models")):
        filename = os.path.join(model_folder, "cntk_models.zip")
        if not os.path.exists(filename):
            url = "https://computervisionworkshop.blob.core.windows.net/models/cntk_models.zip"
            print('Downloading data from ' + url + '...')
            save(url, filename)
        try:
            print('Extracting ' + filename + '...')
            with zipfile.ZipFile(filename) as myzip:
                myzip.extractall(model_folder)
        finally:
            os.remove(filename)
        print('Done.')
    else:
        print('Data already available at ' + model_folder)

def download_beverage_data():
    base_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = os.path.join(base_folder, "dataset")
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    if not os.path.exists(os.path.join(dataset_folder, "Beverages")):
        filename = os.path.join(dataset_folder, "Beverages.zip")
        if not os.path.exists(filename):
            url = "https://computervisionworkshop.blob.core.windows.net/dataset/Beverages.zip"
            print('Downloading data from ' + url + '...')
            save(url, filename)
        try:
            print('Extracting ' + filename + '...')
            with zipfile.ZipFile(filename) as myzip:
                myzip.extractall(dataset_folder)
        finally:
            os.remove(filename)
        print('Done.')
    else:
        print('Data already available at ' + dataset_folder + '/Grocery')


# Plot Metrics
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def classification_report(y_true, y_pred):
    start_time = time.time()

    print(metrics.classification_report(y_true, y_pred))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    class_names = sorted(set(y_true))
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

if __name__ == "__main__":
    download_beverage_data()
    download_beverage_models()
    download_resnet_model()
    download_animals_dataset()
