



import pandas as pd
import os
import math
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam


labels = pd.read_csv('/datasets/faces/labels.csv')

train_datagen = ImageDataGenerator(rescale=1./255)

train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        seed=12345)


# Define la ruta del conjunto de datos
dataset_path = '/datasets/faces/final_files/'

# Carga los datos
labels = pd.read_csv('/datasets/faces/labels.csv')

# Muestra las primeras 10 caras
plt.figure(figsize=(10, 5))
plt.suptitle('Sample of 10 Faces from the Dataset', fontsize=20)
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title(f'Age: {labels.iloc[i]["real_age"]}')
    plt.axis('off')
    img = Image.open(os.path.join(dataset_path, labels.iloc[i]['file_name']))
    plt.imshow(img)
plt.show()



# Plot the distribution of age
plt.figure(figsize=(10, 5))
plt.title('Age Distribution', fontsize=20)
sns.histplot(data=labels, x='real_age', bins=20, kde=True)
plt.show()


import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

# %%
def load_train(path):
    
    datagen = ImageDataGenerator(
        validation_split= 0.25, 
        rescale=1/255)
        
    train_gen_flow = datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col ='file_name',
        y_col = 'real_age',
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset='training',
        seed=12345
    )

    return train_gen_flow
  

# %%
def load_test(path):
    
    test_datagen = ImageDataGenerator(
        validation_split= 0.25, 
        rescale=1/255)
        
    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col ='file_name',
        y_col = 'real_age',
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset='valodation',
        seed=12345
    )

    return test_gen_flow

 
def create_model(input_shape=(224, 224,3)):
    backbone = ResNet50(
        input_shape= input_shape,
        weights='imagenet', 
        include_top=False
)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))
    
    model.compile(
        loss='mse',
        optimizer= Adam(learning_rate=0.0001),
        metrics=['mae'],
    )
    return model


#
# %%
def train_model(
    model,
    train_data, 
    test_data, 
    batch_size=None, 
    epochs=20, 
    steps_per_epoch=None, 
    validation_steps=None
):
    
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)    
   
    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2
    )
    
    return model



# prepara un script para ejecutarlo en la plataforma GPU

init_str = """
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
"""

import inspect

with open('run_model_on_gpu.py', 'w') as f:
    
    f.write(init_str)
    f.write('\n\n')
        
    for fn_name in [load_train, load_test, create_model, train_model]:
        
        src = inspect.getsource(fn_name)
        f.write(src)
        f.write('\n\n')


# Coloca el resultado de la plataforma GPU como una celda Markdown aquí.

# %%
#60/60 - 9s - loss: 10.3787 - mae: 2.4388
#Test MAE: 2.4388




