from keras.utils import to_categorical
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras.regularizers import l2
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

# Directories
TRAIN_DIR='images/train'
TEST_DIR='images/test'

# Function for Dataframe
def create_dataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels    

# Training data
train = pd.DataFrame()
train['image'], train['label'] = create_dataframe(TRAIN_DIR)
print("Train Set:", len(train), "images with", len(set(train.label)), "classes.\n")

# Testing data 
test = pd.DataFrame()
test['image'], test['label'] = create_dataframe(TEST_DIR)
print("Test Set:", len(test), "images with", len(set(test.label)), "classes.\n")

# Extract Features from images
def extract_features(images):
    features = []  # Feature list to store extracted features
    for image in tqdm(images):
        try:
            img = load_img(image, color_mode="grayscale")  # Load the image and convert to grayscale
            img = np.array(img)
            features.append(img)
        except Exception as e:
            print(f"Error loading image {image}: {str(e)}")
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

# Train feature variable using training set
train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

# Using train and test data 
x_train = train_features / 255.0   # Normalize pixel values
x_test = test_features / 255.0 

# Label Encoding
le = LabelEncoder()
le.fit(train['label'])
y_train = to_categorical(le.transform(train['label']), num_classes=7)
y_test = to_categorical(le.transform(test['label']), num_classes=7)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Regularization
reg = 0.0005

# Model Architecture
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(reg)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(reg)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(reg)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

# Compile the model
optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])


# Early Stopping and Reduce Learning Rate on Plateau callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6)

# Train the model with data augmentation
history = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=128),
    steps_per_epoch=len(x_train) / 128,
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)
