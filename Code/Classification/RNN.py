# -*- coding: utf-8 -*-
"""final_rnn_video_model.py

This script will load video data, preprocess it, train an RNN model using ConvLSTM2D layers for video classification, 
and save the trained model as a .pkl file.
"""

import os
import cv2
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import ConvLSTM2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.optimizers import Adam

# Function to load video frames from a folder (convert each video into a sequence of frames)
def load_video_frames(folder, num_frames=10):
    video_data = []
    labels = []
    for label in os.listdir(folder):
        print(f"Processing {label} started!")
        path = folder + '/' + label
        for video in os.listdir(path):
            video_frames = []
            video_path = path + '/' + video
            frames = sorted(os.listdir(video_path))[:num_frames]  # Take first `num_frames` frames

            for frame in frames:
                img = cv2.imread(video_path + '/' + frame, cv2.IMREAD_GRAYSCALE)
                new_img = cv2.resize(img, (100, 100))
                video_frames.append(new_img)
            if len(video_frames) == num_frames:
                video_data.append(np.array(video_frames))  # Shape will be (num_frames, 100, 100)
                labels.append(label)
        print(f"Processing {label} ended!")
    return np.array(video_data), labels

# Loading the train and test video data
train_videos, train_labels = load_video_frames('div_dataset/train')
test_videos, test_labels = load_video_frames('div_dataset/test')

# Shuffle the data
random.shuffle(train_videos)
random.shuffle(test_videos)

# Normalize and reshape the data
train_videos = train_videos.astype('float32') / 255.0
test_videos = test_videos.astype('float32') / 255.0

train_videos = train_videos.reshape((-1, 10, 100, 100, 1))  # Shape: (num_samples, time_steps, height, width, channels)
test_videos = test_videos.reshape((-1, 10, 100, 100, 1))    # Shape: (num_samples, time_steps, height, width, channels)

# Encode the labels
le = LabelEncoder()
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)
test_labels_encoded = le.transform(test_labels)

# One-hot encode the labels
train_labels_one_hot = to_categorical(train_labels_encoded)
test_labels_one_hot = to_categorical(test_labels_encoded)

# Define the RNN model using ConvLSTM2D
def create_rnn_model():
    model = Sequential()

    # First ConvLSTM2D layer (spatial + temporal features)
    model.add(ConvLSTM2D(32, (3, 3), padding='same', activation='relu', input_shape=(10, 100, 100, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second ConvLSTM2D layer
    model.add(ConvLSTM2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third ConvLSTM2D layer
    model.add(ConvLSTM2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten the 3D output from ConvLSTM
    model.add(Flatten())

    # Fully connected dense layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer (for classification)
    model.add(Dense(36, activation='softmax'))  # Assuming 36 classes

    return model

# Create the RNN model
model = create_rnn_model()

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
batch_size = 16
epochs = 50
history = model.fit(train_videos, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_videos, test_labels_one_hot))

# Evaluate the model on the test set
model.evaluate(test_videos, test_labels_one_hot)

# Visualize the loss curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'], 'r', linewidth=2.0)
plt.plot(history.history['val_loss'], 'b', linewidth=2.0)
plt.legend(['Training loss', 'Validation loss'], fontsize=15)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

# Visualize the accuracy curves
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'], 'r', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'b', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=15)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)

# Save the trained model as a pickle file
file_name = 'video_rnn_model.pkl'
with open(file_name, 'wb') as outfile:
    pickle.dump(model, outfile)

# Save the model using Keras (optional, in HDF5 format)
model.save('video_rnn_model.h5')

print(f"Model saved as {file_name}")
