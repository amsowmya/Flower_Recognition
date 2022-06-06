from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

classifier = Sequential()

classifier.add(Conv2D(32, (3,3), input_shape=(64, 64, 3), activation='relu', padding='same'))
classifier.add(Conv2D(32, (3,3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(64, (3,3), activation='relu', padding='same'))
classifier.add(Conv2D(64, (3,3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(64, (3,3), activation='relu', padding='same'))
classifier.add(Conv2D(64, (3,3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(128, activation='relu'))

classifier.add(Dense(5, activation='softmax'))

opt = Adam(learning_rate=0.001, decay=0.001)
classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# PART 2:

from com_utils.utils import unzipFile, splitTrainTestValidFolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator

unzipFile('flowers.zip')
splitTrainTestValidFolders('flowers')

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('output/train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

validation_set = test_datagen.flow_from_directory('output/val',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='categorical')

model = classifier.fit_generator(training_set,
                                 epochs=100,
                                 validation_data=validation_set,
                                 validation_steps=2000)

classifier.save('saved_models/model.h5')
classifier.save_weights('saved_models/classifier_weights.h5')
print('Saved model to disk')

# PART 3: Check Model Performance on Test Data

# Re-initializing the test data generator with shuffel=False to create the confusion matrix
# test_set = test_datagen.flow_from_directory('output/test',
#                                             target_size=(64, 64),
#                                             batch_size=32,
#                                             shuffle=False,
#                                             class_mode='categorical')

# Predict the whole generator to get prediction