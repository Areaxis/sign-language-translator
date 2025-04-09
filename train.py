from glob import glob
import matplotlib.pyplot as plt
from keras.src.applications import InceptionV3
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'C:/Users/harsh/PycharmProjects/SignLangConv/Datasets/train'
valid_path = 'C:/Users/harsh/PycharmProjects/SignLangConv/Datasets/test'

inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in inception.layers:
    layer.trainable = False

folders = glob('C:/Users/harsh/PycharmProjects/SignLangConv/Datasets/train/*')

x = Flatten()(inception.output)

prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=inception.input, outputs=prediction)

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)


training_set = train_datagen.flow_from_directory('C:/Users/harsh/PycharmProjects/SignLangConv/Datasets/train',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('C:/Users/harsh/PycharmProjects/SignLangConv/Datasets/test',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')

r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=10,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

model.save('C:/Users/harsh/PycharmProjects/SignLangConv/sign.h5')
