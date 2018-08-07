# -*- coding: utf-8 -*-

from keras.models import Model,load_model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import config.config as config
import config.globalvar as var


#用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止。
train_datagen = ImageDataGenerator(

    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './num_1679',
        batch_size=var.BATCH_SIZE,
        target_size=(var.IMAGE,var.IMAGE),
        color_mode='grayscale',
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        './num_1679_validation',
        batch_size=var.BATCH_SIZE,
        target_size=(var.IMAGE,var.IMAGE),
        color_mode='grayscale',
        class_mode='categorical')
image_numbers = train_generator.samples

labels = train_generator.class_indices
print labels
# {'1': 0, '9': 3, '7': 2, '6': 1}
#加载模型
# base_model = VGG16(weights='imagenet',pooling='avg',include_top=False)
# #base_model = InceptionV3(include_top=False,pooling='avg')
#
# #添加几全链接层
# x = base_model.output
# x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
# x = Dense(1024, activation='relu', name='fc1')(x)
# x = Dense(1024, activation='relu', name='fc2')(x)
# x = Dense(1024, activation='relu', name='fc3')(x)
# predictions = Dense(2, activation='softmax')(x)
# # 定义模型
# model = Model(inputs=base_model.input, outputs=predictions)
base_model = load_model(config.get_config("result","model_1"))
x = base_model.output
# x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
x = Dense(var.FILTER_4, activation='relu', name='fc1')(x)
predictions = Dense(var.CATEGORY_4, activation='softmax',name='dense_x')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for i, layer in enumerate(model.layers):
   print(i, layer.name,layer.output_shape)

#
# for i, layer in enumerate(model.layers):
#    print(i, layer.name,layer.output_shape)
# #冻结
# for layer in base_model.layers:
#     layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

#model.fit_generator(train_generator,steps_per_epoch=image_numbers)
# train the model on the new data for a few epochs
#model.fit_generator(train_generator,steps_per_epoch=image_numbers)
model.fit_generator(
        train_generator,
        steps_per_epoch=var.STEPS_PER_EPOCH,
        epochs=var.EPOCHS,
        validation_data=validation_generator,
        validation_steps=var.VALIDATION_STEPS)

#model.save('my_model.h5')
model.save(config.get_config("result","model"))
print train_generator
#score = model.evaluate(train_generator, validation_generator, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

