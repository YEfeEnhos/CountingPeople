import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers, models, optimizers

def build_mobilenet_ssd(input_shape, num_classes):
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights='imagenet')

    for layer in base_model.layers[:-20]:  # Adjust the number of frozen layers as needed based on experimentation
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x) 
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)

    return model

def fine_tune_mobilenet_ssd(train_data, val_data, num_classes, input_shape=(224, 224, 3), epochs=20, batch_size=32, learning_rate=0.001):
    model = build_mobilenet_ssd(input_shape, num_classes)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_data
    )

    return model, history

# Example usage:
# Assuming train_data and val_data are tf.data.Dataset objects or Keras ImageDataGenerator instances
# num_classes = number of classes in your new dataset

# Adjust the input shape and other parameters as needed
input_shape = (224, 224, 3)
num_classes = 10  # Example number of classes
epochs = 10
batch_size = 32

# Fine-tune the MobileNet-SSD
# model, history = fine_tune_mobilenet_ssd(train_data, val_data, num_classes, input_shape, epochs, batch_size)

# Save the trained model
# model.save('mobilenet_ssd_finetuned.h5')
