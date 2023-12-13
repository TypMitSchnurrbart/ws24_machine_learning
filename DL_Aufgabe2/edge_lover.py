import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# a) Generation of the data
def generate_dataset(num_samples=1000):
    images = []
    labels = [] 

    for _ in range(num_samples):
        image = np.zeros((50, 50, 1), dtype=np.uint8)
        num_bars = 10
        is_vertical = np.random.randint(2)  # 0 for horizontal, 1 for vertical

        for _ in range(num_bars):
            bar_length = 10
            x_start = np.random.randint(41)  # Random starting position
            y_start = np.random.randint(41) if is_vertical else np.random.randint(41 - bar_length)

            if is_vertical:
                image[y_start:y_start + bar_length, x_start, 0] = 255
            else:
                image[y_start, x_start:x_start + bar_length, 0] = 255

        images.append(image)
        labels.append(is_vertical)


    # Show picture to check
    plt.imshow(images[0], cmap='gray')
    plt.title(f"Sample Picture Label: {labels[0]}")
    plt.axis('off')
    plt.show()


    return np.array(images), np.array(labels)

# b) Build the simplest possible CNN
def build_cnn():
    model = Sequential()
    model.add(Conv2D(1, (5, 5), input_shape=(50, 50, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    return model

# Generate datasets
train_images, train_labels = generate_dataset(1000)
val_images, val_labels = generate_dataset(1000)

# Build and compile the model
model = build_cnn()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=20, validation_data=(val_images, val_labels))


plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# c) Visualize the learned kernel
weights = model.get_weights()[0]
plt.figure(figsize=(5, 5))
plt.imshow(weights[:, :, 0, 0], cmap='gray', interpolation='nearest')
plt.title('Learned Kernel')
plt.show()

