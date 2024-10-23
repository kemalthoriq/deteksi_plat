import numpy as np
from tensorflow.keras import layers, models

def create_model(input_size=(128, 128, 3)):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_size))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Ubah ini sesuai jumlah kelas yang Anda miliki
    return model

# Misalkan Anda memiliki data pelatihan
# train_images = np.random.rand(100, 128, 128, 3)  # Contoh data
# train_labels = np.random.randint(0, 2, 100)  # Contoh label biner

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Latih model dengan data nyata
# model.fit(train_images, train_labels, epochs=10)

# Simpan model setelah pelatihan
model.save('segmentation/model.h5')
print("Model berhasil disimpan.")
