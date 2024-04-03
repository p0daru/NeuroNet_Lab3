import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Тренувати нову модель чи завантажувати існуючу 
train_new_model = True

if train_new_model:
    # Завантаження набору даних MNIST та розділення на тренувальний і тестувальний набори
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Нормалізація даних (приведення до діапазону від 0 до 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Створення моделі нейронної мережі
    # Вхідний шар для плоского зображення 28x28 пікселів
    # Два приховані шари з 128 нейронами та функцією активації ReLU
    # Вихідний шар з 10 нейронами (відповідає 10 цифрам) та функцією активації softmax
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) 
    model.add(tf.keras.layers.Dense(units=128, activation="relu"))
    model.add(tf.keras.layers.Dense(units=128, activation="relu"))
    model.add(tf.keras.layers.Dense(units=10, activation="softmax"))


    # Компіляція та оптимізація моделі
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Тренування моделі
    model.fit(X_train, y_train, epochs=5)

    # Оцінка моделі
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Збереження моделі
    model.save('handwritten_digits.keras')
else:
    # Завантажити існуючу модель
    model = tf.keras.models.load_model('handwritten_digits.keras')

# Завантаження кастомних зображень та їх передбачення
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0] # зчитування зображення та конвертація у чорно-білий формат
        img = np.invert(np.array([img])) # інверсія кольорів та створення numpy масиву
        prediction = model.predict(img) # передбачення
        print("The number is probably a {}".format(np.argmax(prediction))) # argmax – який з нейронів має найвищу активацію (highest activation)
        plt.imshow(img[0], cmap=plt.cm.binary) # відображення зображення
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1

