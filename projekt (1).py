#Importujemy wszystkie potrzebne biblioteki i baze danych do trenowania modelu
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.datasets import mnist

#Wczytujemy baze i dzielimy ją na części do trenowania i testowania
(X_trenowane, Y_trenowane), (X_testowane, Y_testowane) = mnist.load_data()

#Wstępne przetwarzanie danych
X_trenowane = X_trenowane.reshape(X_trenowane.shape[0], 28, 28, 1)
X_testowane = X_testowane.reshape(X_testowane.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

#dodanie tzw. „One hot” kodowania w celu skonwertowania zmiennych na wektory binarne
Y_trenowane = keras.utils.to_categorical(Y_trenowane, num_classes=10)
Y_testowane = keras.utils.to_categorical(Y_testowane, num_classes=10)

#konwersja danych na liczby zmiennoprzecinkowe oraz ich standaryzacja by należały do przedziału [0,1]
X_trenowane = X_trenowane.astype('float32')
X_testowane = X_testowane.astype('float32')
X_trenowane /= 255
X_testowane /= 255

#Tworzymy model
num_classes = 10
model = Sequential()

#dodanie pięciu warstw wejściowych
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) #Żeby zredukować overfitting gdy trenuje
model.add(Flatten())

#Warstwa ukryta
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

#Warstwa wyjściowa
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

#Trenowanie modelu
digitModel = model.fit(X_trenowane, Y_trenowane, batch_size=128, epochs=10, verbose=1,
validation_data=(X_testowane, Y_testowane))
print("--Trening zakończony sukcesem--")

#Zapisujemy model do pliku .h5
model.save('model.h5')
print("Model został zapisany")

#Sprawdzamy skuteczność testu
accuracy_score = model.evaluate(X_testowane, Y_testowane, verbose=0)
print("Skuteczność testu: %.2f%%" % (accuracy_score[1]*100))
