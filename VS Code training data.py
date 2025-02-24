import numpy as np
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# load dataset dari file csv

color_data = pd.read_csv('colors.csv')

x = color_data[['R', 'G', 'B']].values

y = color_data['ColorName'].values

# normalisasi data

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)


# split dataset untuk training dan testing

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# training model ML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# inisialisasi model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train) # training model KNN

# prediksi data
y_pred = knn.predict(x_test,)

# menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi Model : {accuracy * 100: .2f}%')


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:

        break


    #ambil pixel tengah gambar

    height, width, _ = frame.shape

    pixel_center = frame[height//2, width//2]

    pixel_center_scaled = scaler.transform([pixel_center])

    #prediksi warna

    color_pred = knn.predict(pixel_center_scaled)[0]

    cv2.putText(frame, f'color: {color_pred}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break

cap.release()

cv2.destroyAllWindows()

