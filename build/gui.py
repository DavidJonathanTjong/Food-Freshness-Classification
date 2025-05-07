from pathlib import Path
# from tkinter import *
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, font
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, SimpleRNN, LSTM, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import sys

def get_asset_path():
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) /'assets' /'frame0'
    return Path(__file__).parent /'assets' /'frame0'

def get_data_path():
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) /'assets' /'image_collection'
    return Path(__file__).parent /'assets' /'image_collection'

ASSETS_PATH = get_asset_path()
DATASET_PATH = get_data_path()

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def relative_to_datasets(path: str) -> Path:
    return DATASET_PATH / Path(path)

image = None
brightness_level = 1.0
model_choices = 0
model = None

def openFile():
    filepath = filedialog.askopenfilename(filetypes=(("Image files", "*.jpg *.png *.jpeg"),("All files", "*.*")))
    print(filepath)
    if filepath: #jika ada memasukan gambar
        img = Image.open(filepath)  #membuka gambar
        img = img.resize((275,242)) # rezise ukuran 
        saveImage(img) #save image saat ini
        changeImageBox(img) #ganti image di canvas
        namaFile = Path(filepath).name #ambil nama file
        canvas.itemconfig(nama_image, text=namaFile)# update nama file
        canvas.itemconfig(ukuran_image, text="275 x 242")# update ukuran file

def saveImage(imageSekarang):
    global image
    image = imageSekarang

def changeImageBox(myImage):
    curentImage = ImageTk.PhotoImage(myImage) #simpan hasil image saat ini
    canvas.itemconfig(image_6, image=curentImage)  # Ganti gambar di canvas
    canvas.image_6 = curentImage

def resetPrediction():
    global image_image_6 # mengambil variabel image
    image_image_6 = PhotoImage(file=relative_to_assets("image_6.png")) #mendefinisikan ulang image awal
    canvas.itemconfig(image_6, image=image_image_6) # mengupdate canvas
    global image
    image = None
    # update penjelasan prediksi
    canvas.itemconfig(nama_image, text="Nama foto")
    canvas.itemconfig(akurasi, text="Akurasi")
    canvas.itemconfig(ukuran_image, text="Ukuran")
    canvas.itemconfig(kesimpulan, text="Kesimpulan")
    # update penjelasan model
    canvas.itemconfig(nama_model, text="Model")
    canvas.itemconfig(akurasi_model, text="Akurasi Model")
    canvas.itemconfig(recall, text="Recall")
    canvas.itemconfig(precision, text="Precision")
    canvas.itemconfig(penjelasan_model, text="Penjelasan")

# augmentasi
def brightness_up():
    global image, brightness_level
    if image:
        brightness_level += 0.1  # naikkan brightness 10%
        enhancer = ImageEnhance.Brightness(image)
        enhanced_image = enhancer.enhance(brightness_level)
        changeImageBox(enhanced_image)
        print(f"Brightness increased to {brightness_level:.1f}")

def brightness_down():
    global image, brightness_level
    if image:
        brightness_level = max(0.1, brightness_level - 0.1)  # turunkan brightness 10%, dengan minimum brightness 0.1
        enhancer = ImageEnhance.Brightness(image)
        enhanced_image = enhancer.enhance(brightness_level)
        changeImageBox(enhanced_image)
        print(f"Brightness decreased to {brightness_level:.1f}")

# model
def model_rf():
    global model_choices 
    model_choices = 1

    # Image generator untuk preprocessing
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Ambil data dari generator
    x_train, y_train = next(train_generator)
    x_val, y_val = next(val_generator)

    # Flatten gambar jadi vektor 1D
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train_flat, y_train)

    # Prediksi untuk validasi
    y_pred = rf_model.predict(x_val_flat)

    # Evaluasi
    accuracy = accuracy_score(y_val, y_pred)
    precision_val = precision_score(y_val, y_pred)
    recall_val = recall_score(y_val, y_pred)

    # Update penjelasan model
    canvas.itemconfig(nama_model, text="Model Random Forest")
    canvas.itemconfig(akurasi_model, text=f"Akurasi Model: {accuracy:.2f}")
    canvas.itemconfig(recall, text=f"Recall: {recall_val:.2f}")
    canvas.itemconfig(precision, text=f"Precision: {precision_val:.2f}")
    canvas.itemconfig(penjelasan_model, text="Model memanfaatkan fitur citra secara langsung.")
    
    global model 
    model = rf_model

def model_knn():
    global model_choices 
    model_choices = 2
    # from keras.preprocessing import image as keras_image 
    # Image generator untuk preprocessing
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,  
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Ambil data dari generator
    x_train, y_train = next(train_generator)
    x_val, y_val = next(val_generator)

    # Flatten gambar jadi vektor 1D
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)

    # KNN Classifier
    knn_model = KNeighborsClassifier(n_neighbors=5) 
    knn_model.fit(x_train_flat, y_train)

    # Prediksi model
    y_pred = knn_model.predict(x_val_flat)

    # Evaluasi
    accuracy = accuracy_score(y_val, y_pred)
    precision_val = precision_score(y_val, y_pred)
    recall_val = recall_score(y_val, y_pred)

    # Update penjelasan model
    canvas.itemconfig(nama_model, text="Model KNN")
    canvas.itemconfig(akurasi_model, text=f"Akurasi Model: {accuracy:.2f}")
    canvas.itemconfig(recall, text=f"Recall: {recall_val:.2f}")
    canvas.itemconfig(precision, text=f"Precision: {precision_val:.2f}")
    canvas.itemconfig(penjelasan_model, text="Model knn secara langsung.")

    global model 
    model = knn_model

def model_decisionTree():
    global model_choices 
    model_choices = 3
    
    # Image generator untuk preprocessing
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Ambil data dari generator
    x_train, y_train = next(train_generator)
    x_val, y_val = next(val_generator)

    # Flatten gambar jadi vektor 1D
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)

    # Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(x_train_flat, y_train)
    
    # Prediksi
    y_pred = dt_model.predict(x_val_flat)

    # Evaluasi
    accuracy = accuracy_score(y_val, y_pred)
    precision_val = precision_score(y_val, y_pred)
    recall_val = recall_score(y_val, y_pred)

    # Update penjelasan model
    canvas.itemconfig(nama_model, text="Model Decision Tree")
    canvas.itemconfig(akurasi_model, text=f"Akurasi Model: {accuracy:.2f}")
    canvas.itemconfig(recall, text=f"Recall: {recall_val:.2f}")
    canvas.itemconfig(precision, text=f"Precision: {precision_val:.2f}")
    canvas.itemconfig(penjelasan_model, text="Model DT secara langsung.")

    global model 
    model = dt_model

def model_svm():
    global model_choices 
    model_choices = 4
    
    # Image generator untuk preprocessing
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Ambil data dari generator
    x_train, y_train = next(train_generator)
    x_val, y_val = next(val_generator)

    # Flatten gambar jadi vektor 1D
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)

    # SVM Classifier
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(x_train_flat, y_train)

    # Prediksi
    y_pred = svm_model.predict(x_val_flat)

    # Evaluasi
    accuracy = accuracy_score(y_val, y_pred)
    precision_val = precision_score(y_val, y_pred)
    recall_val = recall_score(y_val, y_pred)

    # Update penjelasan model
    canvas.itemconfig(nama_model, text="Model SVM")
    canvas.itemconfig(akurasi_model, text=f"Akurasi Model: {accuracy:.2f}")
    canvas.itemconfig(recall, text=f"Recall: {recall_val:.2f}")
    canvas.itemconfig(precision, text=f"Precision: {precision_val:.2f}")
    canvas.itemconfig(penjelasan_model, text="Model SVM secara langsung.")

    global model 
    model = svm_model

def model_naivebayes():
    global model_choices 
    model_choices = 5
    
    # Image generator untuk preprocessing
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Ambil data dari generator
    x_train, y_train = next(train_generator)
    x_val, y_val = next(val_generator)

    # Flatten gambar jadi vektor 1D
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)

    # Naive Bayes Classifier
    nb_model = GaussianNB()
    nb_model.fit(x_train_flat, y_train)

    # Prediksi
    y_pred = nb_model.predict(x_val_flat)

    # Evaluasi
    accuracy = accuracy_score(y_val, y_pred)
    precision_val = precision_score(y_val, y_pred)
    recall_val = recall_score(y_val, y_pred)

    # Update penjelasan model
    canvas.itemconfig(nama_model, text="Model NaiveBayes")
    canvas.itemconfig(akurasi_model, text=f"Akurasi Model: {accuracy:.2f}")
    canvas.itemconfig(recall, text=f"Recall: {recall_val:.2f}")
    canvas.itemconfig(precision, text=f"Precision: {precision_val:.2f}")
    canvas.itemconfig(penjelasan_model, text="Model NaiveBayes secara langsung.")
    
    global model 
    model = nb_model

def model_logisticregression():
    global model_choices 
    model_choices = 6
    
    # Image generator untuk preprocessing
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Ambil data dari generator
    x_train, y_train = next(train_generator)
    x_val, y_val = next(val_generator)

    # Flatten gambar jadi vektor 1D
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)

    # Logistic Regression Classifier
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(x_train_flat, y_train)

    # Prediksi
    y_pred = lr_model.predict(x_val_flat)

    # Evaluasi
    accuracy = accuracy_score(y_val, y_pred)
    precision_val = precision_score(y_val, y_pred)
    recall_val = recall_score(y_val, y_pred)

    # Update penjelasan model
    canvas.itemconfig(nama_model, text="Model LR")
    canvas.itemconfig(akurasi_model, text=f"Akurasi Model: {accuracy:.2f}")
    canvas.itemconfig(recall, text=f"Recall: {recall_val:.2f}")
    canvas.itemconfig(precision, text=f"Precision: {precision_val:.2f}")
    canvas.itemconfig(penjelasan_model, text="Model LR secara langsung.")

    global model 
    model = lr_model

def model_cnn():
    global model_choices 
    model_choices = 7

    # Augmentasi dan preprocessing
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=8,
        class_mode='binary',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=8,
        class_mode='binary',
        subset='validation'
    )

    model_cnn = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model_cnn.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model_cnn.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator
    )

    # Evaluasi model
    val_generator.reset()
    y_pred_prob = model_cnn.predict(val_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = val_generator.classes

    report = classification_report(y_true, y_pred, output_dict=True)
    acc = report['accuracy']
    recall_score = report['1']['recall']
    precision_score = report['1']['precision']

    # Update penjelasan model
    canvas.itemconfig(nama_model, text="Model CNN")
    canvas.itemconfig(akurasi_model, text=f"Akurasi Model: {acc:.2f}")
    canvas.itemconfig(recall, text=f"Recall: {recall_score:.2f}")
    canvas.itemconfig(precision, text=f"Precision: {precision_score:.2f}")
    canvas.itemconfig(penjelasan_model, text="Model CNN diproses deep learning")
    
    global model 
    model = model_cnn

def model_rnn():
    global model_choices 
    model_choices = 8
    
    # Image generator untuk preprocessing
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Ambil data dari generator
    x_train, y_train = next(train_generator)
    x_val, y_val = next(val_generator)

    # Bentuk input RNN: treat each row (150) as time step, and each row has 150*3 features (RGB channels)
    x_train_rnn = x_train.reshape((x_train.shape[0], 150, 150*3))
    x_val_rnn = x_val.reshape((x_val.shape[0], 150, 150*3))

    # Bangun model RNN
    model_rnn = Sequential([
        SimpleRNN(64, input_shape=(150, 150*3), activation='tanh'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model_rnn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Latih model
    model_rnn.fit(x_train_rnn, y_train, epochs=10, validation_data=(x_val_rnn, y_val), verbose=2)

    # Prediksi
    y_prob = model_rnn.predict(x_val_rnn).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # Evaluasi
    accuracy = accuracy_score(y_val, y_pred)
    precision_val = precision_score(y_val, y_pred)
    recall_val = recall_score(y_val, y_pred)

    # Update penjelasan model
    canvas.itemconfig(nama_model, text="Model RNN")
    canvas.itemconfig(akurasi_model, text=f"Akurasi Model: {accuracy:.2f}")
    canvas.itemconfig(recall, text=f"Recall: {recall_val:.2f}")
    canvas.itemconfig(precision, text=f"Precision: {precision_val:.2f}")
    canvas.itemconfig(penjelasan_model, text="Model RNN secara langsung.")
    
    global model 
    model = model_rnn

def model_lstm():
    global model_choices 
    model_choices = 9
    
    # Image generator untuk preprocessing
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=1000,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Ambil data dari generator
    x_train, y_train = next(train_generator)
    x_val, y_val = next(val_generator)

    # Bentuk input LSTM: treat each row (150) as time step, and each row has 150*3 features (RGB channels)
    x_train_lstm = x_train.reshape((x_train.shape[0], 150, 150*3))
    x_val_lstm = x_val.reshape((x_val.shape[0], 150, 150*3))

    # Bangun model LSTM
    model_lstm = Sequential([
        LSTM(64, input_shape=(150, 150*3), activation='tanh'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Latih model
    model_lstm.fit(x_train_lstm, y_train, epochs=10, validation_data=(x_val_lstm, y_val), verbose=2)

    # Prediksi
    y_prob = model_lstm.predict(x_val_lstm).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # Evaluasi
    accuracy = accuracy_score(y_val, y_pred)
    precision_val = precision_score(y_val, y_pred)
    recall_val = recall_score(y_val, y_pred)

    # Update penjelasan model
    canvas.itemconfig(nama_model, text="Model LSTM")
    canvas.itemconfig(akurasi_model, text=f"Akurasi Model: {accuracy:.2f}")
    canvas.itemconfig(recall, text=f"Recall: {recall_val:.2f}")
    canvas.itemconfig(precision, text=f"Precision: {precision_val:.2f}")
    canvas.itemconfig(penjelasan_model, text="Model LSTM secara langsung.")
    
    global model 
    model = model_lstm

def model_mobileNet():
    global model_choices 
    model_choices = 10

    from tensorflow.keras.models import Model
    
    # Image generator untuk preprocessing
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

    # Ukuran input yang sesuai MobileNetV2
    input_size = (224, 224)

    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=input_size,
        batch_size=32,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=input_size,
        batch_size=32,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Load MobileNetV2 base model tanpa top layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze layer

    # Tambahkan classifier di atas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model_mobileNet = Model(inputs=base_model.input, outputs=predictions)

    # Compile model
    model_mobileNet.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Latih model
    model_mobileNet.fit(train_generator, epochs=10, validation_data=val_generator, verbose=2)

    # Evaluasi model
    # Ambil semua data dari validation generator
    val_generator.reset()
    x_val, y_val = [], []
    for _ in range(len(val_generator)):
        x_batch, y_batch = next(val_generator)
        x_val.extend(x_batch)
        y_val.extend(y_batch)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    # Prediksi dan evaluasi model
    y_prob = model_mobileNet.predict(x_val).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # Evaluasi
    accuracy = accuracy_score(y_val, y_pred)
    precision_val = precision_score(y_val, y_pred)
    recall_val = recall_score(y_val, y_pred)

    # Update penjelasan model
    canvas.itemconfig(nama_model, text="Model MobileNetV2")
    canvas.itemconfig(akurasi_model, text=f"Akurasi Model: {accuracy:.2f}")
    canvas.itemconfig(recall, text=f"Recall: {recall_val:.2f}")
    canvas.itemconfig(precision, text=f"Precision: {precision_val:.2f}")
    canvas.itemconfig(penjelasan_model, text="Model MobileNetV2 untuk klasifikasi gambar fresh atau busuk.")

    global model 
    model = model_mobileNet

# melakukan prediksi  
def run_prediction():
    global image 
    if image is None:
        print("Tidak ada gambar untuk diprediksi.")
        return
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    global model_choices 
    if model_choices >= 1 and model_choices < 7 :
        user_img = image.resize((150, 150))  # resize seperti data latih
        img_array = img_to_array(user_img) / 255.0
        img_array = img_array.reshape(1, -1) 
        # Prediksi gambar
        pred = model.predict(img_array)[0]
        prob = model.predict_proba(img_array)[0][1]
        label = 'Fresh' if pred == 1 else 'Rotten'
        # Hasil Prediksi
        canvas.itemconfig(akurasi, text=f"Akurasi: {prob:.2f}")
        canvas.itemconfig(kesimpulan, text=f"{label} Food")
        return
    elif model_choices == 7:
        img_resized = image.resize((150, 150))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        pred = model.predict(img_array)[0][0]
        if pred > 0.5:
            result_text = "Fresh Food"
        else:
            result_text = "Spoiled Food"

        # Update penjelasan prediksi
        canvas.itemconfig(akurasi, text=f"Akurasi: {pred:.2f}")
        canvas.itemconfig(kesimpulan, text=f"Kesimpulan: {result_text}")
        return
    elif model_choices == 8 or model_choices == 9:
        img = image.resize((150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150 * 3)  # (batch_size, timesteps, features)

        # Prediksi gambar user
        prob = model.predict(img_array)[0][0]
        pred = int(prob > 0.5)
        label = 'Fresh' if pred == 1 else 'Spoiled'

        # Tampilkan hasil
        canvas.itemconfig(akurasi, text=f"Akurasi: {prob:.2f}")
        canvas.itemconfig(kesimpulan, text=f"{label} Food")
        return
    elif model_choices == 10:
        # Prediksi gambar baru
        img = image.resize((224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prob = model.predict(img_array)[0][0]
        pred = int(prob > 0.5)
        label = 'Fresh' if pred == 1 else 'Spoiled'
        
        # Tampilkan hasil
        canvas.itemconfig(akurasi, text=f"Akurasi: {prob:.2f}")
        canvas.itemconfig(kesimpulan, text=f"{label} Food")
        return
    

window = Tk()
window.geometry("960x544")
window.configure(bg = "#FFFFFF")

custom_judul = font.Font(family="Roboto", size=24*-1, weight="bold", slant="italic")
custom_isi = font.Font(family="Roboto", size=22*-1, weight="bold", slant="italic")

canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 544,
    width = 960,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    480.0,
    271.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    480.0,
    272.0,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    645.0,
    192.0,
    image=image_image_3
)

nama_model = canvas.create_text(
    674.0,
    65.0,
    anchor="nw",
    text="Model",
    fill="#E2DED3",
    font=custom_judul
)

penjelasan_model = canvas.create_text(
    368.0,
    210.0,
    anchor="nw",
    text="Penjelasan",
    fill="#E2DED3",
    font=custom_isi
)

nama_image = canvas.create_text(
    368.0,
    124.0,
    anchor="nw",
    text="Nama foto",
    fill="#E2DED3",
    font=custom_isi
)

kesimpulan = canvas.create_text(
    368.0,
    176.0,
    anchor="nw",
    text="Kesimpulan",
    fill="#E2DED3",
    font=custom_isi
)

akurasi = canvas.create_text(
    368.0,
    98.0,
    anchor="nw",
    text="Akurasi",
    fill="#E2DED3",
    font=custom_isi
)

canvas.create_text(
    368.0,
    65.0,
    anchor="nw",
    text="Hasil Prediksi ",
    fill="#E2DED3",
    font=custom_judul
)

ukuran_image = canvas.create_text(
    368.0,
    150.0,
    anchor="nw",
    text="Ukuran",
    fill="#E2DED3",
    font=custom_isi
)

precision = canvas.create_text(
    674.0,
    150.0,
    anchor="nw",
    text="Precision",
    fill="#E2DED3",
    font=custom_isi
)

recall = canvas.create_text(
    674.0,
    124.0,
    anchor="nw",
    text="Recall",
    fill="#E2DED3",
    font=custom_isi
)

canvas.create_text(
    674.0,
    176.0,
    anchor="nw",
    text="Dataset",
    fill="#E2DED3",
    font=custom_isi
)

akurasi_model = canvas.create_text(
    674.0,
    98.0,
    anchor="nw",
    text="Akurasi Model",
    fill="#E2DED3",
    font=custom_isi
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=model_lstm,
    relief="flat"
)
button_1.place(
    x=492.0,
    y=428.0,
    width=116.0,
    height=40.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=model_mobileNet,
    relief="flat"
)
button_2.place(
    x=492.0,
    y=374.0,
    width=116.0,
    height=40.0
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    480.0,
    22.0,
    image=image_image_4
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=model_svm,
    relief="flat"
)
button_3.place(
    x=28.0,
    y=428.0,
    width=116.0,
    height=40.0
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=model_cnn,
    relief="flat"
)
button_4.place(
    x=376.0,
    y=374.0,
    width=116.0,
    height=40.0
)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=model_logisticregression,
    relief="flat"
)
button_5.place(
    x=260.0,
    y=428.0,
    width=116.0,
    height=40.0
)

button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_6 = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=model_naivebayes,
    relief="flat"
)
button_6.place(
    x=144.0,
    y=428.0,
    width=116.0,
    height=40.0
)

button_image_7 = PhotoImage(
    file=relative_to_assets("button_7.png"))
button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=brightness_down,
    relief="flat"
)
button_7.place(
    x=750.0,
    y=428.0,
    width=116.0,
    height=40.0
)

button_image_8 = PhotoImage(
    file=relative_to_assets("button_8.png"))
button_8 = Button(
    image=button_image_8,
    borderwidth=0,
    highlightthickness=0,
    command=model_decisionTree,
    relief="flat"
)
button_8.place(
    x=260.0,
    y=374.0,
    width=116.0,
    height=40.0
)

button_image_9 = PhotoImage(
    file=relative_to_assets("button_9.png"))
button_9 = Button(
    image=button_image_9,
    borderwidth=0,
    highlightthickness=0,
    command=model_knn,
    relief="flat"
)
button_9.place(
    x=144.0,
    y=374.0,
    width=116.0,
    height=40.0
)

button_image_10 = PhotoImage(
    file=relative_to_assets("button_10.png"))
button_10 = Button(
    image=button_image_10,
    borderwidth=0,
    highlightthickness=0,
    command=model_rnn,
    relief="flat"
)
button_10.place(
    x=376.0,
    y=428.0,
    width=116.0,
    height=40.0
)

button_image_11 = PhotoImage(
    file=relative_to_assets("button_11.png"))
button_11 = Button(
    image=button_image_11,
    borderwidth=0,
    highlightthickness=0,
    command=run_prediction,
    relief="flat"
)
button_11.place(
    x=570.0,
    y=486.0,
    width=360.0,
    height=40.0
)

button_image_12 = PhotoImage(
    file=relative_to_assets("button_12.png"))
button_12 = Button(
    image=button_image_12,
    borderwidth=0,
    highlightthickness=0,
    command=openFile,
    relief="flat"
)
button_12.place(
    x=28.0,
    y=486.0,
    width=360.0,
    height=40.0
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    177.0,
    192.0,
    image=image_image_5
)

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(
    177.335693359375,
    191.9558868408203,
    image=image_image_6
)

image_image_7 = PhotoImage(
    file=relative_to_assets("image_7.png"))
image_7 = canvas.create_image(
    17.0,
    18.0,
    image=image_image_7
)

image_image_8 = PhotoImage(
    file=relative_to_assets("image_8.png"))
image_8 = canvas.create_image(
    942.0,
    18.0,
    image=image_image_8
)

image_image_9 = PhotoImage(
    file=relative_to_assets("image_9.png"))
image_9 = canvas.create_image(
    17.0,
    526.0,
    image=image_image_9
)

image_image_10 = PhotoImage(
    file=relative_to_assets("image_10.png"))
image_10 = canvas.create_image(
    942.0,
    525.9999938804057,
    image=image_image_10
)

button_image_13 = PhotoImage(
    file=relative_to_assets("button_13.png"))
button_13 = Button(
    image=button_image_13,
    borderwidth=0,
    highlightthickness=0,
    command=resetPrediction,
    relief="flat"
)
button_13.place(
    x=442.0,
    y=486.0,
    width=85.0,
    height=40.0
)

button_image_14 = PhotoImage(
    file=relative_to_assets("button_14.png"))
button_14 = Button(
    image=button_image_14,
    borderwidth=0,
    highlightthickness=0,
    command=model_rf,
    relief="flat"
)
button_14.place(
    x=28.0,
    y=374.0,
    width=116.0,
    height=40.0
)

button_image_15 = PhotoImage(
    file=relative_to_assets("button_15.png"))
button_15 = Button(
    image=button_image_15,
    borderwidth=0,
    highlightthickness=0,
    command=brightness_up,
    relief="flat"
)
button_15.place(
    x=750.0,
    y=374.0,
    width=116.0,
    height=40.0
)

image_image_11 = PhotoImage(
    file=relative_to_assets("image_11.png"))
image_11 = canvas.create_image(
    807.0,
    348.0,
    image=image_image_11
)

image_image_12 = PhotoImage(
    file=relative_to_assets("image_12.png"))
image_12 = canvas.create_image(
    316.0,
    348.0,
    image=image_image_12
)

canvas.create_rectangle(
    -1.9999645840023277,
    32.99999994831518,
    2.0001063563636308,
    514.0,
    fill="#FFFFFF",
    outline="")

canvas.create_rectangle(
    35.0,
    542.0,
    925.0,
    544.0,
    fill="#FFFFFF",
    outline="")

canvas.create_rectangle(
    35.0,
    1.0,
    925.0,
    3.0,
    fill="#FFFFFF",
    outline="")
window.resizable(False, False)
window.mainloop()
