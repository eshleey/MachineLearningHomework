import pandas as pd
import librosa
import numpy as np
import os
import time  # Süre takibi için
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
from tqdm import tqdm

# librosa'dan gelebilecek olası FutureWarning'ları gizle
warnings.filterwarnings('ignore')

# 1. ORTAM VE YOL AYARLARI
BASE_PATH = r'C:\Users\Leonidas\Downloads\archive'
AUDIO_PATH = BASE_PATH
CSV_PATH = os.path.join(BASE_PATH, 'UrbanSound8K.csv')

# Sabitler
TARGET_SR = 45  # SORUDA BELİRTİLEN ANORMAL ÖRNEKLEME HIZI
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Başlangıç zamanını kaydetme
start_time = time.time()
print("Başlangıç Zamanı:", time.ctime(start_time))
print("-" * 50)

# 2. VERİ YÜKLEME, MAKSİMUM UZUNLUK BULMA VE SIFIR DOLDURMA (ZERO-PADDING)
print("2. Adım: Tüm ses dosyaları okunuyor, maksimum uzunluk hesaplanıyor ve dolduruluyor...")

metadata = pd.read_csv(CSV_PATH)
raw_data_list = []
labels = []
max_len = 0  # Maksimum örnek sayısını tutacak
toplam_dosya_sayisi = len(metadata)

# tqdm kullanarak ilerleme çubuğunu döngüye dahil etme
# desc='Dosyalar Okunuyor', ilerleme çubuğunun açıklamasını belirler.
for index, row in tqdm(metadata.iterrows(), total=toplam_dosya_sayisi, desc='Dosyalar Okunuyor'):
    # Dosya yolunu oluşturma
    file_name = os.path.join(
        AUDIO_PATH,
        'fold' + str(row["fold"]),
        str(row["slice_file_name"])
    )
    class_id = row["classID"]

    try:
        data, sr = librosa.load(file_name, sr=TARGET_SR)

        raw_data_list.append(data)
        labels.append(class_id)
        if len(data) > max_len:
            max_len = len(data)
    except Exception as e:
        # Hata durumunda atlanacak
        continue  # Hatalı dosyayı atla

print(f"\nToplam başarılı dosya sayısı: {len(raw_data_list)}")
print(f"Maksimum örnek sayısı (max_len): {max_len}")

# --- Zero-Padding Uygulama ---
padded_data = []
print("Ham veriler aynı uzunluğa (Zero-Padding) getiriliyor...")

for data in raw_data_list:
    padding_length = max_len - len(data)
    padded_array = np.pad(data, (0, padding_length), mode='constant', constant_values=0)
    padded_data.append(padded_array)

X = np.array(padded_data)
y = np.array(labels)

print(f"X (Özellik Matrisi - Ham Veri) Şekli: {X.shape}")
print("-" * 50)

# ----------------------------------------------------------------------
# 3. VERİ KÜMESİ BÖLÜMLEME
# ----------------------------------------------------------------------
# Stratify=y, her alt kümede sınıfların oransal dağılımını korur.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Eğitim Kümesi Şekli (X_train): {X_train.shape}")
print(f"Test Kümesi Şekli (X_test): {X_test.shape}")
print("-" * 50)

# ----------------------------------------------------------------------
# 4. RANDOM FOREST EĞİTİMİ VE TAHMİN
# ----------------------------------------------------------------------
print("4. Adım: Random Forest Modeli Eğitiliyor (n_estimators=100)...")
# n_jobs=-1, tüm işlemci çekirdeklerini kullanmayı dener.
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Eğitim Tamamlandı.")

# Test kümesi üzerinde tahmin yapma
y_pred = rf_model.predict(X_test)
print("-" * 50)

# ----------------------------------------------------------------------
# 5. MODEL DEĞERLENDİRME VE YORUMLAMA
# ----------------------------------------------------------------------
print("5. Adım: Model Değerlendirme Sonuçları")

# Sınıflandırma Raporu (Precision, Recall, F1-Score)
print("\nSınıflandırma Raporu:")
target_names = [f'Sınıf {i}' for i in range(10)]
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Karmaşıklık Matrisi (Confusion Matrix)
print("\nKarmaşıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))

# Modelin genel doğruluk (Accuracy) değeri
accuracy = rf_model.score(X_test, y_test)
print(f"\nModel Doğruluğu (Accuracy): {accuracy:.4f}")
print("-" * 50)

print(f"Model, Ham Ses Verisi ({X.shape[1]} boyutlu vektörler) ile eğitilmiştir.")
print(f"Elde edilen doğruluk: {accuracy:.4f}")

# Bitiş zamanını kaydetme
end_time = time.time()
print(f"\nToplam Çalışma Süresi: {end_time - start_time:.2f} saniye")
