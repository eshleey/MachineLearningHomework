import os, time, numpy as np, pandas as pd, librosa
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# 1. Veri yolları ve ayarlar
# ======================================================
BASE_PATH = r"C:\Users\Leonidas\Downloads\archive"
META_CSV = os.path.join(BASE_PATH, "UrbanSound8K.csv")
AUDIO_PATH = os.path.join(BASE_PATH, "")

SAMPLE_RATE = 22050
N_MELS = 128
N_TREES = 20
MAX_DEPTH = 10
TEST_SIZE = 0.2
rng = np.random.default_rng(42)

FEATURE_CSV = "mel_features_rf.csv"

# ======================================================
# 2. Özellik çıkarımı
# ======================================================
if not os.path.exists(FEATURE_CSV):
    meta = pd.read_csv(META_CSV)
    features, labels = [], []

    print(f"Toplam {len(meta)} dosya işlenecek...\n")
    for idx, row in meta.iterrows():
        file_path = os.path.join(AUDIO_PATH, f"fold{row['fold']}", row['slice_file_name'])
        if not os.path.exists(file_path):
            continue

        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mels_mean = np.mean(mels, axis=1)

        features.append(mels_mean)
        labels.append(row["class"])

        percent = (idx + 1) / len(meta) * 100
        print(f"\rÖzellik çıkarımı: %{percent:.1f}", end="")

    print("\nÖzellik çıkarımı tamamlandı. CSV kaydediliyor...")
    df = pd.DataFrame(features, columns=[f"mel_{i}" for i in range(len(features[0]))])
    df["label"] = labels
    df.to_csv(FEATURE_CSV, index=False)
else:
    print("Özellik dosyası bulundu, doğrudan yükleniyor...")
    df = pd.read_csv(FEATURE_CSV)

# ======================================================
# 3. Veri bölme
# ======================================================
X = df.drop("label", axis=1).values
y = df["label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
print(f"\nEğitim kümesi: {X_train.shape}, Test kümesi: {X_test.shape}\n")

# ======================================================
# 4. From-scratch Decision Tree & Random Forest
# ======================================================
def gini_impurity(y):
    counts = Counter(y)
    impurity = 1.0
    for lbl in counts:
        p = counts[lbl] / len(y)
        impurity -= p ** 2
    return impurity

def split_dataset(X, y, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

def best_split(X, y, features_subset, n_thresholds=16, min_leaf=10):
    best_gain, best_feature, best_thresh = 0, None, None
    current_impurity = gini_impurity(y)
    n = len(y)

    for feature in features_subset:
        col = X[:, feature]
        qs = np.linspace(0.05, 0.95, n_thresholds)
        thresholds = np.unique(np.quantile(col, qs))
        for t in thresholds:
            X_left, X_right, y_left, y_right = split_dataset(X, y, feature, t)
            if len(y_left) < min_leaf or len(y_right) < min_leaf:
                continue
            p = len(y_left) / n
            gain = current_impurity - p * gini_impurity(y_left) - (1 - p) * gini_impurity(y_right)
            if gain > best_gain:
                best_gain, best_feature, best_thresh = gain, feature, t
    return best_gain, best_feature, best_thresh

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=20, min_samples_leaf=10, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        features = rng.choice(num_features, self.n_features, replace=False)
        gain, feature, threshold = best_split(X, y, features, n_thresholds=32, min_leaf=self.min_samples_leaf)

        if gain <= 0 or feature is None:
            return Counter(y).most_common(1)[0][0]

        X_left, X_right, y_left, y_right = split_dataset(X, y, feature, threshold)
        left = self.fit(X_left, y_left, depth + 1)
        right = self.fit(X_right, y_right, depth + 1)
        return (feature, threshold, left, right)

    def train(self, X, y):
        self.tree = self.fit(X, y)

    def predict_one(self, x, node=None):
        if node is None:
            node = self.tree
        if not isinstance(node, tuple):
            return node
        feature, threshold, left, right = node
        branch = left if x[feature] <= threshold else right
        return self.predict_one(x, branch)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

class RandomForest:
    def __init__(self, n_trees=20, max_depth=10, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y, sample_ratio=0.8):
        n_samples = len(X)
        for i in range(self.n_trees):
            m = int(sample_ratio * n_samples)
            idxs = rng.choice(n_samples, m, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = DecisionTree(max_depth=self.max_depth, n_features=self.n_features)
            tree.train(X_sample, y_sample)
            self.trees.append(tree)
            percent = (i + 1) / self.n_trees * 100
            print(f"\rAğaç {i+1}/{self.n_trees} eğitiliyor... %{percent:.1f}", end="")
        print("\nRandom Forest eğitimi tamamlandı.\n")

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        y_pred = []
        for i in range(X.shape[0]):
            votes = Counter(tree_preds[:, i])
            y_pred.append(votes.most_common(1)[0][0])
        return np.array(y_pred)

# ======================================================
# 5. Model Eğitimi
# ======================================================
start_global = time.time()
rf = RandomForest(n_trees=N_TREES, max_depth=MAX_DEPTH, n_features=64)
rf.fit(X_train, y_train)
print(f"Eğitim süresi: {(time.time()-start_global):.2f} sn\n")

# ======================================================
# 6. Test ve Değerlendirme
# ======================================================
y_pred = rf.predict(X_test)
print("Sonuçlar:\n")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall   :", recall_score(y_test, y_pred, average='macro'))
print("\nKarmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
# ======================================================
# 7. Görselleştirilmiş Karışıklık Analizi (Yeni Bölüm)
# ======================================================
print("\nSınıfların Karışma Analizi (Karmaşıklık Matrisi Grafiği):\n")

# 1. Karmaşıklık Matrisini Hesapla
cm = confusion_matrix(y_test, y_pred)

# 2. Sınıf İsimlerini Al
class_names = np.unique(y) # Tüm veri setindeki benzersiz sınıf isimleri

# 3. Grafik Çizimi
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,              # Sayı değerlerini hücrelere yaz
    fmt='d',                 # Sayıları ondalık değil, tam sayı (digit) olarak göster
    cmap='Blues',            # Mavi renk skalası kullan
    xticklabels=class_names, # X ekseni etiketleri (Tahmin edilen sınıflar)
    yticklabels=class_names  # Y ekseni etiketleri (Gerçek sınıflar)
)

plt.title('Random Forest Karmaşıklık Matrisi')
plt.ylabel('Gerçek Sınıf (True Label)')
plt.xlabel('Tahmin Edilen Sınıf (Predicted Label)')
plt.show()