import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# NLTK ve Türkçe stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))

# Türkçe stopwords ve tokenizasyon
def preprocess_text(text):
    text = text.lower()  # Küçük harfe dönüştürme
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldırma
    tokens = word_tokenize(text)  # Tokenizasyon
    tokens = [word for word in tokens if word not in stop_words]  # Stopwords kaldırma
    return tokens

# Veriyi yükle ve ön işleme
def load_data(file_path):
    df = pd.read_csv(file_path)
    label_mapping = {'neutral': 0, 'negative': 1, 'positive': 2}
    df['sentiment'] = df['sentiment'].map(label_mapping)
    df['comment'] = df['comment'].apply(preprocess_text)
    texts = df['comment'].apply(lambda x: ' '.join(x)).values  # Listeleri birleştirip metin haline getiriyoruz
    labels = df['sentiment'].astype(int).values
    return texts, labels

# Veriyi yükle
texts, labels = load_data('new_data/new_video_comments2.csv')

# Eğitim ve test verilerini ayırma
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Eğitim verilerini ve doğrulama verilerini string formatında almak
X_train_str = [' '.join(text) for text in X_train]  # Listelerden birleştirilmiş metinler oluşturuyoruz
X_val_str = [' '.join(text) for text in X_val]

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# BERT Tokenizer ve Model Yükleme
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
bert_model = TFBertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-cased', num_labels=3)

# BERT'in metinleri tokenlemesi
def encode_text(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=120, return_tensors='tf')

train_encodings = encode_text(X_train_str)
val_encodings = encode_text(X_val_str)

# Dataset formatına dönüştürme
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).shuffle(len(y_train)).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    y_val
)).batch(32)

# Modeli derleme
bert_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Modeli eğitme
history = bert_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=25,  # Daha fazla epoch
    class_weight=class_weights_dict  # Sınıf ağırlıkları
)

# Modeli kaydetme
bert_model.save_pretrained('saved_bert_model4')
tokenizer.save_pretrained('saved_bert_model4')

# Modeli değerlendirme
y_pred = bert_model.predict(val_dataset)
y_pred_classes = np.argmax(y_pred.logits, axis=-1)

# Sonuçları yazdırma
print(classification_report(y_val, y_pred_classes, zero_division=1))