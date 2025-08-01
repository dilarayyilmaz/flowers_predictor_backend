from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFile
import io
import time
import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)


MODEL_PATH = 'flower_model.keras'
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    _ = model.predict(np.zeros((1, 150, 150, 3)))
    app.logger.info(f"Model '{MODEL_PATH}' başarıyla yüklendi.")
except Exception as e:
    app.logger.error(f"Model yüklenirken kritik bir hata oluştu: {e}")

CLASS_NAMES = ['lale', 'nergis', 'orkide', 'ortanca', 'papatya']


def prepare_image(image_bytes):
    """
    Gelen byte verisini modele uygun hale getiren fonksiyon.
    EĞİTİMDEKİ ÖN İŞLEME ADIMLARIYLA AYNI OLACAK ŞEKİLDE DÜZELTİLDİ.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        target_size = (150, 150)
        image = image.resize(target_size, Image.Resampling.BILINEAR)

        img_array = np.array(image, dtype=np.float32)

        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        app.logger.error(f"Resim hazırlama sırasında hata: {e}")
        return None


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify(
            {'error': 'Model sunucuda yüklenemedi, lütfen logları kontrol edin.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'İstek içerisinde "file" anahtarıyla bir dosya gönderilmedi.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi.'}), 400

    try:
        img_bytes = file.read()
        image_tensor = prepare_image(img_bytes)

        if image_tensor is None:
            return jsonify({'error': 'Gönderilen dosya geçerli bir resim değil veya işlenemedi.'}), 400

        prediction = model.predict(image_tensor)

        predicted_index = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        predicted_class = CLASS_NAMES[predicted_index]

        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence:.4f}",
        })

    except Exception as e:
        app.logger.error(f"Tahmin endpoint'inde beklenmedik bir hata oluştu: {e}")
        return jsonify({'error': 'Tahmin sırasında sunucuda bir hata oluştu.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)