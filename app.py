from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFile
import io
import time
import logging
import requests
import base64
import os


ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)


MODEL_PATH = 'flower_model.keras'
CLASS_NAMES = ['lale', 'nergis', 'orkide', 'ortanca', 'papatya']
model = None


GEMINI_API_KEY = "YOUR_API_KEY_HERE"
if not GEMINI_API_KEY:
    app.logger.warning(
        "Gemini API anahtarı ayarlanmadı. Model güvenilirliği eşiğinin altında kalan tahminler için Gemini kullanılamayacak.")


try:
    model = tf.keras.models.load_model(MODEL_PATH)

    _ = model.predict(np.zeros((1, 150, 150, 3)))
    app.logger.info(f"Model '{MODEL_PATH}' başarıyla yüklendi.")
except Exception as e:
    app.logger.error(f"Model yüklenirken kritik bir hata oluştu: {e}")


def prepare_image(image_bytes):
    """
    Gelen byte verisini TensorFlow modeline uygun hale getiren fonksiyon.
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


def predict_with_gemini(image_bytes):
    """
    Gemini API'sini kullanarak resimdeki çiçeği tahmin eden fonksiyon.
    """
    if not GEMINI_API_KEY:
        app.logger.warning("Gemini API anahtarı boş olduğu için Gemini tahmini atlanıyor.")
        return None

    try:

        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        headers = {
            'Content-Type': 'application/json',
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Bu resimdeki çiçeğin türü nedir? Sadece çiçeğin adını ver, başka hiçbir şey söyleme."},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",  # image/png veya image/jpeg olabilir
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # HTTP hatalarını kontrol et

        result = response.json()


        if 'candidates' in result and len(result['candidates']) > 0:
            candidate = result['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                if len(parts) > 0 and 'text' in parts[0]:
                    predicted_text = parts[0]['text'].strip()
                    app.logger.info(f"Gemini tahmini: {predicted_text}")
                    return predicted_text

        return None

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Gemini API isteği sırasında hata: {e}")
        return None
    except Exception as e:
        app.logger.error(f"Gemini API yanıtını işlerken hata: {e}")
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

        confidence_threshold = 0.7

        if confidence > confidence_threshold:
            return jsonify({
                'class': predicted_class,
                'confidence': f"{confidence:.4f}",
                'source': 'TensorFlow'
            })
        else:
            app.logger.info(f"TensorFlow güvenilirliği düşük ({confidence:.4f}), Gemini'ye geçiliyor.")
            gemini_prediction = predict_with_gemini(img_bytes)

            if gemini_prediction:
                return jsonify({
                    'class': gemini_prediction,
                    'confidence': 'Yüksek (Gemini)',
                    'source': 'Gemini'
                })
            else:
                return jsonify({
                    'class': 'Bilinmiyor',
                    'confidence': f"{confidence:.4f}",
                    'source': 'TensorFlow'
                })

    except Exception as e:
        app.logger.error(f"Tahmin endpoint'inde beklenmedik bir hata oluştu: {e}")
        return jsonify({'error': 'Tahmin sırasında sunucuda bir hata oluştu.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)