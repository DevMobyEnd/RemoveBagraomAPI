from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image, ImageEnhance
import os
import io
import logging
import hashlib
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # Esto habilita CORS para todas las rutas

# Ruta donde se guardarán las imágenes procesadas
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'static/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Crear una sesión de rembg con configuraciones optimizadas
session = new_session("u2net")

# Implementar caché para imágenes procesadas
@lru_cache(maxsize=100)
def process_image(image_hash):
    file_path = os.path.join(UPLOAD_FOLDER, f"{image_hash}.png")
    with Image.open(file_path) as img:
        # Convertir la imagen a bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        input_image = img_byte_arr.getvalue()
        
        # Remover el fondo con configuraciones optimizadas
        output_image = remove(
            input_image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            post_process_mask=True,
            only_mask=False
        )
        
        # Post-procesamiento adicional
        output = Image.open(io.BytesIO(output_image))
        
        # Convertir a modo RGBA si no lo está
        if output.mode != 'RGBA':
            output = output.convert('RGBA')
        
        # Umbral para considerar un píxel como parte del fondo
        threshold = 200
        
        # Crear una nueva imagen con fondo transparente
        new_output = Image.new('RGBA', output.size, (0, 0, 0, 0))
        
        for x in range(output.width):
            for y in range(output.height):
                r, g, b, a = output.getpixel((x, y))
                if a > 0 and (r > threshold and g > threshold and b > threshold):
                    # Si el píxel es muy claro y no es completamente transparente, lo hacemos transparente
                    new_output.putpixel((x, y), (0, 0, 0, 0))
                else:
                    new_output.putpixel((x, y), (r, g, b, a))
        
        # Convertir la imagen procesada de nuevo a bytes
        final_byte_arr = io.BytesIO()
        new_output.save(final_byte_arr, format='PNG')
        return final_byte_arr.getvalue()

@app.route('/supported-formats', methods=['GET'])
def supported_formats():
    formats = ['JPEG', 'PNG', 'WebP', 'BMP', 'TIFF']
    return jsonify({"supported_formats": formats})

# Ruta para subir la imagen y remover el fondo
@app.route('/remove-background', methods=['POST'])
def remove_background():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Generar hash de la imagen para caché
        image_hash = hashlib.md5(file.read()).hexdigest()
        file.seek(0)  # Resetear el puntero del archivo
        
        # Guardar la imagen temporalmente en la carpeta de uploads
        file_path = os.path.join(UPLOAD_FOLDER, f"{image_hash}.png")
        file.save(file_path)
        
        logging.info(f"File saved at: {file_path}")

        # Procesar la imagen (usando caché)
        output_image = process_image(image_hash)
        
        # Guardar la imagen procesada
        output_path = os.path.join(PROCESSED_FOLDER, f"processed_{image_hash}.png")
        with open(output_path, "wb") as output_file:
            output_file.write(output_image)
        
        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        logging.error(f"Error processing the image: {str(e)}")
        return jsonify({"error": f"Error processing the image: {str(e)}"}), 500

    finally:
        # Limpiar archivos temporales
        if os.path.exists(file_path):
            os.remove(file_path)

# Ruta de ejemplo para verificar que la API esté funcionando
@app.route('/')
def index():
    return jsonify({"message": "API de remoción de fondo está activa"})

# if __name__ == '__main__':
#     # Importar bibliotecas necesarias
#     from tensorflow.keras.models import load_model, Sequential
#     from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
#     from tensorflow.keras.preprocessing.image import img_to_array
#     import numpy as np

#     def create_unet_model():
#         model = Sequential()
#         # Encoder
#         model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
#         model.add(MaxPooling2D((2, 2)))
#         model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#         model.add(MaxPooling2D((2, 2)))
#         # Decoder
#         model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#         model.add(UpSampling2D((2, 2)))
#         model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#         model.add(UpSampling2D((2, 2)))
#         model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
#         return model

#     # Cargar o crear el modelo
#     try:
#         model = load_model('background_removal_model.h5')
#     except:
#         # Si no existe, crea un nuevo modelo
#         model = create_unet_model()

#     # Función para entrenar el modelo con nuevas imágenes
#     def train_model_with_new_data():
#         processed_images = os.listdir(PROCESSED_FOLDER)
#         if len(processed_images) > 100:  # Entrenar cuando haya suficientes imágenes
#             X, y = [], []
#             for img_name in processed_images[:100]:  # Usa las últimas 100 imágenes
#                 original = Image.open(os.path.join(UPLOAD_FOLDER, img_name.replace('processed_', '')))
#                 processed = Image.open(os.path.join(PROCESSED_FOLDER, img_name))
#                 X.append(img_to_array(original))
#                 y.append(img_to_array(processed))
#             X = np.array(X) / 255.0
#             y = np.array(y) / 255.0
#             model.fit(X, y, epochs=5, batch_size=8)
#             model.save('background_removal_model.h5')

    

#     # Modificar la función process_image para usar el modelo
#     def process_image(image_hash):
#         file_path = os.path.join(UPLOAD_FOLDER, f"{image_hash}.png")
#         with Image.open(file_path) as img:
#             img_array = img_to_array(img.resize((256, 256))) / 255.0
#             prediction = model.predict(np.expand_dims(img_array, 0))[0]
#             output = Image.fromarray((prediction * 255).astype(np.uint8))
#             output = output.resize(img.size)
            
#             final_byte_arr = io.BytesIO()
#             output.save(final_byte_arr, format='PNG')
#             return final_byte_arr.getvalue()

#     # Programar el entrenamiento periódico
#     from apscheduler.schedulers.background import BackgroundScheduler
#     scheduler = BackgroundScheduler()
#     scheduler.add_job(train_model_with_new_data, 'interval', hours=24)
#     scheduler.start()

#               # Verificar la instalación
# try:
#     import tensorflow
#     import apscheduler
#     print("Todas las dependencias se han instalado correctamente.")
# except ImportError as e:
#     print(f"Error al importar: {e}")
#     print("Por favor, asegúrate de instalar todas las dependencias manualmente.")
#     print("Usa el siguiente comando:")
#     print("pip install flask flask-cors rembg Pillow tensorflow numpy apscheduler")
#     exit(1)

app.run(debug=True)