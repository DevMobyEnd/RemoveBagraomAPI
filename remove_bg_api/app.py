import os
import io
import logging
import hashlib
import numpy as np
from functools import lru_cache
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from apscheduler.schedulers.background import BackgroundScheduler

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear la aplicación Flask y configurar CORS
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuración adicional de Flask
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Usar variables de entorno para las rutas
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads/')
PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', 'static/')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Crear una sesión de rembg con configuraciones optimizadas
session = new_session("u2net")

# Crear o cargar el modelo U-Net mejorado
def create_unet_model():
    inputs = layers.Input((256, 256, 3))
    
    # Encoder (más profundo)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder (más profundo)
    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = layers.Conv2D(256, 2, activation='relu', padding='same')(up5)
    merge5 = layers.Concatenate()([conv3, up5])
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = layers.Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = layers.Concatenate()([conv2, up6])
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = layers.Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = layers.Concatenate()([conv1, up7])
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

try:
    model = models.load_model('background_removal_model.h5')
    logger.info("Modelo cargado exitosamente.")
except:
    logger.info("Creando nuevo modelo...")
    model = create_unet_model()

# Función para entrenar el modelo con nuevas imágenes
def train_model_with_new_data():
    processed_images = os.listdir(PROCESSED_FOLDER)
    total_images = len(processed_images)
    
    update_progress(total_images, 200)
    
    if total_images > 200:
        X, y = [], []
        for i, img_name in enumerate(processed_images[:200]):
            original = Image.open(os.path.join(UPLOAD_FOLDER, img_name.replace('processed_', '')))
            processed = Image.open(os.path.join(PROCESSED_FOLDER, img_name))
            X.append(image.img_to_array(original.resize((256, 256))))
            y.append(image.img_to_array(processed.resize((256, 256))))
            update_progress(i+1, 200)
        
        X = np.array(X) / 255.0
        y = np.array(y) / 255.0
        
        history = model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2,
                            callbacks=[TrainingCallback(update_progress)])
        
        model.save('background_removal_model.h5')
        logger.info("Modelo entrenado y guardado exitosamente.")

# Implementar caché para imágenes procesadas
@lru_cache(maxsize=100)
def process_image(image_hash):
    file_path = os.path.join(UPLOAD_FOLDER, f"{image_hash}.png")
    with Image.open(file_path) as img:
        # Usar tanto rembg como el modelo U-Net
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        input_image = img_byte_arr.getvalue()
        
        # Remover el fondo con rembg (parámetros ajustados)
        output_image = remove(
            input_image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=220,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=10,
            post_process_mask=True,
            only_mask=False
        )
        
        # Procesar con el modelo U-Net
        img_array = image.img_to_array(img.resize((256, 256))) / 255.0
        prediction = model.predict(np.expand_dims(img_array, 0))[0]
        unet_output = Image.fromarray((prediction.squeeze() * 255).astype(np.uint8))
        unet_output = unet_output.resize(img.size)
        
        # Combinar los resultados (fusión mejorada)
        rembg_output = Image.open(io.BytesIO(output_image))
        final_output = Image.new('RGBA', img.size, (0, 0, 0, 0))
        for x in range(img.width):
            for y in range(img.height):
                r, g, b, a = rembg_output.getpixel((x, y))
                unet_alpha = unet_output.getpixel((x, y))
                # Usar una combinación ponderada
                final_alpha = int(0.7 * a + 0.3 * unet_alpha)
                if final_alpha > 0:
                    final_output.putpixel((x, y), (r, g, b, final_alpha))
        
        # Mejorar la calidad de la imagen final (post-procesamiento mejorado)
        final_output = final_output.filter(ImageFilter.SMOOTH)
        enhancer = ImageEnhance.Contrast(final_output)
        final_output = enhancer.enhance(1.2)
        sharpener = ImageEnhance.Sharpness(final_output)
        final_output = sharpener.enhance(1.1)
        
        final_byte_arr = io.BytesIO()
        final_output.save(final_byte_arr, format='PNG')
        return final_byte_arr.getvalue()

# Ruta del dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Función para actualizar el progreso
def update_progress(processed_count, total_count, epoch=None, loss=None, accuracy=None):
    socketio.emit('update_progress', {
        'processed_count': processed_count,
        'total_count': total_count,
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy
    })

class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, update_func):
        self.update_func = update_func

    def on_epoch_end(self, epoch, logs=None):
        self.update_func(200, 200, epoch+1, logs.get('loss'), logs.get('accuracy'))

@app.route('/')
def home():
    return jsonify({'message': 'Bienvenido a la API de eliminación de fondo'}), 200

@app.route('/remove_background', methods=['POST'])
def remove_background():
    if 'file' not in request.files:
        return jsonify({'error': 'No se ha proporcionado ningún archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se ha seleccionado ningún archivo'}), 400
    
    if file:
        try:
            # Generar un hash único para la imagen
            image_hash = hashlib.md5(file.read()).hexdigest()
            file.seek(0)  # Reiniciar el puntero del archivo

            # Guardar el archivo original
            original_file_path = os.path.join(UPLOAD_FOLDER, f"{image_hash}.png")
            file.save(original_file_path)
            
            # Procesar la imagen
            output = process_image(image_hash)
            
            # Guardar la imagen procesada
            processed_file_path = os.path.join(PROCESSED_FOLDER, f"processed_{image_hash}.png")
            with open(processed_file_path, 'wb') as f:
                f.write(output)
            
            return send_file(
                io.BytesIO(output),
                mimetype='image/png',
                as_attachment=True, 
                download_name=f"processed_{image_hash}.png"
            )
        except Exception as e:
            logger.error(f"Error al procesar la imagen: {str(e)}")
            logger.exception("Excepción completa:")
            return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Ruta no encontrada'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Error interno del servidor'}
    ), 500

if __name__ == '__main__':
    # Programar el entrenamiento periódico
    scheduler = BackgroundScheduler()
    scheduler.add_job(train_model_with_new_data, 'interval', hours=12)  # Entrenamiento cada 12 horas
    scheduler.start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    
   
  