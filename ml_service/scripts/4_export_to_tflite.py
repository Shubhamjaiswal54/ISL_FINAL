# 4_export_to_tflite.py
# Quantization and TFLite model export script
import numpy as np
import tensorflow as tf
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

def convert_and_quantize(model_path='models/isl_model.keras',
                         tflite_path='models/isl_model.tflite'):
    """Load the Keras model and export an INT8-quantized TFLite model."""
    model_path = BASE_DIR / model_path
    tflite_path = BASE_DIR / tflite_path
    
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Representative dataset for INT8 quantization
    # Use a small subset of training data for calibration
    calib_data = np.load(BASE_DIR / 'data/processed/X_train.npy')
    calib_set = calib_data[:100]  # 100 samples for calibration
    
    def representative_dataset():
        for data in calib_set:
            yield [tf.cast(data, tf.float32)]
    
    # Convert to TFLite with dynamic range quantization
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_quant_model = converter.convert()
        export_type = 'INT8'
    except Exception as error:
        print(f'INT8 export failed, falling back to float: {error}')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_quant_model = converter.convert()
        export_type = 'Float/Select TF Ops'
    
    # Save the quantized TFLite model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_quant_model)
    
    print(f'{export_type} TFLite model saved to {tflite_path}')
    
    # Also save the non-quantized version for reference
    converter_no_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_no_quant.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter_no_quant._experimental_lower_tensor_list_ops = False
    tflite_full_model = converter_no_quant.convert()
    with open(BASE_DIR / 'models/isl_model_full.tflite', 'wb') as f:
        f.write(tflite_full_model)
    print('Full (non-quantized) TFLite model saved to models/isl_model_full.tflite')
    
    return tflite_path

if __name__ == '__main__':
    convert_and_quantize()