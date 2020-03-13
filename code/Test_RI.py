import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.visible_device_list = "0"
config.allow_soft_placement=True
config.log_device_placement=True

set_session(tf.Session(config=config))

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()



from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from model_IR_2 import ResearchModels
from data_RI import DataSet
import functools
import keras.metrics




def validate(data_type, model, seq_length=16, saved_model=None,
             class_limit=None, image_shape=None):
    batch_size = 16

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Evaluate!
    results = rm.model.evaluate_generator(
        generator=val_generator,
        val_samples=12866,  # You might need to change this accordingly
        use_multiprocessing=True,
        workers=1)

    print(results)
    print(rm.model.metrics_names)

def main():
    model = 'capsule'
    saved_model = 'data/checkpoints_RI_2/capsule-features.144-0.200-RI-caps_2.hdf5'

    data_type = 'features'
    image_shape = None

    validate(data_type, model, saved_model=saved_model,
             image_shape=image_shape, class_limit=16)

if __name__ == '__main__':
    main()
