import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.visible_device_list = "1"
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
from sklearn.metrics import classification_report
import numpy as np





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

    #val_generator = data.frame_generator(1,'test', data_type)
    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)
    # Evaluate!
    scores=np.zeros([14])
    total=np.zeros([14])

    val_trues=[]
    val_preds=[]
    for X,y in data.gen_test('test', data_type): 
        results = rm.model.predict(X)
        predicted=np.argmax(results, axis=-1)
        idx=np.where(np.array(y)==1)
        true_label=idx[1]
        print(true_label)

        total[true_label]=total[true_label]+1
        
        print(len(predicted))
        print(len(true_label))
        if predicted[0]==true_label[0]:
            scores[true_label]=scores[true_label]+1
        
        
        

    
    #val_preds = np.argmax(results, axis=-1)
    

    print(classification_report(true_label, predicted))
    print(scores)
    print('\n****************')
    print(total)



def main():
    model = 'capsule'
    saved_model = 'data/checkpoints_RI_2/capsule-features.144-0.200-RI-caps_2.hdf5'
    data_type = 'features'
    image_shape = None

    validate(data_type, model, saved_model=saved_model,
             image_shape=image_shape, class_limit=16)

if __name__ == '__main__':
    main()
