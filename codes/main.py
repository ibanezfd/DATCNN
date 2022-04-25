'''
####################################
MAIN CODE FOR DAT-CNN
####################################
'''

import glob
import os
GPU = "0" 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU
import numpy as np
import sys
import datetime
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from model import DAT_CNN
from utils import s2s3_patches_s3_pixel_5, trainScaler, applyScaler, s2s3_patches_s3_pixel_5, compute_metrics
from tensorflow import keras
from tensorflow.python.keras.backend import set_session


## TF AND KERAS CONFIGURATIONS ##

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction=0.5
config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
tf.compat.v1.keras.backend.set_session(sess)
set_session(sess)  # set this TensorFlow session as the default session for Keras
sess.close()
set_session(sess)
begin_time = datetime.datetime.now()

### PATHS AND DATA ###

## PARAMETERS ##
RF = 3 # Reduction Factor
PATCH_SIZE = 15//RF # 32, 16, 8
STEP = 15//RF  # 10, 8
S3_PROD_NAME = 'ndvi' 
RAND_STATE = 5 # random.randint(1,100) # random state used for the tra/tst partioning 
VAL_SIZE = 0.2
TEST_SIZE = 0.4
depth = 2 # Number of temporal samples to take, before and after. Temporal stack of size 2*depth +1
POS_OUT = []
LEARNING_RATE = 0.001 
LR = LEARNING_RATE
OPTIMIZER = keras.optimizers.Adam(lr=LEARNING_RATE)
LOSS = 'mean_squared_error'
METRICS = ['mean_squared_error', 'mean_absolute_error']
BATCH_SIZE = 128  
EPOCHS = 150 
s3_shape = [1,1,4]  # shape of S3 temporal data


## PATHS ##
folder_name = 'results' # Folder to save data
in_folder = "/input/" #path edit
s3_all_names = glob.glob(in_folder+'/s3*.tif')
s2_all_names = glob.glob(in_folder+'/s2*.tif')

#################################################
##  MAIN CODE 
#################################################

### PREVIOUS ###

## TO EXTRACT AND SPLIT S2 AND S3 NAMES ##
print('Total s3 images:{}'.format(len(s3_all_names)))
print('Total s2 images:{}'.format(len(s2_all_names)))
fday = lambda x:x.split('/')[-1].split('.')[0].split('_')[-1]   # Get the day (WE CONSIDER S2 AND S3 OF SAME DAY BY NOW)

s3_days = list ( set(map(fday, s3_all_names)) & set(map(fday, s2_all_names)) )  # Split according the day
s2_days = s3_days

s3_names = []
for name in s3_all_names:   # Create a list and include S3 names in it
    if fday(name) in s3_days:
        s3_names.append(name)

s2_names = []
for name in s2_all_names:   # Same for S2
    if fday(name) in s2_days:
        s2_names.append(name)

s3_names.sort() # Order them
s2_names.sort()
print('Total number of considered s3/s2 image pairs:{}'.format(len(s3_names)))

s3_names_out = []
s3_names_in = []
s2_names_out = []
s2_names_in = []

for p in range(len(s3_names)):  # Extract the images for visuals
    if p in POS_OUT:
        s3_names_out.append(s3_names[p])
        s2_names_out.append(s2_names[p])
    else:
        s3_names_in.append(s3_names[p])
        s2_names_in.append(s2_names[p])

s3_names = s3_names_in
s2_names = s2_names_in

## GENERATE OR ASSING OUTPUT FOLDER ##
script_name = sys.argv[0].split('.')[0]
out_folder = '/output/' + folder_name # we use the script name for the output folder

if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

timer_file = out_folder + "/{}_{}_timer.txt".format(script_name)
with open(timer_file, 'w') as f:
    f.write(script_name + '\n')
    f.write('Start time: {}\n'.format(begin_time))

### DATASET GENERATION ###
FILE_TRATST = out_folder+"/{}_travaltst_patch{}_step{}.npz".format(RAND_STATE, PATCH_SIZE, STEP)

if not os.path.exists(FILE_TRATST): 
    list_s2_patches = []
    list_s3_patches = []
    list_s3_prod_vals = []
    print('Extracting patches and values...')
    s2_names_for3D = s2_names[depth:-depth] 
    s3_names_for3D = s3_names[depth:-depth]

    for i in range(len(s2_names_for3D)):   
        i_temp = i + depth
        s2_temporal = s2_names[i_temp-depth:i_temp+depth+1]
        s3_temporal = s3_names[i_temp-depth:i_temp+depth+1]
        s3_temporal.pop(2) 
        [x1i, x2i, yi] = s2s3_patches_s3_pixel_5(s3_names[i_temp], s2_temporal, s3_temporal, S3_PROD_NAME)
        list_s2_patches.append(x1i)
        list_s3_patches.append(x2i)
        list_s3_prod_vals.append(yi)

    print('Stacking all patches and values...')

    x1 = np.concatenate(list_s2_patches, axis=0) # (num_patches, rows, cols, 13, temp)
    x2 = np.concatenate(list_s3_patches, axis=0) # (num_patches, rows, cols, temp)
    y = np.concatenate(list_s3_prod_vals, axis=0) # (num_patches, 1 (product value))

    del list_s2_patches, list_s3_prod_vals, list_s3_patches
    print('Total extracted samples={}'.format(x1.shape[0]))

    print('Splitting training/validation/test data...')
    
    [x1tra, x1tst, x2tra, x2tst, ytra, ytst] = train_test_split(x1, x2, y, test_size=TEST_SIZE, random_state=RAND_STATE)
    [x1tra, x1val, x2tra, x2val, ytra, yval] = train_test_split(x1tra, x2tra, ytra, test_size=VAL_SIZE, random_state=RAND_STATE)

    print('Saving data... ')
    np.savez(FILE_TRATST, x1tra=x1tra, x1val=x1val, x1tst=x1tst, x2tra=x2tra, x2val=x2val, x2tst=x2tst, ytra=ytra, yval=yval, ytst=ytst)
    with open(timer_file, 'w') as f:
        f.write('Total extracted samples 1={}\n'.format(x1.shape[0]))
        f.write('tra:{}, val:{}, tst:{}\n'.format(x1tra.shape[0], x1val.shape[0], x1tst.shape[0]))

        f.write('Total extracted samples 2={}\n'.format(x2.shape[0]))
        f.write('tra:{}, val:{}, tst:{}\n'.format(x2tra.shape[0], x2val.shape[0], x2tst.shape[0]))

else:

    print('Loading tra/val/tst data from file...')
    with np.load(FILE_TRATST) as data:
        x1tra = data['x1tra']
        x1val = data['x1val']
        x1tst = data['x1tst']
        x2tra = data['x2tra']
        x2val = data['x2val']
        x2tst = data['x2tst']
        ytra = data['ytra']
        yval = data['yval']
        ytst = data['ytst']

print('before scaling')
print('range of xtra {}; max {}; min {}:'.format(np.ptp(xtra), np.max(xtra), np.min(xtra)))
print('range of ytra {}; max {}; min {}:'.format(np.ptp(ytra), np.max(ytra), np.min(ytra)))

## DATA PROCESSING (SCALER) ##

FILE_SCALER_X1 = out_folder+"/scalerx1.pkl"

if not os.path.exists(FILE_SCALER_X1):  
    print('Computing the scaler from training data 1...')
    SCALER_X1 = trainScaler(x1tra, 'minmax') 
    print('Saving the estimated scaler...')
    with open(FILE_SCALER_X1, 'wb') as file:
        pickle.dump(SCALER_X1, file)

else:
    print('Loading scaler from file...')
    with open(FILE_SCALER_X1, 'rb') as file:
        SCALER_X1 = pickle.load(file)

print('Scaling the input data according to the training set...')
x1tra = applyScaler(x1tra, SCALER_X1)
x1val = applyScaler(x1val, SCALER_X1)
x1tst = applyScaler(x1tst, SCALER_X1)

print('after scaling')   
Xval = x1val
Xtst = x1tst

print('range of x1tra {}; max {}; min {}'.format(np.ptp(x1tra), np.max(x1tra), np.min(x1tra)))
print('range of xtst {}; max {}; min {}'.format(np.ptp(Xtst), np.max(Xtst), np.min(Xtst)))


FILE_SCALER_X2 = out_folder+"/scalerx2.pkl"

if not os.path.exists(FILE_SCALER_X2):
    print('Computing the scaler from training data 2...')
    SCALER_X2 = trainScaler(x2tra, 'minmax') 
    print('Saving the estimated scaler...')
    with open(FILE_SCALER_X2, 'wb') as file:
        pickle.dump(SCALER_X2, file)

else:
    print('Loading scaler from file...')
    with open(FILE_SCALER_X2, 'rb') as file:
        SCALER_X2 = pickle.load(file)

print('Scaling the input data according to the training set...')
x2tra = applyScaler(x2tra, SCALER_X2)
x2val = applyScaler(x2val, SCALER_X2)
x2tst = applyScaler(x2tst, SCALER_X2)


FILE_SCALER_Y = out_folder+"/scalery.pkl" 

if not os.path.exists(FILE_SCALER_Y):
    print('Computing the scaler from training data...')
    SCALER_Y = trainScaler(ytra, 'minmax') 
    print('Saving the estimated scaler...')
    with open(FILE_SCALER_Y, 'wb') as file:
        pickle.dump(SCALER_Y, file)

else:
    print('Loading scaler from file...')
    with open(FILE_SCALER_Y, 'rb') as file:
        SCALER_Y = pickle.load(file)

print('Scaling the input data according to the training set...')
ytra = applyScaler(ytra, SCALER_Y)
yval = applyScaler(yval, SCALER_Y)
ytst = applyScaler(ytst, SCALER_Y)

print('after scaling')
print('range of ytra {}; max {}; min {}'.format(np.ptp(ytra), np.max(ytra), np.min(ytra)))
print('range of ytst {}; max {}; min {}'.format(np.ptp(ytst), np.max(ytst), np.min(ytst)))

### TRAINING ###

print("-Training the CNN...")

NUM_METRICS = 3 # rmse, mse, mae
RESULTS = np.zeros([1, NUM_METRICS])
n=-1 # auxiliary index

dl_loop_start = datetime.datetime.now()

Xtra = x1tra
Xval = x1val
Xtst = x1tst

n+=1

FILE_MODEL = out_folder+"/DAT_CNN_model.npz"
if not os.path.exists(FILE_MODEL):
    print('--Defining the model...')
    model = DAT_CNN(Xtra.shape[1:],x2tra.shape[1:], s3_shape) 

    LEARNING_RATE = LR
    OPTIMIZER = keras.optimizers.Adam(lr=LEARNING_RATE)

    print('--Compiling the model...')
    model.compile(optimizer=OPTIMIZER, loss=LOSS,metrics=METRICS,)

    for layer in model.layers: 
        print(layer.output_shape)

    print('--Training the model...')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=FILE_MODEL, monitor="val_mean_squared_error", verbose=1, mode='min', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_mean_squared_error", patience=40, verbose=1, mode='min'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=True, mode="min"), 
    ]

    history = model.fit(x=[Xtra,x2tra],    
                    y=ytra,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    validation_data=([Xval,x2val], yval),
                    shuffle=True)

    model.save(FILE_MODEL)

else:
    print('--Loading trained model...')
    model = keras.models.load_model(FILE_MODEL, compile=False)
    print('--Compiling the model...')
    model.compile( optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS,)

print('--Predicting test...')
ypred = model.predict([Xtst,x2tst]) 

print('range of ypred {}; max {}; min {}:'.format(np.ptp(ypred), np.max(ypred), np.min(ypred)))
print('--Evaluating the model...')
RESULTS[n,:] = compute_metrics(ypred, ytst, verbose=1)

ypred_inv = SCALER_Y.inverse_transform(ypred)
print('range of ypred_inv {}; max {}; min {}:'.format(np.ptp(ypred_inv), np.max(ypred_inv), np.min(ypred_inv)))
with open(timer_file, 'a') as f:
    f.write('{}\t{}\t{}\t{}\n'.format(datetime.datetime.now() - dl_loop_start, np.max(ypred_inv), np.min(ypred_inv)))

## SAVING QUANTITATIVE RESULTS ##
print('Saving global table of results...')
FILE_RESULTS = out_folder+"/RESULTS_{}.txt".format(script_name)
np.savetxt(FILE_RESULTS, RESULTS, fmt='%.8f', delimiter='\t')

#### ELAPSED TIME ####
end_time = datetime.datetime.now()
print('Start time: {} \nEnd time: {} \nTime elapsed: {}'.format(begin_time, end_time, end_time - begin_time))
