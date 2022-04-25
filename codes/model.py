from tensorflow import keras
from attention_keras import channel_attention3d, spatial_attention3d, attention2d


def DAT_CNN(INPUT_A, INPUT_B, n_out=1):
    ## Feature extraction
    model_input_A = keras.Input(shape = INPUT_A, name="Patches_3D_S2")

    # HEAD 1
    CA1 = channel_attention3d()(model_input_A)
    SA1 = spatial_attention3d()(CA1)
    C1 = keras.layers.Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", data_format='channels_last', input_shape=INPUT_A)(CA1)
    B1 = keras.layers.BatchNormalization()(C1)
    A1 = keras.layers.Activation('relu')(B1)   
    P1 = keras.layers.MaxPooling3D((2, 2, 1), strides=(1, 1, 1), padding='same',data_format='channels_last')(A1)
    # HEAD 2
    C2 = keras.layers.Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", data_format='channels_last')(P1)
    B2 = keras.layers.BatchNormalization()(C2)
    A2 = keras.layers.Activation('relu')(B2)   
    P2 = keras.layers.MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding='same',data_format='channels_last')(A2)
    # BODY 1 
    C3 = keras.layers.Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", data_format='channels_last')(P2)
    B3 = keras.layers.BatchNormalization()(C3)
    A3 = keras.layers.Activation('relu')(B3)   
    C4 = keras.layers.Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", data_format='channels_last')(A3)
    B4 = keras.layers.BatchNormalization()(C4)
    A4 = keras.layers.Activation('relu')(B4)  
    P3 = keras.layers.MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding='same',data_format='channels_last')(A4)
    # BODY 2
    C5 = keras.layers.Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", data_format='channels_last')(P3)
    B5 = keras.layers.BatchNormalization()(C5)
    A5 = keras.layers.Activation('relu')(B5)   
    C6 = keras.layers.Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", data_format='channels_last')(A5)
    B6 = keras.layers.BatchNormalization()(C6)
    A6 = keras.layers.Activation('relu')(B6)  
    P4 = keras.layers.MaxPooling3D((2, 2, 3), strides=(2, 2, 3), padding='same',data_format='channels_last')(A6)
    
    # HEAD 1 S3
    model_input_B = keras.Input(shape = INPUT_B, name="Patches_3D_S3")
    I1 = keras.layers.Reshape((1,1,4))(model_input_B)
    SA2 = attention2d()(I1)
    C12 = keras.layers.Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding="same", data_format='channels_last', input_shape=INPUT_B)(SA2)
    B12 = keras.layers.BatchNormalization()(C12)
    A12 = keras.layers.Activation('relu')(B12)   
    P12= keras.layers.MaxPooling2D((1, 1), strides=(1, 1), padding='same',data_format='channels_last')(A12)
    # BODY 1 S3
    C32 = keras.layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding="same", data_format='channels_last')(P12)
    B32 = keras.layers.BatchNormalization()(C32)
    A32 = keras.layers.Activation('relu')(B32)   
    C42 = keras.layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding="same", data_format='channels_last')(A32)
    B42 = keras.layers.BatchNormalization()(C42)
    A42 = keras.layers.Activation('relu')(B42)  
    P32 = keras.layers.MaxPooling2D((1, 1), strides=(1, 1), padding='same',data_format='channels_last')(A42)#(A52)

    J1 = keras.layers.concatenate([P32,P32],axis=3)
    J2 = keras.layers.Reshape((1,1,1,256))(J1) 
    P4 = keras.layers.Reshape((1,1,1,1024))(P4) 
    J3 = keras.layers.concatenate([P4,J2])
    # TAIL
    F1 = keras.layers.Flatten()(J3)
    D1 = keras.layers.Dense(1024)(F1)
    B7 = keras.layers.BatchNormalization()(D1)
    A7 = keras.layers.Activation('relu')(B7)
    model_output = keras.layers.Dense(n_out, activation='linear')(A7) 
    model = keras.Model(inputs=[model_input_A, model_input_B], outputs = model_output,name="dat_model")
    model.summary()
    return model