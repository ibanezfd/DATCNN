
import numpy as np
import cv2
import rasterio
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def trainScaler(features, scaler_name="standard"):
    n_samples = features.shape[0]
    vec_features = np.reshape(features,(n_samples,-1))
    if scaler_name.lower()=="standard":
        scaler = StandardScaler().fit(vec_features)
    elif scaler_name.lower()=="minmax":
        scaler = MinMaxScaler().fit(vec_features)
    elif scaler_name.lower()=="robust":
        scaler = RobustScaler().fit(vec_features)
    else:
        raise Exception("Undefined scaler name (i.e. 'standard', 'minmax')!")
    return scaler

def applyScaler(features, scaler):
    n_samples = features.shape[0]
    vec_features = np.reshape(features,(n_samples,-1))
    scaled_vec_features = scaler.transform(vec_features)
    scaled_features = np.reshape(scaled_vec_features,features.shape)
    return scaled_features

def compute_s2_product(s2_img, prod_name='ndvi'):

    s2_img = s2_img.astype(np.float32)
    np.seterr(divide='ignore', invalid='ignore')
    s2_img_prod = (s2_img[:,:,7] - s2_img[:,:,3])/(s2_img[:,:,7] + s2_img[:,:,3])
    s2_img_prod = np.clip(s2_img_prod, -1, 1) # to avoid any possible wrong value
    
    return s2_img_prod

def compute_s3_product(s3_img, prod_name='ndvi'):

    s3_img = s3_img.astype(np.float32)
    np.seterr(divide='ignore',invalid='ignore')
#https://land.copernicus.eu/global/sites/cgls.vito.be/files/products/CGLOPS1_ATBD_NDVI300m-V2_I1.00.pdf
    R = (s3_img[:,:,6] + s3_img[:,:,7] + s3_img[:,:,8] + s3_img[:,:,9])/4
    NIR = (s3_img[:,:,15]+s3_img[:,:,16]+s3_img[:,:,17])/3
    s3_img_prod = (NIR-R)/(NIR+R)
    s3_img_prod = np.clip(s3_img_prod,-1, 1)

    return s3_img_prod

def s2s3_patches_s3_pixel_5(s3_name, s2_temporal, s3_temporal, prod_name='ndvi', RF=3):

    # t dat s2 75x75 to 25x25
    # t-1 data s3 5x5
    # S2 processing
    s2_data = np.zeros((5490//RF,5490//RF,12,len(s2_temporal)),dtype=np.float32) #Data to store everything
    s3_data = np.zeros((366,366,len(s3_temporal)),dtype=np.float32)
    # S2 temporal processing
    for i in range(len(s2_temporal)):
        print('--Loading s2 file "{}"'.format( s2_temporal[i]))
        s2 = rasterio.open(s2_temporal[i]).read().astype('uint16') # 12, rows, cols)
        s2 = np.moveaxis(s2, source=0, destination=-1) # (rows, cols, 12)

        print('--Product pre-processing...')
        s2 = np.clip(s2.astype(np.float32)/10000, 0, 1)
        s2 = cv2.resize(s2, (s2.shape[0]//RF, s2.shape[1]//RF), cv2.INTER_LANCZOS4).astype(np.float32) # reducing s2_prod to s3 size
        s2_data[:,:,:,i] = s2

        print('--Product s2 added to temporal set.')
    
    # S3 temporal processing
    for i in range(len(s3_temporal)):
        print('--Loading s3 file "{}"'.format(s3_temporal[i]))
        s3 = rasterio.open(s3_temporal[i]).read().astype('float32') # (21, rows, cols)
        s3 = np.moveaxis(s3, source=0, destination=-1) # (rows, cols, 21)
        s3 = np.clip(s3.astype(np.float32), 0, 1) # recovering the original reflectances of s2 (see https://forum.step.esa.int/t/dn-to-reflectance/15763/8, https://gis.stackexchange.com/questions/233874/what-is-the-range-of-values-of-sentinel-2-level-2a-images)

        print('--Computing s3 product...')
        s3 = compute_s3_product(s3, prod_name)
        s3_data[:,:,i] = s3

        print('--Product s3 added to temporal set.')
     
    # S3 output processing
    print('--Loading s3 file "{}"'.format(s3_name))
    s3 = rasterio.open(s3_name).read().astype('float32') # (21, rows, cols)
    s3 = np.moveaxis(s3, source=0, destination=-1) # (rows, cols, 21)
    s3 = np.clip(s3.astype(np.float32), 0, 1) # recovering the original reflectances of s2 (see https://forum.step.esa.int/t/dn-to-reflectance/15763/8, https://gis.stackexchange.com/questions/233874/what-is-the-range-of-values-of-sentinel-2-level-2a-images)

    print('--Computing s3 OUTPUT product...')
    s3_prod = compute_s3_product(s3, prod_name)
    list_s2_3D_patches = []
    list_s3_3D_patches = []
    list_s3_prod_vals = []

    # Adapt data to cnn input
    s2_data = np.moveaxis(s2_data,source=2, destination = 3)

    MAX_PATCH_SIZE_S3 = 1
    PATCH_SIZE_S3 = 1
    STEP_S3 = 1

    MAX_PATCH_SIZE_S2=(MAX_PATCH_SIZE_S3*15)//RF 
    PATCH_SIZE_S2 = (PATCH_SIZE_S3*15)//RF
    STEP_S2 = (STEP_S3*15)//RF

    print('--Extracting coupled patches/values...')
    for r in range(MAX_PATCH_SIZE_S2//2,s2.shape[0]-MAX_PATCH_SIZE_S2//2+1,STEP_S2):
        for c in range(MAX_PATCH_SIZE_S2//2,s2.shape[1]-MAX_PATCH_SIZE_S2//2+1,STEP_S2):
            ini_row_S2 = r-PATCH_SIZE_S2//2
            end_row_S2 = (r+PATCH_SIZE_S2//2)+1
            ini_col_S2 = c-PATCH_SIZE_S2//2
            end_col_S2 = (c+PATCH_SIZE_S2//2)+1
            if r == 0:
                r2 = 0
            else:
                r2 = r//STEP_S2
            if c == 0:
                c2 = 0
            else:
                c2 = c//STEP_S2
            list_s2_3D_patches.append(s2_data[ini_row_S2:end_row_S2,ini_col_S2:end_col_S2,:,:])
            list_s3_3D_patches.append(s3_data[r2,c2,:])
            list_s3_prod_vals.append(s3_prod[r2,c2])
    
    print('--Stacking patches and values...')
    x1 = np.stack(list_s2_3D_patches, axis=0) # (num_patches, rows, cols, 21)
    x2 = np.stack(list_s3_3D_patches, axis=0) # (num_patches, rows, cols)
    y = np.stack(list_s3_prod_vals, axis=0) # (num_patches, 1 (product value))
    del list_s2_3D_patches, list_s3_prod_vals

    return [x1, x2, y]

def compute_metrics(ypred, ytst, out_name=None, verbose=0):

    mse  = sklearn.metrics.mean_squared_error(ypred.astype(np.float32), ytst.astype(np.float32))
    mae  = sklearn.metrics.mean_absolute_error(ypred.astype(np.float32), ytst.astype(np.float32))
    rmse = math.sqrt(mse)

    RESULTS = [ "RMSE: {}".format(rmse),
                "MSE:  {}".format(mse),
                "MAE:  {}".format(mae)]

    if out_name:
        with open(out_name, "w") as f:
            f.write("\n".join(RESULTS))

    if verbose:
        print(RESULTS)

    return [rmse, mse, mae]