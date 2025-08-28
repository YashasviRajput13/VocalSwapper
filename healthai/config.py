# config.py
# import os

# class Config:
#     BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
#     DATA_DIR      = os.path.join(BASE_DIR, 'data')
#     OUTPUT_DIR    = os.path.join(BASE_DIR, 'output')
#     MODEL_DIR     = os.path.join(OUTPUT_DIR, 'models')
#     LOG_DIR       = os.path.join(OUTPUT_DIR, 'logs')
#     MODALITIES    = ['xray','ct','mri','ultrasound','fundus']
#     DISEASES      = ['pneumonia','tuberculosis','lung_cancer','alzheimer',
#                      'brain_tumor','fracture','fatty_liver','cardio_calc','retinopathy']
#     IMAGE_SIZE_2D = (224,224)
#     IMAGE_SIZE_3D = (128,128,128)
#     BATCH_SIZE    = 16
#     EPOCHS        = 100
#     LR            = 1e-4
#     FEDERATED     = True
#     FL_SERVER     = "localhost:8080"
#     GAN_LATENT    = 100
#     GAN_EPOCHS    = 200
#     GAN_BATCH_SIZE= 16
#     GAN_LR        = 1e-4



import os

class Config:
    BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR      = os.path.join(BASE_DIR, 'data')
    OUTPUT_DIR    = os.path.join(BASE_DIR, 'output')
    MODEL_DIR     = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR       = os.path.join(OUTPUT_DIR, 'logs')
    MODALITIES    = ['xray','ct','mri','ultrasound','fundus']
    DISEASES      = ['pneumonia','tuberculosis','lung_cancer','alzheimer',
                     'brain_tumor','fracture','fatty_liver','cardio_calc','retinopathy']
    IMAGE_SIZE_2D = (224,224)
    IMAGE_SIZE_3D = (128,128,128)
    BATCH_SIZE    = 16
    EPOCHS        = 100
    LR            = 1e-4
    FEDERATED     = True
    FL_SERVER     = "localhost:8080"
    GAN_LATENT    = 100
    GAN_EPOCHS    = 200
    GAN_BATCH_SIZE= 16
    GAN_LR        = 1e-4
    AUDIO_CACHE   = os.path.join(BASE_DIR, 'media', 'audio_cache')
    MEDIA_ROOT    = os.path.join(BASE_DIR, 'media')