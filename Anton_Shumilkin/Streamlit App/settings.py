CLASS_LABELS = {
    'Background' : (0, 0, 0), 
    'Field'      : (0.294, 0, 0.51), 
    'Forest'     : (1, 0, 0), 
    'Grass'      : (0, 0, 1), 
    'Power lines': (0, 0.502, 0), 
    'Road'       : (1, 0.549, 0),
    'Water'      : (1, 0.753, 0.796)
}

CLASS_LABELS = {
    name: tuple(int(ch * 255) for ch in color) 
    for name, color in CLASS_LABELS.items()
}


MODELS_GDRIVE = {
    # 'unet_pspnet_model_best.h5': 'https://drive.google.com/uc?id=1Hd1Qjeg9e2A6a-FLXBqqgVogpWOI86-p',  # 
    'unet_model_best__UNet_PSP_plus.h5':          'https://drive.google.com/uc?id=1ggG4ak73nX08kcOPZbZLe4_U1fSP9sdM',   # 

    'best_detect_YOLO8.pt':                       'https://drive.google.com/uc?id=1nQs75C05BChBUl_dh7hIDgiZl9Wb5Kvi',  #
    'bisenetv2_fcn_4xb4-80k_evod2-1024x1024.pth': 'https://drive.google.com/uc?id=1P0IU8FEutFo8jOc5cKZ6AkTBY2t-Ef14',  #
    'psp_unet_model_best__12.10.2023.h5':         'https://drive.google.com/uc?id=1p8uC1zW6LAdtkePH_FbUgbFA7a3IPbqv',  #
    'unet_model_best__08.10.2023_60ep.h5':        'https://drive.google.com/uc?id=1Fxo_XMDuseHJghD6faWp88VJNbHtUUWY',  #
    'weights_yolo8s-seg.pt':                      'https://drive.google.com/uc?id=1wwuxuhNUKVrjRtiYjyFoXmK_Ult-5kBu',  #
    'weights_yolo8x-seg.0.87.pt':                 'https://drive.google.com/uc?id=1ZmZCv1koxcmdH0MIcYqDeKFiqKPolEMz',  #
    'Yv8xSeg_67-71_best.pt':                      'https://drive.google.com/uc?id=1II0vSTvVdtACXVBjKKqFEJDq4lykLFQK'  #
}


VIDEOS_GDRIVE = {
    'Убранное поле+поле под паром.mp4':                'https://drive.google.com/uc?id=1JUnXrS2goXXOxljCNA3QGXLXUn6opUv6',  # video_1
    'Перепаханное поле+дорога+дома+под паром+лес.mp4': 'https://drive.google.com/uc?id=16ujmFjLvcsYsyUn9mvEelITAUWx2DbK_',  # video_2
    'Убранное поле+панорама.mp4':                      'https://drive.google.com/uc?id=1l3Vj9Dy--TBcGkNY4G8sTRNc2hMLeYjI',  # video_5
}


EXAMPLES = {
    'Test 1: Forest + Road': {
        'url':    'https://www.youtube.com/watch?v=5LAgrI-hH2c', 
        'start':  0.0
    }, 
    'Test 2: Fields': {
        'url':    'https://www.youtube.com/watch?v=04z02TNjio0', 
        'start':  0.0
    }, 
    
    'Test 3: Forest': {
        'url':    'https://www.youtube.com/watch?v=Woo-9cduWiE', 
        'start':  0.0
    }
}
