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


MODELS_URLS = [
    'https://drive.google.com/uc?id=1Hd1Qjeg9e2A6a-FLXBqqgVogpWOI86-p',  # unet_pspnet_model_best.h5
    'https://drive.google.com/uc?id=1PWerQAn10BIRZ-quZ7h2A82Aykn-IZ17'   # unet_model2_best_add+.h5
]


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
