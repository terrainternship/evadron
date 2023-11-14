import streamlit as st

import cv2
import imgviz
from ultralytics.utils.ops import scale_image
from ultralytics import YOLO
import distutils.version

import os
from pathlib import Path
import numpy as np
import gdown
from stqdm import stqdm

import tensorflow as tf
from tensorflow.keras.models import load_model as load_model_tensorflow

from settings import CLASS_LABELS, MODELS_GDRIVE, VIDEOS_GDRIVE

###
### APP CORE
###


TF_HASH_FUNCS = {
    tf.compat.v1.Session: id
}


BATCH_SIZE = 16
MODELS_DIR = 'models'


###==============================================


def is_yolo_model(model):
    # return str(type(model).__name__) == 'YOLO'
    return isinstance(model, YOLO)


def download_videos(videos_urls=VIDEOS_GDRIVE, videos_dir='.'):
    '''Download pretrained models'''

    # os.makedirs(videos_dir, exist_ok=True)
    # videos_list = []
    # Download each model
    for video_name, video_url in videos_urls.items():
        output_path = f'{videos_dir}/{video_name}'
        if not os.path.exists(output_path):
            with st.spinner(f"Downloading video {video_name}... this may take a while! \n Don't stop!"):
                # Download the file from Google Drive
                gdown.download(video_url, output_path, quiet=False)
        # models_list.append(output_path)
    # return models_list


def download_models(models_urls=MODELS_GDRIVE, models_dir=MODELS_DIR):
    '''Download pretrained models'''

    os.makedirs(models_dir, exist_ok=True)
    models_list = []
    # Download each model
    for model_name, model_url in models_urls.items():
        output_path = f'{models_dir}/{model_name}'
        if not os.path.exists(output_path):
            with st.spinner(f"Downloading model {model_name}... this may take a while! \n Don't stop!"):
                # Download the file from Google Drive
                gdown.download(model_url, output_path, quiet=False)
        models_list.append(output_path)
    return models_list


@st.cache_resource
def load_models(models_urls=None, models_dir=MODELS_DIR):
    '''Load pretrained models'''

    # Open a new TensorFlow session
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    session = tf.compat.v1.Session(config=config)

    with session.as_default():
        models = {}
        save_dest = Path(models_dir)
        save_dest.mkdir(exist_ok=True)

        if models_urls:
            models_list = download_models(models_urls=models_urls, models_dir=models_dir)
        else:
            # Load from the local dir
            models_list = [os.path.join(models_dir, f) for f in os.listdir(models_dir) 
                           if os.path.isfile(os.path.join(models_dir, f)) 
                           and f.split('.')[-1] in ('h5', 'pt', 'pth')]
            print('Found model files locally:')
            [print(os.path.basename(m)) for m in models_list]

        for model_file in models_list:
            model_ext = os.path.splitext(model_file)[-1]

            if model_ext == '.h5':
                model = load_model_tensorflow(model_file)
                model.task = 'segment'
            elif model_ext == '.pt':
                model = YOLO(model_file)
            elif model_ext == '.pth':
                pass

            model_title = os.path.splitext(os.path.basename(model_file))[0]
            models.update({model_title: model})

            # model.name_ = model_title
    return session, models


def predict_frame(model, frame, 
                  task=None, 
                  is_yolo=None, 
                  to_resize=False, 
                  image_size=None, 
                  draw_titles=False, 
                  batch_size=BATCH_SIZE):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert color palette to RGB

    if task == 'detect':
        frame_masked = model(frame)[0].plot(boxes=True, masks=True, labels=True, conf=False)
    else:
        if is_yolo is None:
            is_yolo = is_yolo_model(model)

        if is_yolo:
            results = model(frame)[0]
            names = results.names
            classes = [int(c) for c in results.boxes.cls]  # get predicted classes indices

            if results.masks:
                masks_arr = results.masks.cpu().data.numpy()
                mask_size = masks_arr.shape[1:]
                multiclass_mask = np.zeros(mask_size, dtype=np.uint8)
                for class_id, binary_mask in zip(classes, masks_arr):  # CLASS_LABELS.values()
                    # st.write(class_id, binary_mask.shape, multiclass_mask.shape)
                    # if not class_id == 0:
                    multiclass_mask = np.where(binary_mask, class_id - 1, multiclass_mask)
                    # mask = np.all(image_arr == i, axis=0)
                    # result[:, mask] = class_color
            else:
                multiclass_mask = np.zeros((*frame.shape[:-1], 1), dtype=np.uint8)
        else:
            # Resize the frame if needed
            frame_pred = cv2.resize(frame, image_size[::-1]) if to_resize else frame
            predict = np.argmax(
                model.predict(
                    [frame_pred[None, ...] / 255.0],
                    # batch_size=BATCH_SIZE
                ), axis=-1
            )
            multiclass_mask = predict[0].astype(np.uint8)

        # Scale the mask to the original frame size
        multiclass_mask = scale_image(multiclass_mask, frame.shape[:-1]).squeeze(axis=2)
        # Create mask overlay for the frame image
        frame_masked = imgviz.label2rgb(multiclass_mask, frame, 
                                        label_names=dict(list(enumerate(CLASS_LABELS.keys()))[1:]),  # omit background label in the legend
                                        colormap=np.array(list(CLASS_LABELS.values())), alpha=0.5, 
                                        font_size=int(frame.shape[1] / 40), font_path=None, 
                                        thresh_suppress=0, 
                                        loc='centroid' if draw_titles else 'rb')

    # response = requests.post(url='http://127.0.0.1:8000/model-predict', data=json.dumps(inputs))

    return frame_masked#, response


def process_video(model, 
                  video_path, 
                  output_path, 
                  frame_rate, 
                  frame_size, 

                  task=None, 
                  is_yolo=None, 
                  to_resize=False, 
                  image_size=None, 
                  draw_titles=True, 
                  batch_size=BATCH_SIZE,
                  col_stop_conversion_button=None):

    stop_conversion_button = col_stop_conversion_button.button('Cancel')

    vidcap = cv2.VideoCapture(video_path)

    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_file = output_path

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # fourcc = int(vidcap.get(cv2.CAP_PROP_FOURCC))  # use the original codec


    # Create VideoWriter object to write the resized video
    out = cv2.VideoWriter(
        output_file,
        fourcc,                       # video codec
        frame_rate,                   # fps
        frame_size,  # frame size
        True
    )
    if not out.isOpened():
        st.error(f'Error: Could not open video file ({output_file}) for writing.')

    num_frames = frame_count
    # Create a tqdm progress bar
    # progress_bar = tqdm(
    #     total=frame_count, position=0, leave=True, desc='Processing Frames'
    # )
    progress_bar = stqdm(
        total=frame_count, position=0, leave=True, desc='Processing Frames'#, st_container=st.sidebar
    )

    # detected = []  # a list of detected objects
    task = model.task
    is_yolo = is_yolo_model(model)

    # frame_id = 1
    for _ in range(num_frames):
        # vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # jump <step> frames forward
        # frame_id = vidcap.get(cv2.CAP_PROP_POS_FRAMES)

        hasFrame, frame = vidcap.read()
        # st.write(hasFrame, frame.shape)

        if not hasFrame or stop_conversion_button:
            break

        # MODEL INFERENCE
        frame_masked = predict_frame(model, frame, 
                                     task=task, 
                                     is_yolo=is_yolo, 
                                     to_resize=to_resize, 
                                     image_size=image_size, 
                                     draw_titles=draw_titles)
        # results = model.detect([image], verbose=1)
        # r = results[0]
        # output = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        # detected.append(r['class_ids'])

        frame_masked = cv2.cvtColor(frame_masked, cv2.COLOR_RGB2BGR)

        out.write(frame_masked)
        progress_bar.update(1)

        # frame_id += 1
        
    out.release()

    st.info(f'Output video is created at {output_file}')

    # return detected


def predict_api():
    if st.button('Predict'):
        # files=files, params=payload
        response = requests.post(url='http://127.0.0.1:8000/model-predict', data=json.dumps(inputs))
        st.success(f'The predicted data is : {response.text}')
