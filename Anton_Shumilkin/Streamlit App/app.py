import streamlit as st
import os

import cv2
# import skimage as ski
# import imgviz

# from core import *
import core
# from examples import EXAMPLES
from settings import CLASS_LABELS, MODELS_GDRIVE, EXAMPLES

# import numpy as np
import pandas as pd
# import tempfile


from yt_dlp import YoutubeDL
import ffmpeg


###
### APP UI
###

# 📷 🎞️ 🗺️ 🌍 🤖 🧾 📊 📈 🖥️ 💻 🔍
st.set_page_config('EvoDrone 🛸', '🛸')

st.markdown(
    '''
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
    ''', 
    unsafe_allow_html=True
)

# Load pretrained models and start new tensorflow session
# session, MODELS = load_models(MODELS_GDRIVE)
session, MODELS = core.load_models()

src_help = '''
Select the **Video Source** and then
- post a `video URL`
- or upload a `video file`
'''  # has to be moved outside of function for markdown to work


def ui_main():
    '''Draw main UI layout'''
    st.title('EvoDrone')
    
    st.sidebar.title('Options')

    MODE_OPTIONS = {
        '🛸 Demo': page_demo,
        '🤖 Convert': page_convert_video,
        '🔴 YouTube': page_youtube
    }
    option_source = st.sidebar.selectbox('Mode:', [*MODE_OPTIONS], 
                                         help=src_help)

    option_task = st.sidebar.selectbox('Task:', ['Segment', 'Detect'])
    
    global option_model  # will be used in `ui_recognition()`
    option_model = st.sidebar.radio('Model:', [m for m in MODELS if MODELS[m].task == option_task.lower()])

    global model
    model = MODELS[option_model]  # get the currently selected model

    st.sidebar.markdown('---')

    global draw_titles
    draw_titles = st.sidebar.checkbox('Draw titles')
    # with st.sidebar.expander('wat'):
        # st.code('wat')

    # Show class labels for segmentation
    if option_task == 'Segment':
        show_classes(CLASS_LABELS)

    # Show summary of the selected model
    if hasattr(model, 'summary'):
        global INPUT_SHAPE, IMAGE_SIZE
        INPUT_SHAPE = model.input_shape[1:]  # (288, 512, 3)
        IMAGE_SIZE = INPUT_SHAPE[:-1]
        with st.sidebar.expander('Model Summary'):
            st.caption(f'Input size: {model.input_shape[1:-1]}')

    MODE_OPTIONS[option_source]()  # access page UI for the selected source
    

def show_classes(class_labels):
    '''Display classes with rendered colors'''
    # Convert the class_labels to a DataFrame
    df = pd.DataFrame(list(class_labels.keys())[1:], columns=['Class'])
    df['Color'] = ''

    # Define a function to apply a custom style to each cell based on a list of colors
    def color_cells(df, colors):
        styled_df = pd.DataFrame(index=df.index, columns=df.columns)
        styled_df['Color'] = [f'background-color: rgb{colors[cl]}' for cl in df['Class']]
        return styled_df

    # Apply the custom style to the 'Color' column
    styled_df = df.style.apply(color_cells, colors=class_labels, axis=None)

    # Display the styled table
    st.sidebar.dataframe(styled_df, use_container_width=True, hide_index=True)


def page_youtube():
    '''Obtain video from YouTube'''
    with st.expander('Choose Example Preset'):
        example = EXAMPLES[st.selectbox('Example:', {k: v for k, v in EXAMPLES.items()}, index=0)]

    col_url, col_start_from = st.columns([5, 2])
    url = col_url.text_input('YouTube video URL:', example['url'])
    start_from = col_start_from.number_input(
        'Start From:', 
        min_value=0.0, step=0.5, format='%f', value=example['start'], 
        help='Time shift from the beginning (in seconds)'
    )
    
    if url:
        try:
            with YoutubeDL({'format': 'best'}) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:
            st.error(e)
        else:
            str_title = f"<div style='float: left; text-align: left; width=50%'>\
            **Title:** [{info['title']}]({url})</div>"
            str_duration = f"<div style='float: right; text-align: right; width=50%'>\
            **Overall Duration:** {info['duration']} sec.</div>"
            st.write(f"<small>{str_title + str_duration}</small>", 
                     unsafe_allow_html=True)

            # video_url = info['requested_formats'][0]['url']

            video_url = info['url']

            out, err = (
                ffmpeg
                .input(video_url, ss=start_from, t=5)
                .output('temp.mp4', vcodec='copy')
                .overwrite_output()
                .run()
            )
            st.video('temp.mp4')

            # audio_wav, audio_np = proc_raw_audio(audio_data)

            # ui_processed_video(audio_wav, audio_np)


def page_demo():
    if 'file_selector_is_expanded' not in st.session_state:
        st.session_state['file_selector_is_expanded'] = True
    file_selector_container = st.sidebar.expander(
        'Choose a video file', 
        expanded=st.session_state['file_selector_is_expanded']
    )

    # Choose file upload mode
    with file_selector_container:
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        upload_mode = st.toggle('Local dir', help='Choosing between uploading and local directory files list', value=True)

        if upload_mode:
            def file_selector(folder_path='.'):
                is_video_file = lambda f: any(f.lower().endswith(ext) for ext in video_extensions)

                video_files = [f for f in os.listdir(folder_path) if is_video_file(f)]

                if not video_files:
                    st.warning('No video files found in the selected directory.')
                    return None

                selected_filename = st.selectbox('Select a video file', video_files, help=f'from {folder_path}')
                return os.path.join(folder_path, selected_filename)

            videofile = file_selector()
            # videofile_name = os.path.split(videofile)[-1]
            file_path_input = st.text_input('Video file path:', videofile)
        else:
            videofile = st.file_uploader('Upload a video', type=video_extensions)
            # videofile_name = videofile.name if videofile else ''

    # Perform videocapture and inference if a video file is present
    if videofile:
        if 'demo_playing' not in st.session_state:
            st.session_state['demo_playing'] = False

        vidcap = cv2.VideoCapture(videofile)  # load the video for capturing

        # Get video properties
        get_prop = lambda p: int(round(vidcap.get(p)))
        frame_count, frame_rate = get_prop(cv2.CAP_PROP_FRAME_COUNT), get_prop(cv2.CAP_PROP_FPS)
        video_length = int(frame_count / frame_rate)
        frame_width, frame_height = get_prop(cv2.CAP_PROP_FRAME_WIDTH), get_prop(cv2.CAP_PROP_FRAME_HEIGHT)

        st.info(' | '.join([
            f'Video: `{videofile}`', 
            f'Length: `{video_length}s`', 
            f'Frames: `{frame_count}`', 
            f'FPS: `{frame_rate}`', 
            f'Size: `{frame_width}x{frame_height}`'
        ]))

        is_yolo = core.is_yolo_model(model)
        task = model.task

        if is_yolo:
            to_resize = False
            image_size = None
        else:
            # Decide whether to resize the video to size acceptable by the model
            to_resize = True if (frame_height, frame_width) != IMAGE_SIZE else False
            image_size = IMAGE_SIZE

        # exclude_classes = col_file.multiselect('Exclude classes', CLASS_LABELS, help=f'help')

        # col_start_from, col_step = st.columns([5, 2])
        col_start_from, _, _, col_step = st.columns([2, 1, 1, 2])
        start_from = col_start_from.number_input(
            'Start From, sec.:', 
            min_value=0.0, max_value=float(video_length), step=0.5, format='%f', value=0.0, 
            help='Time shift from the beginning (in seconds)'
        )
        step = col_step.number_input(
            'Step:', 
            min_value=1, max_value=100, step=1, format='%i', value=5, 
            help='Step (frames)'
        )

        # label_start, label_stop = '⭕ Start', '▣ Stop'
        label_start, label_stop = '🛸 Start', '▣ Stop'

        col_start_button, col_stop_button, _ = st.columns([1, 5, 1])
        start_button = col_start_button.button(label_start, type='primary')
        stop_button = col_stop_button.empty()

        frame_placeholder = st.empty()  # placeholder for the frame display
        frame_time_col, frame_id_col, _ = st.columns([1, 1, 1])
        frame_time_placeholder = frame_time_col.empty()
        frame_id_placeholder = frame_id_col.empty()

        # DRAW FIRST FRAME
        frame_id = int(start_from * (frame_rate + 1))  # starting frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        frame = vidcap.read()[1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, use_column_width=True, clamp=True)

        num_frames = frame_count  # 100

        if start_button:
            st.session_state['demo_playing'] = True
            stop_button = col_stop_button.button(label_stop)

            for i in range(int(num_frames - start_from)):
                if step > 1:
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # jump <step> frames forward
                    frame_id = vidcap.get(cv2.CAP_PROP_POS_FRAMES)

                hasFrame, frame = vidcap.read()  # get the next frame from the video
                if not hasFrame or stop_button:
                    break

                # MODEL INFERENCE
                frame_masked = core.predict_frame(model, frame, 
                                                  task=task,
                                                  is_yolo=is_yolo, 
                                                  to_resize=to_resize, 
                                                  image_size=image_size, 
                                                  draw_titles=draw_titles)

                # caching.clear_cache()

                # DRAW IMAGE
                frame_placeholder.image(frame_masked, use_column_width=True, clamp=True)  # channels='RGB')

                seconds_to_time = lambda s: f'{s // 3600 :02}:{s % 3600 // 60 :02}:{s % 60 :02}'
                current_time = seconds_to_time(int(vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
                full_time = seconds_to_time(video_length)
                frame_time_placeholder.markdown(
                    f"<mark style='text-align: right'>time: `{current_time} / {full_time}`</mark>", 
                    unsafe_allow_html=True
                )
                frame_id_placeholder.markdown(
                    f"<mark style='text-align: right'>frame: `{int(frame_id)} / {frame_count}`</mark>", 
                    unsafe_allow_html=True
                )

                frame_id += step

        stop_button = col_stop_button.empty()
        st.session_state['demo_playing'] = False

        
        output_file = f'{os.path.splitext(videofile)[0]}_{task}_masked.avi'
        label_start_convertion, label_stop_convertion = '🤖 Convert', '▣ Cancel'
        col_start_convertion_button, col_stop_conversion_button, _ = st.columns([1, 5, 1])
        start_conversion_button = col_start_convertion_button.button(label_start_convertion)
        stop_conversion_button = col_stop_conversion_button.empty()

        if start_conversion_button:
            core.process_video(model, 
                               videofile, 
                               output_file, 
                               frame_rate, 
                               (frame_width, frame_height),

                               task=task, 
                               is_yolo=is_yolo, 
                               to_resize=to_resize, 
                               image_size=image_size, 
                               draw_titles=draw_titles,
                               col_stop_conversion_button=col_stop_conversion_button)


def page_convert_video():
    '''Convert a full video file'''
    pass

    # start_button_disabled = False
    # st.session_state['start_button_disabled'] = False
    # with sr.Microphone(device_index=MIC_OPTIONS[option_mic], 
    #                    sample_rate=SAMPLE_RATE) as source:
    #     with st.spinner('< ..speak now.. >'):
    #         audio = r.listen(source)

    # recognized = recognize_GSR(r, audio)
    # # recognized = recognize_GCS(r, audio)
    # if recognized:
    #     st.info(recognized.capitalize())

    # audio_data = audio.get_wav_data()
    # audio_wav, audio_np = proc_raw_audio(audio_data)

    # ui_processed_video(audio_wav, audio_np, show_button=False)




    # with st.expander('Choose Example Preset'):
    #     example = EXAMPLES[st.selectbox('Example:', [*examples_filtered], index=0)]

    # col_url, col_start_from = st.columns([5, 2])
    # url = col_url.text_input('YouTube video URL:', example['url'])
    # start_from = col_start_from.number_input(
    #     'Start From:', 
    #     min_value=0.0, step=0.5, format='%f', value=example['start'], 
    #     help='Time shift from the beginning (in seconds)'
    # )
    
    # if url:
    #     try:
    #         with YoutubeDL({'format': 'best+bestaudio'}) as ydl:
    #             info = ydl.extract_info(url, download=False)
    #     except Exception as e:
    #         st.error(e)
    #     else:
    #         str_title = f"<div style='float: left; text-align: left; width=50%'>\
    #         **Title:** [{info['title']}]({url})</div>"
    #         str_duration = f"<div style='float: right; text-align: right; width=50%'>\
    #         **Overall Duration:** {info['duration']} sec.</div>"
    #         st.write(f"<small>{str_title + str_duration}</small>", 
    #                  unsafe_allow_html=True)

    #         # video_url = info['requested_formats'][0]['url']

    #         video_url = info['url']

    #         out, err = (
    #             ffmpeg
    #             .input(video_url, ss=start_from, t=5)
    #             .output('temp.mp4', vcodec='copy')
    #             .overwrite_output()
    #             .run()
    #         )
    #         st.video('temp.mp4')

    #         # audio_wav, audio_np = proc_raw_audio(audio_data)

    #         # ui_processed_video(audio_wav, audio_np)


def ui_processed_video(audio_wav, audio_np, show_button=True):
    '''UI to show sound processing results'''
    _, center_h, _ = st.columns([1, 2, 1])
    center_h.header('Processed Video')

    st.audio(audio_wav)
    features = get_features(audio_np)
    
    if show_button:
        _, center, _ = st.columns([1, 1, 1])
        if center.button('🔮 Perform Segmentation'):
            ui_recognition(features)
    else:
        ui_recognition(features)


def ui_recognition(features):
    '''Recognition UI interface'''
    with st.spinner('Predicting..'):
        probs, accent = recognize_accent(MODELS[option_model], features)

    st.success(f'🔢 Class: **`{accent.upper()}`**')

    # Compute and display prediction probabilities
    probs_df = pd.DataFrame.from_dict(MODELS[option_model].accents_dict, 
                                      orient='index', columns=['Accent'])
    probs_df['Probability'] = [f'{x:.4f}%' for x in probs]
    probs_df.set_index('Accent', inplace=True)
    probs_df.sort_values('Probability', ascending=False, inplace=True)
    
    st.write('🎲 Prediction Probabilities:')
    # st.success(f'API response: {response.text}')

    get_tone = lambda c: hex(int(255 - 255 * c))[2:]
    make_color = lambda c: f'#FFFF{get_tone(c)}'.ljust(7, '0')
    highlight_max = lambda cells: [
        f'background-color: {make_color(float(c[:-1]))}'
        for c in cells
    ]
    st.dataframe(probs_df.style.apply(highlight_max))


if __name__ == "__main__":
    ui_main()
