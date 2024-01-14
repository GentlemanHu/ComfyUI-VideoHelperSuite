from notifier.notify import notifyAll
import os
import sys
import json
import subprocess
import numpy as np
import re
from typing import List
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pathlib import Path

import folder_paths
from .logger import logger
from .image_latent_nodes import *
from .load_video_nodes import LoadVideoUpload, LoadVideoPath
from .load_images_nodes import LoadImagesFromDirectoryUpload, LoadImagesFromDirectoryPath
from .batched_nodes import VAEEncodeBatched, VAEDecodeBatched
from .utils import ffmpeg_path, get_audio, hash_path, validate_path

folder_paths.folder_names_and_paths["VHS_video_formats"] = (
    [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats"),
    ],
    [".json"]
)

def gen_format_widgets(video_format):
    for k in video_format:
        if k.endswith("_pass"):
            for i in range(len(video_format[k])):
                if isinstance(video_format[k][i], list):
                    item = [video_format[k][i]]
                    yield item
                    video_format[k][i] = item[0]
        else:
            if isinstance(video_format[k], list):
                item = [video_format[k]]
                yield item
                video_format[k] = item[0]

def get_video_formats():
    formats = []
    for format_name in folder_paths.get_filename_list("VHS_video_formats"):
        format_name = format_name[:-5]
        video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
        with open(video_format_path, 'r') as stream:
            video_format = json.load(stream)
        widgets = [w[0] for w in gen_format_widgets(video_format)]
        if (len(widgets) > 0):
            formats.append(["video/" + format_name, widgets])
        else:
            formats.append("video/" + format_name)
    return formats

def get_format_widget_defaults(format_name):
    video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    results = {}
    for w in gen_format_widgets(video_format):
        if len(w[0]) > 2 and 'default' in w[0][2]:
            default = w[0][2]['default']
        else:
            if type(w[0][1]) is list:
                default = w[0][1][0]
            else:
                #NOTE: This doesn't respect max/min, but should be good enough as a fallback to a fallback to a fallback
                default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[0][1]]
        results[w[0][0]] = default
    return results


def apply_format_widgets(format_name, kwargs):
    video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    for w in gen_format_widgets(video_format):
        assert(w[0][0] in kwargs)
        w[0] = str(kwargs[w[0][0]])
    return video_format

def tensor_to_int(tensor, bits):
    #TODO: investigate benefit of rounding by adding 0.5 before clip/cast
    tensor = tensor.cpu().numpy() * (2**bits-1)
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)

class VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        #Hide ffmpeg formats if ffmpeg isn't available
        if ffmpeg_path is not None:
            ffmpeg_formats = get_video_formats()
        else:
            ffmpeg_formats = []
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats,),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
                "notify_all": ("BOOLEAN", {"default": True}),
                "notify_all_with_meta": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("VHS_AUDIO",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES","STRING")
    RETURN_NAMES = ("Filenames","Video Path")
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        images,
        frame_rate: int,
        loop_count: int,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
        manual_format_widgets=None,
        notify_all=True,
        notify_all_with_meta=True
    ):
        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files = []

        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = prompt
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]

        # comfy counter workaround
        max_counter = 0

        # Loop through the existing files
        matcher = re.compile(f"{re.escape(filename)}_(\d+)\D*\.[a-zA-Z0-9]+")
        for existing_file in os.listdir(full_output_folder):
            # Check if the file matches the expected format
            match = matcher.fullmatch(existing_file)
            if match:
                # Extract the numeric portion of the filename
                file_counter = int(match.group(1))
                # Update the maximum counter value if necessary
                if file_counter > max_counter:
                    max_counter = file_counter

        # Increment the counter by 1 to get the next available value
        counter = max_counter + 1

        # save first frame as png to keep metadata
        file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, file)
        Image.fromarray(tensor_to_bytes(images[0])).save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )
        output_files.append(file_path)

        format_type, format_ext = format.split("/")
        if format_type == "image":
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            images = tensor_to_bytes(images)
            if pingpong:
                images = np.concatenate((images, images[-2:0:-1]))
            frames = [Image.fromarray(f) for f in images]
            # Use pillow directly to save an animated image
            frames[0].save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
            )
            output_files.append(file_path)
        else:
            # Use ffmpeg to save a video
            if ffmpeg_path is None:
                #Should never be reachable
                raise ProcessLookupError("Could not find ffmpeg")

            #Acquire additional format_widget values
            kwargs = None
            if manual_format_widgets is None:
                if prompt is not None:
                    kwargs = prompt[unique_id]['inputs']
                else:
                    manual_format_widgets = {}
            if kwargs is None:
                kwargs = get_format_widget_defaults(format_ext)
                missing = {}
                for k in kwargs.keys():
                    if k in manual_format_widgets:
                        kwargs[k] = manual_format_widgets[k]
                    else:
                        missing[k] = kwargs[k]
                if len(missing) > 0:
                    logger.warn("Extra format values were not provided, the following defaults will be used: " + str(kwargs) + "\nThis is likely due to usage of ComfyUI-to-python. These values can be manually set by supplying a manual_format_widgets argument")

            video_format = apply_format_widgets(format_ext, kwargs)
            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = tensor_to_shorts(images)
                if images.shape[-1] == 4:
                    i_pix_fmt = 'rgba64'
                else:
                    i_pix_fmt = 'rgb48'
            else:
                images = tensor_to_bytes(images)
                if images.shape[-1] == 4:
                    i_pix_fmt = 'rgba'
                else:
                    i_pix_fmt = 'rgb24'
            if pingpong:
                images = np.concatenate((images, images[-2:0:-1]))
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            dimensions = f"{len(images[0][0])}x{len(images[0])}"
            loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(len(images))]
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    "-s", dimensions, "-r", str(frame_rate), "-i", "-"] \
                    + loop_args + video_format['main_pass'] + bitrate_arg

            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])
            res = None
            if video_format.get('save_metadata', 'False') != 'False':
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                metadata = json.dumps(video_metadata)
                metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
                #metadata from file should  escape = ; # \ and newline
                metadata = metadata.replace("\\","\\\\")
                metadata = metadata.replace(";","\\;")
                metadata = metadata.replace("#","\\#")
                metadata = metadata.replace("=","\\=")
                metadata = metadata.replace("\n","\\\n")
                metadata = "comment=" + metadata
                with open(metadata_path, "w") as f:
                    f.write(";FFMETADATA1\n")
                    f.write(metadata)
                m_args = args[:1] + ["-i", metadata_path] + args[1:]
                try:
                    res = subprocess.run(m_args + [file_path], input=images.tobytes(),
                                         capture_output=True, check=True, env=env)
                except subprocess.CalledProcessError as e:
                    #Check if output file exists. If it does, the re-execution
                    #will also fail. This obscures the cause of the error
                    #and seems to never occur concurrent to the metadata issue
                    if os.path.exists(file_path):
                        raise Exception("An error occured in the ffmpeg subprocess:\n" \
                                + e.stderr.decode("utf-8"))
                    #Res was not set
                    print(e.stderr.decode("utf-8"), end="", file=sys.stderr)
                    logger.warn("An error occurred when saving with metadata")

            if not res:
                try:
                    res = subprocess.run(args + [file_path], input=images.tobytes(),
                                         capture_output=True, check=True, env=env)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode("utf-8"))
            if res.stderr:
                print(res.stderr.decode("utf-8"), end="", file=sys.stderr)
            output_files.append(file_path)


            # Audio Injection after video is created, saves additional video with -audio.mp4

            # Create audio file if input was provided
            if audio:
                output_file_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
                output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
                if "audio_pass" not in video_format:
                    logger.warn("Selected video format does not have explicit audio support")
                    video_format["audio_pass"] = ["-c:a", "libopus"]


                # FFmpeg command with audio re-encoding
                #TODO: expose audio quality options if format widgets makes it in
                #Reconsider forcing apad/shortest
                mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,
                            "-i", "-", "-c:v", "copy"] \
                            + video_format["audio_pass"] \
                            + ["-af", "apad", "-shortest", output_file_with_audio_path]

                try:
                    res = subprocess.run(mux_args, input=audio(), env=env,
                                         capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode("utf-8"))
                if res.stderr:
                    print(res.stderr.decode("utf-8"), end="", file=sys.stderr)
                output_files.append(output_file_with_audio_path)
                #Return this file with audio to the webui.
                #It will be muted unless opened or saved with right click
                file = output_file_with_audio
        if notify_all:
            notifyAll(os.path.join(full_output_folder, file),f"{prompt}" if notify_all_with_meta else "===")
        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_output else "temp",
                "format": format,
            }
        ]
        return {"ui": {"gifs": previews}, "result": ((save_output, output_files),os.path.join(full_output_folder, file))}
    @classmethod
    def VALIDATE_INPUTS(self, format, **kwargs):
        return True


# All beelow generated by ChatGPT, not test enough, just for fun
class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        #Hide ffmpeg formats if ffmpeg isn't available
        return {
            "required": {
                "audio_file": ("STRING", {"default": "input/", "vhs_path_extensions": ['wav','mp3','ogg','m4a','flac']}),
                },
            "optional" : {"seek_seconds": ("FLOAT", {"default": 0, "min": 0})}
        }

    RETURN_TYPES = ("VHS_AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "load_audio"
    def load_audio(self, audio_file, seek_seconds):
        if audio_file is None or validate_path(audio_file) != True:
            raise Exception("audio_file is not a valid path: " + audio_file)
        #Eagerly fetch the audio since the user must be using it if the
        #node executes, unlike Load Video
        audio = get_audio(audio_file, start_time=seek_seconds)
        return (lambda : audio,)

    @classmethod
    def IS_CHANGED(s, audio_file, seek_seconds):
        return hash_path(audio_file)

    @classmethod
    def VALIDATE_INPUTS(s, audio_file, **kwargs):
        return validate_path(audio_file, allow_none=True)

import os
import subprocess
import datetime

class MergeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file_1": ("STRING", {"validate": "is_file"}),
                "audio_file_2": ("STRING", {"validate": "is_file"}),
                "output_file_name": ("STRING", {"default": "output.mp3"}),
            },
        }

    RETURN_TYPES = ("VHS_AUDIO", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("audio", "output_file", "audio_1_duration", "audio_2_duration", "merged_audio_duration")
    CATEGORY = "Video Helper Suite 🎥VHS"
    FUNCTION = "merge_audio"

    @classmethod
    def merge_audio(cls, audio_file_1, audio_file_2, output_file_name):
        # Get the absolute path of the output file
        _datetime = datetime.datetime.now().strftime("%Y%m%d")
        _datetime = _datetime + datetime.datetime.now().strftime("%H%M%S%f")

        output_file_abs = os.path.abspath(f"audio_{_datetime}.mp3")

        # Construct FFmpeg command
        ffmpeg_cmd = ['ffmpeg', '-i', audio_file_1, '-i', audio_file_2, '-filter_complex',
                      '[0:a][1:a]concat=n=2:v=0:a=1[a]', '-map', '[a]', output_file_abs]

        # Execute FFmpeg command
        subprocess.run(ffmpeg_cmd)

        # Read the merged audio file
        audio = get_audio(output_file_abs)

        # Get the duration of each audio file and the merged audio file
        audio_1_duration = get_audio_duration(audio_file_1)
        audio_2_duration = get_audio_duration(audio_file_2)
        merged_audio_duration = get_audio_duration(output_file_abs)

        # Return the merged audio, output file absolute path, and durations as a lambda function
        return (lambda: audio, output_file_abs, audio_1_duration, audio_2_duration, merged_audio_duration)

    @classmethod
    def IS_CHANGED(cls, audio_file_1, audio_file_2, output_file_name):
        return hash_path(output_file)

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file_1, audio_file_2, **kwargs):
        if not validate_path(audio_file_1, allow_none=True) or not validate_path(audio_file_2, allow_none=True):
            return False
        return True

def get_audio_duration(audio_file):
    """
    Get the duration of an audio file in seconds.

    Args:
        audio_file: The absolute path to the audio file.

    Returns:
        The duration of the audio file in seconds.
    """
    ffmpeg_cmd = ['ffmpeg', '-i', audio_file]
    output = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration_regex = re.compile(r"Duration: (?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+\.\d+)")
    match = duration_regex.search(output.stdout.decode('utf-8'))
    if match:
        hours = float(match.group('hours'))
        minutes = float(match.group('minutes'))
        seconds = float(match.group('seconds'))
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("Could not parse audio file duration.")



import os,io
from shortGPT.audio import audio_utils
from shortGPT.audio.audio_duration import get_asset_duration
from shortGPT.config.asset_db import AssetDatabase
from shortGPT.config.languages import Language
from shortGPT.editing_framework.editing_engine import EditingEngine, EditingStep
from shortGPT.editing_utils import captions

class VideoCaptions:
    @classmethod
    def INPUT_TYPES(cls):
        default_template_para = """
"fontsize": 100,
"font": "Roboto-Bold",
"color": "white",
"stroke_width": 4,
"stroke_color": "black",
"method": "caption"   
"""

        return {
            "required": {
                "video_path": ("STRING",{"default": ""}),
                "output_filename": ("STRING",{"default": "_captioned"}),
                "is_vertical": ("BOOLEAN",{"default": True}),
                "add_subscription_anim": ("BOOLEAN",{"default": False}),
                "notify_all": ("BOOLEAN",{"default":True})
            },
            "optional": {
                "audio_path": ("STRING",{"default":""}),
                "water_mark": ("STRING",{"default":"OnePieOne"}),
                "caption_json_param": ("STRING",{"default":f"{default_template_para}","multiline":True})    
            },
            "hidden": {},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Video Path",)
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "add_captions"

    def add_captions(self, video_path, output_filename, audio_path,is_vertical,add_subscription_anim,water_mark,notify_all,caption_json_param):
        m_is_vertical = is_vertical  # Set this based on your requirements
        # TODO - select by user from node
        language = Language.ENGLISH  # Set this based on your requirements

        if audio_path is None or audio_path == "" :
            m_audio_path = video_path
        else:
            m_audio_path = audio_path

        _datetime = datetime.datetime.now().strftime("%Y%m%d")
        _datetime = _datetime + datetime.datetime.now().strftime("%H%M%S%f")

        output_path = os.path.join(folder_paths.get_output_directory(), output_filename+f"_{_datetime}.mp4")

        if not os.path.exists(output_path):
            print("Rendering short: Starting automated editing...")

            print(f"Caption Param --- {caption_json_param}")

            timed_captions = self._time_captions(m_audio_path, m_is_vertical)

            video_editor = EditingEngine()
            video_editor.addEditingStep(EditingStep.ADD_VOICEOVER_AUDIO, {"url": m_audio_path})

            if add_subscription_anim:
                video_editor.addEditingStep(
                    EditingStep.ADD_SUBSCRIBE_ANIMATION, {"url": "public/subscribe-animation.mp4"}
                )

            _, vid_length = get_asset_duration(video_path)
            video_editor.addEditingStep(
                EditingStep.ADD_BACKGROUND_VIDEO,
                {"url": video_path, "set_time_start": 0, "set_time_end": vid_length},
            )

            video_editor.addEditingStep(EditingStep.CROP_1920x1080, {"url": video_path})

            video_editor.addEditingStep(EditingStep.ADD_WATERMARK,{"text":water_mark})

            if m_is_vertical:
                caption_type = (
                    EditingStep.ADD_CAPTION_SHORT_ARABIC
                    if language == Language.ARABIC
                    else EditingStep.ADD_CAPTION_SHORT
                )
            else:
                caption_type = (
                    EditingStep.ADD_CAPTION_LANDSCAPE_ARABIC
                    if language == Language.ARABIC
                    else EditingStep.ADD_CAPTION_LANDSCAPE
                )

            for (t1, t2), text in timed_captions:
                # string_io = io.StringIO(f"{{caption_json_param}}")

                #TODO - temp for test, then optimize
                formated = '{' + caption_json_param.strip() + '}'
                
                template_dict = json.loads(formated,strict=False)

                # Add the text, start time, and end time to the dictionary
                template_dict["text"] = text.upper()
                template_dict["set_time_start"] = t1
                template_dict["set_time_end"] = t2
                video_editor.addEditingStep(
                    caption_type,
                    template_dict,
                )

            video_editor.renderVideo(output_path, None)

        if notify_all:
            notifyAll(output_path, "====Caption====")
        return {"ui": {"video": [{"filename": output_filename, "subfolder": "", "type": "output"}]}, "result": (output_path,)}

    def _time_captions(self, audio_path, is_vertical=True):
        whisper_analysis = audio_utils.audioToText(audio_path)
        max_len = 15

        if is_vertical:
            max_len = 30

        result = captions.getCaptionsWithTime(whisper_analysis, maxCaptionSize=max_len)

        print(result)
        return result



from .caption import GentleCaption
import ast
class VideoGentleCaptions:
    def __init__(self) -> None:
        self.cp = GentleCaption()
        pass
    
    
    @classmethod
    def INPUT_TYPES(cls):
        default_template_para = """
'Fontname': 'Lemon-Regular',
'Alignment': 5,
'BorderStyle': '1',
'Outline': '1',
'Shadow': '2',
'Blur': '21',
'Fontsize': 22,
'MarginL': '0',
'MarginR': '0',
'tag': -1,
'highlight_color': 'white',
'karaoke': False,
'vad': True,
'word_level': True,
'segment_level': True
"""

        return {
            "required": {
                "video_path": ("STRING",{"default": ""}),
                "output_filename": ("STRING",{"default": "_captioned"}),
                "is_vertical": ("BOOLEAN",{"default": True}),
                "notify_all": ("BOOLEAN",{"default":True})
            },
            "optional": {
                "audio_path": ("STRING",{"default":""}),
                "caption_json_param": ("STRING",{"default":f"{default_template_para}","multiline":True})    
            },
            "hidden": {},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Video Path",)
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "add_gentle_captions"

    def add_gentle_captions(self, video_path, output_filename, audio_path,is_vertical,notify_all,caption_json_param):
        
        
        # 删除换行符和缩进 TODO - 手动拼接 ，不知道如何转dict，临时使用
        para = caption_json_param.replace('\n', '')
        para = para.replace(' ', '')
        para = "{"+para+"}"
        video_result =  self.cp.makeVideo(bg_video_path=video_path,bg_audio_path=audio_path,output_filename=output_filename,extra_para=dict(eval(para)))

        if notify_all:
            notifyAll(video_result, "====Caption====")
        return {"ui": {"video": [{"filename": output_filename, "subfolder": "", "type": "output"}]}, "result": (video_result,)}


def string_to_dict(string):
  """
  将一个字符串转换为字典

  Args:
    string: 要转换的字符串

  Returns:
    一个字典
  """

  # 将字符串拆分为键值对列表
  key_value_pairs = string.split(',')

  # 创建一个字典
  d = {}

  # 遍历键值对列表
  for key_value_pair in key_value_pairs:
    # 将键值对拆分为键和值
    key, value = key_value_pair.split(':')

    # 将键和值添加到字典中
    d[key] = value

  # 返回字典
  return d






NODE_CLASS_MAPPINGS = {
    "VHS_VideoCombine": VideoCombine,
    "VHS_LoadVideo": LoadVideoUpload,
    "VHS_LoadVideoPath": LoadVideoPath,
    "VHS_LoadImages": LoadImagesFromDirectoryUpload,
    "VHS_LoadImagesPath": LoadImagesFromDirectoryPath,
    "VHS_LoadAudio": LoadAudio,
    "VHS_MergeAudio": MergeAudio,
    "VHS_VideoCaptions": VideoCaptions,
    "VHS_VideoGentleCaptions": VideoGentleCaptions,
    # Latent and Image nodes
    "VHS_SplitLatents": SplitLatents,
    "VHS_SplitImages": SplitImages,
    "VHS_SplitMasks": SplitMasks,
    "VHS_MergeLatents": MergeLatents,
    "VHS_MergeImages": MergeImages,
    "VHS_MergeMasks": MergeMasks,
    "VHS_SelectEveryNthLatent": SelectEveryNthLatent,
    "VHS_SelectEveryNthImage": SelectEveryNthImage,
    "VHS_SelectEveryNthMask": SelectEveryNthMask,
    "VHS_GetLatentCount": GetLatentCount,
    "VHS_GetImageCount": GetImageCount,
    "VHS_GetMaskCount": GetMaskCount,
    "VHS_DuplicateLatents": DuplicateLatents,
    "VHS_DuplicateImages": DuplicateImages,
    "VHS_DuplicateMasks": DuplicateMasks,
    # Batched Nodes
    "VHS_VAEEncodeBatched": VAEEncodeBatched,
    "VHS_VAEDecodeBatched": VAEDecodeBatched,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_VideoCombine": "Video Combine 🎥🅥🅗🅢",
    "VHS_LoadVideo": "Load Video (Upload) 🎥🅥🅗🅢",
    "VHS_LoadVideoPath": "Load Video (Path) 🎥🅥🅗🅢",
    "VHS_LoadImages": "Load Images (Upload) 🎥🅥🅗🅢",
    "VHS_LoadImagesPath": "Load Images (Path) 🎥🅥🅗🅢",
    "VHS_LoadAudio": "Load Audio (Path)🎥🅥🅗🅢",
    "VHS_MergeAudio": "Merge Audio 🎥🅥🅗🅢",
    "VHS_VideoCaptions": "Video Captions 🎥🅥🅗🅢",
    "VHS_VideoGentleCaptions": "Video Gentle Captions 🎥🅥🅗🅢",
    # Latent and Image nodes
    "VHS_SplitLatents": "Split Latent Batch 🎥🅥🅗🅢",
    "VHS_SplitImages": "Split Image Batch 🎥🅥🅗🅢",
    "VHS_SplitMasks": "Split Mask Batch 🎥🅥🅗🅢",
    "VHS_MergeLatents": "Merge Latent Batches 🎥🅥🅗🅢",
    "VHS_MergeImages": "Merge Image Batches 🎥🅥🅗🅢",
    "VHS_MergeMasks": "Merge Mask Batches 🎥🅥🅗🅢",
    "VHS_SelectEveryNthLatent": "Select Every Nth Latent 🎥🅥🅗🅢",
    "VHS_SelectEveryNthImage": "Select Every Nth Image 🎥🅥🅗🅢",
    "VHS_SelectEveryNthMask": "Select Every Nth Mask 🎥🅥🅗🅢",
    "VHS_GetLatentCount": "Get Latent Count 🎥🅥🅗🅢",
    "VHS_GetImageCount": "Get Image Count 🎥🅥🅗🅢",
    "VHS_GetMaskCount": "Get Mask Count 🎥🅥🅗🅢",
    "VHS_DuplicateLatents": "Duplicate Latent Batch 🎥🅥🅗🅢",
    "VHS_DuplicateImages": "Duplicate Image Batch 🎥🅥🅗🅢",
    "VHS_DuplicateMasks": "Duplicate Mask Batch 🎥🅥🅗🅢",
    # Batched Nodes
    "VHS_VAEEncodeBatched": "VAE Encode Batched 🎥🅥🅗🅢",
    "VHS_VAEDecodeBatched": "VAE Decode Batched 🎥🅥🅗🅢",
}
