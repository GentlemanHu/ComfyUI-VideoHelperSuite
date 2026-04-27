try:
    from notifier.notify import notifyAll
except Exception:
    def notifyAll(*args, **kwargs):
        return None
import os
import sys
import json
import subprocess
import numpy as np
import re
import datetime
import tempfile
from typing import List
import torch
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
from pathlib import Path
from string import Template
import itertools
import functools

import folder_paths
from .logger import logger
from .image_latent_nodes import *
from .load_video_nodes import LoadVideoUpload, LoadVideoPath, LoadVideoFFmpegUpload, LoadVideoFFmpegPath, LoadImagePath
from .load_images_nodes import LoadImagesFromDirectoryUpload, LoadImagesFromDirectoryPath
from .batched_nodes import VAEEncodeBatched, VAEDecodeBatched
from .utils import ffmpeg_path, get_audio, hash_path, validate_path, requeue_workflow, \
        gifski_path, calculate_file_hash, strip_path, try_download_video, is_url, \
        imageOrLatent, BIGMAX, merge_filter_args, ENCODE_ARGS, floatOrInt, cached, \
        ContainsAll
from comfy.utils import ProgressBar

# Import custom additions (按能力分组，避免单点导入失败导致全部不可用)
HAS_VIDEO_OPS = False
HAS_DEPTH_FEATURES = False
HAS_GENTLE_CAPTION = False
HAS_SHORTGPT = False

try:
    from .video_ops import (
        CompositeMedia,
        CompositeMultiVideo,
        MovisTimelinePro,
        MovisCreateTimeline,
        MovisAddVideoTrack,
        MovisAddImageTrack,
        MovisMergeTimeline,
        MovisAddImageSequenceTrack,
        MovisAddImageMotionTrack,
        MovisAddVideoMotionTrack,
        MovisAddAudioTrack,
        MovisSetBGM,
        MovisAddTextOverlay,
        MovisRenderTimeline,
        MovisAssemble,
        MovisUniversalStudio,
        MovisSmartMerge,
        MovisSetGlobalTransition,
        MovisSetClipTransition,
        MovisAddClipFX,
        MovisApplyFXPreset,
        MovisEnableLayeredFXEngine,
        MovisAddClipFXLayered,
        MovisApplyFXPresetLayered,
        MovisGPUShaderRender,
        MovisSetClipShader,
        MovisTrimClip,
        MovisDeleteClip,
        MovisBuildFX,
        MovisBuildFXPreset,
        MovisChainFX,
        MovisBuildShader,
        MovisBuildAudio,
        MovisChainAudio,
        MovisBatchAddVideoTracks,
        MovisQuickBuildTimeline,
    )
    HAS_VIDEO_OPS = True
except ImportError:
    print("video_ops not available")

try:
    from .depth_generator import *
    HAS_DEPTH_FEATURES = True
except ImportError:
    print("depth_generator not available")

try:
    from .caption import GentleCaption, resolve_media_path
    HAS_GENTLE_CAPTION = True
except ImportError:
    print("caption module not available")

try:
    from shortGPT.audio import audio_utils
    from shortGPT.audio.audio_duration import get_asset_duration
    from shortGPT.config.languages import Language
    from shortGPT.editing_framework.editing_engine import EditingEngine, EditingStep
    from shortGPT.editing_utils import captions
    HAS_SHORTGPT = True
except ImportError:
    print("shortGPT not available")

HAS_CUSTOM_FEATURES = HAS_VIDEO_OPS or HAS_DEPTH_FEATURES or HAS_GENTLE_CAPTION or HAS_SHORTGPT

if 'VHS_video_formats' not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["VHS_video_formats"] = ((),{".json"})
if len(folder_paths.folder_names_and_paths['VHS_video_formats'][1]) == 0:
    folder_paths.folder_names_and_paths["VHS_video_formats"][1].add(".json")
audio_extensions = ['mp3', 'mp4', 'wav', 'ogg']

def flatten_list(l):
    ret = []
    for e in l:
        if isinstance(e, list):
            ret.extend(e)
        else:
            ret.append(e)
    return ret

def iterate_format(video_format, for_widgets=True):
    """Provides an iterator over widgets, or arguments"""
    def indirector(cont, index):
        if isinstance(cont[index], list) and (not for_widgets
          or len(cont[index])> 1 and not isinstance(cont[index][1], dict)):
            inp = yield cont[index]
            if inp is not None:
                cont[index] = inp
                yield
    for k in video_format:
        if k == "extra_widgets":
            if for_widgets:
                yield from video_format["extra_widgets"]
        elif k.endswith("_pass"):
            for i in range(len(video_format[k])):
                yield from indirector(video_format[k], i)
            if not for_widgets:
                video_format[k] = flatten_list(video_format[k])
        else:
            yield from indirector(video_format, k)

base_formats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats")
@cached(5)
def get_video_formats():
    format_files = {}
    for format_name in folder_paths.get_filename_list("VHS_video_formats"):
        format_files[format_name] = folder_paths.get_full_path("VHS_video_formats", format_name)
    for item in os.scandir(base_formats_dir):
        if not item.is_file() or not item.name.endswith('.json'):
            continue
        format_files[item.name[:-5]] = item.path
    formats = []
    format_widgets = {}
    for format_name, path in format_files.items():
        with open(path, 'r') as stream:
            video_format = json.load(stream)
        if "gifski_pass" in video_format and gifski_path is None:
            #Skip format
            continue
        widgets = list(iterate_format(video_format))
        formats.append("video/" + format_name)
        if (len(widgets) > 0):
            format_widgets["video/"+ format_name] = widgets
    return formats, format_widgets

def apply_format_widgets(format_name, kwargs):
    if os.path.exists(os.path.join(base_formats_dir, format_name + ".json")):
        video_format_path = os.path.join(base_formats_dir, format_name + ".json")
    else:
        video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name)
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    for w in iterate_format(video_format):
        if w[0] not in kwargs:
            if len(w) > 2 and 'default' in w[2]:
                default = w[2]['default']
            else:
                if type(w[1]) is list:
                    default = w[1][0]
                else:
                    #NOTE: This doesn't respect max/min, but should be good enough as a fallback to a fallback to a fallback
                    default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[1]]
            kwargs[w[0]] = default
            logger.warn(f"Missing input for {w[0]} has been set to {default}")
    wit = iterate_format(video_format, False)
    for w in wit:
        while isinstance(w, list):
            if len(w) == 1:
                #TODO: mapping=kwargs should be safer, but results in key errors, investigate why
                w = [Template(x).substitute(**kwargs) for x in w[0]]
                break
            elif isinstance(w[1], dict):
                w = w[1][str(kwargs[w[0]])]
            elif len(w) > 3:
                w = Template(w[3]).substitute(val=kwargs[w[0]])
            else:
                w = str(kwargs[w[0]])
        wit.send(w)
    return video_format

def tensor_to_int(tensor, bits):
    tensor = tensor.cpu().numpy() * (2**bits-1) + 0.5
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)

def ffmpeg_process(args, video_format, video_metadata, file_path, env):

    res = None
    frame_data = yield
    total_frames_output = 0
    if video_format.get('save_metadata', 'False') != 'False':
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
        #metadata from file should  escape = ; # \ and newline
        def escape_ffmpeg_metadata(key, value):
            value = str(value)
            value = value.replace("\\","\\\\")
            value = value.replace(";","\\;")
            value = value.replace("#","\\#")
            value = value.replace("=","\\=")
            value = value.replace("\n","\\\n")
            return f"{key}={value}"

        with open(metadata_path, "w") as f:
            f.write(";FFMETADATA1\n")
            if "prompt" in video_metadata:
                f.write(escape_ffmpeg_metadata("prompt", json.dumps(video_metadata["prompt"])) + "\n")
            if "workflow" in video_metadata:
                f.write(escape_ffmpeg_metadata("workflow", json.dumps(video_metadata["workflow"])) + "\n")
            for k, v in video_metadata.items():
                if k not in ["prompt", "workflow"]:
                    f.write(escape_ffmpeg_metadata(k, json.dumps(v)) + "\n")

        m_args = args[:1] + ["-i", metadata_path] + args[1:] + ["-metadata", "creation_time=now", "-movflags", "use_metadata_tags"]
        with subprocess.Popen(m_args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    #TODO: skip flush for increased speed
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                err = proc.stderr.read()
                #Check if output file exists. If it does, the re-execution
                #will also fail. This obscures the cause of the error
                #and seems to never occur concurrent to the metadata issue
                if os.path.exists(file_path):
                    raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                            + err.decode(*ENCODE_ARGS))
                #Res was not set
                print(err.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                logger.warn("An error occurred when saving with metadata")
    if res != b'':
        with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                res = proc.stderr.read()
                raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                        + res.decode(*ENCODE_ARGS))
    yield total_frames_output
    if len(res) > 0:
        print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)

def gifski_process(args, dimensions, frame_rate, video_format, file_path, env):
    frame_data = yield
    with subprocess.Popen(args + video_format['main_pass'] + ['-f', 'yuv4mpegpipe', '-'],
                          stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, env=env) as procff:
        with subprocess.Popen([gifski_path] + video_format['gifski_pass']
                              + ['-W', f'{dimensions[0]}', '-H', f'{dimensions[1]}']
                              + ['-r', f'{frame_rate}']
                              + ['-q', '-o', file_path, '-'], stderr=subprocess.PIPE,
                              stdin=procff.stdout, stdout=subprocess.PIPE,
                              env=env) as procgs:
            try:
                while frame_data is not None:
                    procff.stdin.write(frame_data)
                    frame_data = yield
                procff.stdin.flush()
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                outgs = procgs.stdout.read()
            except BrokenPipeError as e:
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                raise Exception("An error occurred while creating gifski output\n" \
                        + "Make sure you are using gifski --version >=1.32.0\nffmpeg: " \
                        + resff.decode(*ENCODE_ARGS) + '\ngifski: ' + resgs.decode(*ENCODE_ARGS))
    if len(resff) > 0:
        print(resff.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    if len(resgs) > 0:
        print(resgs.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    #should always be empty as the quiet flag is passed
    if len(outgs) > 0:
        print(outgs.decode(*ENCODE_ARGS))

def to_pingpong(inp):
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp)-2,0,-1):
        yield inp[i]

class VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [['lossless', "BOOLEAN", {'default': True}]]
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (
                    floatOrInt,
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats, {'formats': format_widgets}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "notify_all": ("BOOLEAN", {"default": False}),
                "notify_all_with_meta": ("BOOLEAN", {"default": False}),
            },
            "hidden": ContainsAll({
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }),
        }

    RETURN_TYPES = ("VHS_FILENAMES", "STRING")
    RETURN_NAMES = ("Filenames", "Video Path")
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        frame_rate: int,
        loop_count: int,
        images=None,
        latents=None,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None,
        vae=None,
        notify_all=False,
        notify_all_with_meta=False,
        **kwargs
    ):
        if latents is not None:
            images = latents
        if images is None:
            return ((save_output, []), "")
        if vae is not None:
            if isinstance(images, dict):
                images = images['samples']
            else:
                vae = None

        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ((save_output, []), "")
        num_frames = len(images)
        pbar = ProgressBar(num_frames)
        if vae is not None:
            downscale_ratio = getattr(vae, "downscale_ratio", 8)
            width = images.size(-1)*downscale_ratio
            height = images.size(-2)*downscale_ratio
            frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
            #Python 3.12 adds an itertools.batched, but it's easily replicated for legacy support
            def batched(it, n):
                while batch := tuple(itertools.islice(it, n)):
                    yield batch
            def batched_encode(images, vae, frames_per_batch):
                for batch in batched(iter(images), frames_per_batch):
                    image_batch = torch.from_numpy(np.array(batch))
                    yield from vae.decode(image_batch)
            images = batched_encode(images, vae, frames_per_batch)
            first_image = next(images)
            #repush first_image
            images = itertools.chain([first_image], images)
            #A single image has 3 dimensions. Discard higher dimensions
            while len(first_image.shape) > 3:
                first_image = first_image[0]
        else:
            first_image = images[0]
            images = iter(images)
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
            video_metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
            extra_options = extra_pnginfo.get('workflow', {}).get('extra', {})
        else:
            extra_options = {}
        # metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        if meta_batch is not None and unique_id in meta_batch.outputs:
            (counter, output_process) = meta_batch.outputs[unique_id]
        else:
            # comfy counter workaround
            max_counter = 0

            # Loop through the existing files
            matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
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
            output_process = None

        # save first frame as png to keep metadata
        first_image_file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, first_image_file)
        if extra_options.get('VHS_MetadataImage', True) != False:
            Image.fromarray(tensor_to_bytes(first_image)).save(
                file_path,
                pnginfo=metadata,
                compress_level=4,
            )
        output_files.append(file_path)

        format_type, format_ext = format.split("/")
        if format_type == "image":
            if meta_batch is not None:
                raise Exception("Pillow('image/') formats are not compatible with batched output")
            image_kwargs = {}
            if format_ext == "gif":
                image_kwargs['disposal'] = 2
            if format_ext == "webp":
                #Save timestamp information
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs['exif'] = exif
                image_kwargs['lossless'] = kwargs.get("lossless", True)
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            if pingpong:
                images = to_pingpong(images)
            def frames_gen(images):
                for i in images:
                    pbar.update(1)
                    yield Image.fromarray(tensor_to_bytes(i))
            frames = frames_gen(images)
            # Use pillow directly to save an animated image
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs
            )
            output_files.append(file_path)
        else:
            # Use ffmpeg to save a video
            if ffmpeg_path is None:
                raise ProcessLookupError(f"ffmpeg is required for video outputs and could not be found.\nIn order to use video outputs, you must either:\n- Install imageio-ffmpeg with pip,\n- Place a ffmpeg executable in {os.path.abspath('')}, or\n- Install ffmpeg and add it to the system path.")

            if manual_format_widgets is not None:
                logger.warn("Format args can now be passed directly. The manual_format_widgets argument is now deprecated")
                kwargs.update(manual_format_widgets)

            has_alpha = first_image.shape[-1] == 4
            kwargs["has_alpha"] = has_alpha
            video_format = apply_format_widgets(format_ext, kwargs)
            dim_alignment = video_format.get("dim_alignment", 2)
            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                #output frames must be padded
                to_pad = (-first_image.shape[1] % dim_alignment,
                          -first_image.shape[0] % dim_alignment)
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                           to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)
                def pad(image):
                    image = image.permute((2,0,1))#HWC to CHW
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1,2,0))
                images = map(pad, images)
                dimensions = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                              -first_image.shape[0] % dim_alignment + first_image.shape[0])
                logger.warn("Output images were not of valid resolution and have had padding applied")
            else:
                dimensions = (first_image.shape[1], first_image.shape[0])
            if pingpong:
                if meta_batch is not None:
                    logger.error("pingpong is incompatible with batched output")
                images = to_pingpong(images)
                if num_frames > 2:
                    num_frames += num_frames -2
                    pbar.total = num_frames
            if loop_count > 0:
                loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(num_frames)]
            else:
                loop_args = []
            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = map(tensor_to_shorts, images)
                if has_alpha:
                    i_pix_fmt = 'rgba64'
                else:
                    i_pix_fmt = 'rgb48'
            else:
                images = map(tensor_to_bytes, images)
                if has_alpha:
                    i_pix_fmt = 'rgba'
                else:
                    i_pix_fmt = 'rgb24'
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    # The image data is in an undefined generic RGB color space, which in practice means sRGB.
                    # sRGB has the same primaries and matrix as BT.709, but a different transfer function (gamma),
                    # called by the sRGB standard name IEC 61966-2-1. However, video hosting platforms like YouTube
                    # standardize on full BT.709 and will convert the colors accordingly. This last minute change
                    # in colors can be confusing to users. We can counter it by lying about the transfer function
                    # on a per format basis, i.e. for video we will lie to FFmpeg that it is already BT.709. Also,
                    # because the input data is in RGB (not YUV) it is more efficient (fewer scale filter invocations)
                    # to specify the input color space as RGB and then later, if the format actually wants YUV,
                    # to convert it to BT.709 YUV via FFmpeg's -vf "scale=out_color_matrix=bt709".
                    "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
                    "-color_trc", video_format.get("fake_trc", "iec61966-2-1"),
                    "-s", f"{dimensions[0]}x{dimensions[1]}", "-r", str(frame_rate), "-i", "-"] \
                    + loop_args

            images = map(lambda x: x.tobytes(), images)
            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])

            if "pre_pass" in video_format:
                if meta_batch is not None:
                    #Performing a prepass requires keeping access to all frames.
                    #Potential solutions include keeping just output frames in
                    #memory or using 3 passes with intermediate file, but
                    #very long gifs probably shouldn't be encouraged
                    raise Exception("Formats which require a pre_pass are incompatible with Batch Manager.")
                images = [b''.join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                in_args_len = args.index("-i") + 2 # The index after ["-i", "-"]
                pre_pass_args = args[:in_args_len] + video_format['pre_pass']
                merge_filter_args(pre_pass_args)
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env,
                                   capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n" \
                            + e.stderr.decode(*ENCODE_ARGS))
            if "inputs_main_pass" in video_format:
                in_args_len = args.index("-i") + 2 # The index after ["-i", "-"]
                args = args[:in_args_len] + video_format['inputs_main_pass'] + args[in_args_len:]

            if output_process is None:
                if 'gifski_pass' in video_format:
                    format = 'image/gif'
                    output_process = gifski_process(args, dimensions, frame_rate, video_format, file_path, env)
                    audio = None
                else:
                    args += video_format['main_pass'] + bitrate_arg
                    merge_filter_args(args)
                    output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
                #Proceed to first yield
                output_process.send(None)
                if meta_batch is not None:
                    meta_batch.outputs[unique_id] = (counter, output_process)

            for image in images:
                pbar.update(1)
                output_process.send(image)
            if meta_batch is not None:
                requeue_workflow((meta_batch.unique_id, not meta_batch.has_closed_inputs))
            if meta_batch is None or meta_batch.has_closed_inputs:
                #Close pipe and wait for termination.
                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    pass
                if meta_batch is not None:
                    meta_batch.outputs.pop(unique_id)
                    if len(meta_batch.outputs) == 0:
                        meta_batch.reset()
            else:
                #batch is unfinished
                #TODO: Check if empty output breaks other custom nodes
                return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []), "")}

            output_files.append(file_path)


            a_waveform = None
            if audio is not None:
                try:
                    #safely check if audio produced by VHS_LoadVideo actually exists
                    a_waveform = audio['waveform']
                except:
                    pass
            if a_waveform is not None:
                # Create audio file if input was provided
                output_file_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
                output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
                if "audio_pass" not in video_format:
                    logger.warn("Selected video format does not have explicit audio support")
                    video_format["audio_pass"] = ["-c:a", "libopus"]


                # FFmpeg command with audio re-encoding
                #TODO: expose audio quality options if format widgets makes it in
                #Reconsider forcing apad/shortest
                channels = audio['waveform'].size(1)
                min_audio_dur = total_frames_output / frame_rate + 1
                if video_format.get('trim_to_audio', 'False') != 'False':
                    apad = []
                else:
                    apad = ["-af", "apad=whole_dur="+str(min_audio_dur)]
                mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,
                            "-ar", str(audio['sample_rate']), "-ac", str(channels),
                            "-f", "f32le", "-i", "-", "-c:v", "copy"] \
                            + video_format["audio_pass"] \
                            + apad + ["-shortest", output_file_with_audio_path]

                audio_data = audio['waveform'].squeeze(0).transpose(0,1) \
                        .numpy().tobytes()
                merge_filter_args(mux_args, '-af')
                try:
                    res = subprocess.run(mux_args, input=audio_data,
                                         env=env, capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode(*ENCODE_ARGS))
                if res.stderr:
                    print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                output_files.append(output_file_with_audio_path)
                #Return this file with audio to the webui.
                #It will be muted unless opened or saved with right click
                file = output_file_with_audio
        
        # Notify if enabled
        if notify_all and HAS_CUSTOM_FEATURES:
            try:
                notifyAll(os.path.join(full_output_folder, file), 
                         f"{prompt}" if notify_all_with_meta else "===")
            except Exception as e:
                logger.warn(f"Notification failed: {e}")
        
        if extra_options.get('VHS_KeepIntermediate', True) == False:
            for intermediate in output_files[1:-1]:
                if os.path.exists(intermediate):
                    os.remove(intermediate)
        preview = {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_output else "temp",
                "format": format,
                "frame_rate": frame_rate,
                "workflow": first_image_file,
                "fullpath": output_files[-1],
            }
        if num_frames == 1 and 'png' in format and '%03d' in file:
            preview['format'] = 'image/png'
            preview['filename'] = file.replace('%03d', '001')
        return {"ui": {"gifs": [preview]}, "result": ((save_output, output_files), os.path.join(full_output_folder, file))}

class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        #Hide ffmpeg formats if ffmpeg isn't available
        return {
            "required": {
                "audio_file": ("STRING", {"default": "input/", "vhs_path_extensions": ['wav','mp3','ogg','m4a','flac']}),
                },
            "optional" : {
                "seek_seconds": ("FLOAT", {"default": 0, "min": 0, "widgetType": "VHSTIMESTAMP"}),
                "duration": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01, "widgetType": "VHSTIMESTAMP"}),
                          }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "duration")
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/audio"
    FUNCTION = "load_audio"
    def load_audio(self, audio_file, seek_seconds=0, duration=0):
        audio_file = strip_path(audio_file)
        if audio_file is None or validate_path(audio_file) != True:
            raise Exception("audio_file is not a valid path: " + audio_file)
        if is_url(audio_file):
            audio_file = try_download_video(audio_file) or audio_file
        #Eagerly fetch the audio since the user must be using it if the
        #node executes, unlike Load Video
        audio = get_audio(audio_file, start_time=seek_seconds, duration=duration)
        loaded_duration = audio['waveform'].size(2)/audio['sample_rate']
        return (audio, loaded_duration)

    @classmethod
    def IS_CHANGED(s, audio_file, **kwargs):
        return hash_path(audio_file)

    @classmethod
    def VALIDATE_INPUTS(s, audio_file, **kwargs):
        return validate_path(audio_file, allow_none=True)

class LoadAudioUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        return {"required": {
                    "audio": (sorted(files),),},
                "optional": {
                    "start_time": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01, "widgetType": "VHSTIMESTAMP"}),
                    "duration": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01, "widgetType": "VHSTIMESTAMP"}),
                     },
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/audio"

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "duration")
    FUNCTION = "load_audio"

    def load_audio(self, start_time=0, duration=0, **kwargs):
        audio_file = folder_paths.get_annotated_filepath(strip_path(kwargs['audio']))
        if audio_file is None or validate_path(audio_file) != True:
            raise Exception("audio_file is not a valid path: " + audio_file)

        audio = get_audio(audio_file, start_time, duration)
        loaded_duration = audio['waveform'].size(2)/audio['sample_rate']
        return (audio, loaded_duration)

    @classmethod
    def IS_CHANGED(s, audio, **kwargs):
        audio_file = folder_paths.get_annotated_filepath(strip_path(audio))
        return hash_path(audio_file)

    @classmethod
    def VALIDATE_INPUTS(s, audio, **kwargs):
        audio_file = folder_paths.get_annotated_filepath(strip_path(audio))
        return validate_path(audio_file, allow_none=True)

class AudioToVHSAudio:
    """Legacy method for external nodes that utilized VHS_AUDIO,
    VHS_AUDIO is deprecated as a format and should no longer be used"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO",)}}
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/audio"

    RETURN_TYPES = ("VHS_AUDIO", )
    RETURN_NAMES = ("vhs_audio",)
    FUNCTION = "convert_audio"

    def convert_audio(self, audio):
        ar = str(audio['sample_rate'])
        ac = str(audio['waveform'].size(1))
        mux_args = [ffmpeg_path, "-f", "f32le", "-ar", ar, "-ac", ac,
                    "-i", "-", "-f", "wav", "-"]

        audio_data = audio['waveform'].squeeze(0).transpose(0,1) \
                .numpy().tobytes()
        try:
            res = subprocess.run(mux_args, input=audio_data,
                                 capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception("An error occured in the ffmpeg subprocess:\n" \
                    + e.stderr.decode(*ENCODE_ARGS))
        if res.stderr:
            print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
        return (lambda: res.stdout,)

class VHSAudioToAudio:
    """Legacy method for external nodes that utilized VHS_AUDIO,
    VHS_AUDIO is deprecated as a format and should no longer be used"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vhs_audio": ("VHS_AUDIO",)}}
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/audio"

    RETURN_TYPES = ("AUDIO", )
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert_audio"

    def convert_audio(self, vhs_audio):
        if not vhs_audio or not vhs_audio():
            raise Exception("audio input is not valid")
        args = [ffmpeg_path, "-i", '-']
        try:
            res =  subprocess.run(args + ["-f", "f32le", "-"], input=vhs_audio(),
                                  capture_output=True, check=True)
            audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        except subprocess.CalledProcessError as e:
            raise Exception("An error occured in the ffmpeg subprocess:\n" \
                    + e.stderr.decode(*ENCODE_ARGS))
        match = re.search(', (\\d+) Hz, (\\w+), ',res.stderr.decode(*ENCODE_ARGS))
        if match:
            ar = int(match.group(1))
            #NOTE: Just throwing an error for other channel types right now
            #Will deal with issues if they come
            ac = {"mono": 1, "stereo": 2}[match.group(2)]
        else:
            ar = 44100
            ac = 2
        audio = audio.reshape((-1,ac)).transpose(0,1).unsqueeze(0)
        return ({'waveform': audio, 'sample_rate': ar},)

class PruneOutputs:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "filenames": ("VHS_FILENAMES",),
                    "options": (["Intermediate", "Intermediate and Utility"],)
                    }
                }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "prune_outputs"

    def prune_outputs(self, filenames, options):
        if len(filenames[1]) == 0:
            return ()
        assert(len(filenames[1]) <= 3 and len(filenames[1]) >= 2)
        delete_list = []
        if options in ["Intermediate", "Intermediate and Utility", "All"]:
            delete_list += filenames[1][1:-1]
        if options in ["Intermediate and Utility", "All"]:
            delete_list.append(filenames[1][0])
        if options in ["All"]:
            delete_list.append(filenames[1][-1])

        output_dirs = [folder_paths.get_output_directory(),
                       folder_paths.get_temp_directory()]
        for file in delete_list:
            #Check that path is actually an output directory
            if (os.path.commonpath([output_dirs[0], file]) != output_dirs[0]) \
                    and (os.path.commonpath([output_dirs[1], file]) != output_dirs[1]):
                        raise Exception("Tried to prune output from invalid directory: " + file)
            if os.path.exists(file):
                os.remove(file)
        return ()

class BatchManager:
    def __init__(self, frames_per_batch=-1):
        self.frames_per_batch = frames_per_batch
        self.inputs = {}
        self.outputs = {}
        self.unique_id = None
        self.has_closed_inputs = False
        self.total_frames = float('inf')
    def reset(self):
        self.close_inputs()
        for key in self.outputs:
            if getattr(self.outputs[key][-1], "gi_suspended", False):
                try:
                    self.outputs[key][-1].send(None)
                except StopIteration:
                    pass
        self.__init__(self.frames_per_batch)
    def has_open_inputs(self):
        return len(self.inputs) > 0
    def close_inputs(self):
        for key in self.inputs:
            if getattr(self.inputs[key][-1], "gi_suspended", False):
                try:
                    self.inputs[key][-1].send(1)
                except StopIteration:
                    pass
        self.inputs = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "frames_per_batch": ("INT", {"default": 16, "min": 1, "max": BIGMAX, "step": 1})
                    },
                "hidden": {
                    "prompt": "PROMPT",
                    "unique_id": "UNIQUE_ID"
                },
                }

    RETURN_TYPES = ("VHS_BatchManager",)
    RETURN_NAMES = ("meta_batch",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "update_batch"

    def update_batch(self, frames_per_batch, prompt=None, unique_id=None):
        if unique_id is not None and prompt is not None:
            requeue = prompt[unique_id]['inputs'].get('requeue', 0)
        else:
            requeue = 0
        if requeue == 0:
            self.reset()
            self.frames_per_batch = frames_per_batch
            self.unique_id = unique_id
        else:
            num_batches = (self.total_frames+self.frames_per_batch-1)//frames_per_batch
            print(f'Meta-Batch {requeue}/{num_batches}')
        #onExecuted seems to not be called unless some message is sent
        return (self,)


class VideoInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "video_info": ("VHS_VIDEOINFO",),
                    }
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("FLOAT","INT", "FLOAT", "INT", "INT", "FLOAT","INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = (
        "source_fps🟨",
        "source_frame_count🟨",
        "source_duration🟨",
        "source_width🟨",
        "source_height🟨",
        "loaded_fps🟦",
        "loaded_frame_count🟦",
        "loaded_duration🟦",
        "loaded_width🟦",
        "loaded_height🟦",
    )
    FUNCTION = "get_video_info"

    def get_video_info(self, video_info):
        keys = ["fps", "frame_count", "duration", "width", "height"]

        source_info = []
        loaded_info = []

        for key in keys:
            source_info.append(video_info[f"source_{key}"])
            loaded_info.append(video_info[f"loaded_{key}"])

        return (*source_info, *loaded_info)


class VideoInfoSource:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "video_info": ("VHS_VIDEOINFO",),
                    }
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("FLOAT","INT", "FLOAT", "INT", "INT",)
    RETURN_NAMES = (
        "fps🟨",
        "frame_count🟨",
        "duration🟨",
        "width🟨",
        "height🟨",
    )
    FUNCTION = "get_video_info"

    def get_video_info(self, video_info):
        keys = ["fps", "frame_count", "duration", "width", "height"]

        source_info = []

        for key in keys:
            source_info.append(video_info[f"source_{key}"])

        return (*source_info,)


class VideoInfoLoaded:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "video_info": ("VHS_VIDEOINFO",),
                    }
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("FLOAT","INT", "FLOAT", "INT", "INT",)
    RETURN_NAMES = (
        "fps🟦",
        "frame_count🟦",
        "duration🟦",
        "width🟦",
        "height🟦",
    )
    FUNCTION = "get_video_info"

    def get_video_info(self, video_info):
        keys = ["fps", "frame_count", "duration", "width", "height"]

        loaded_info = []

        for key in keys:
            loaded_info.append(video_info[f"loaded_{key}"])

        return (*loaded_info,)

class SelectFilename:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"filenames": ("VHS_FILENAMES",), "index": ("INT", {"default": -1, "step": 1, "min": -1})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES =("Filename",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "select_filename"

    def select_filename(self, filenames, index):
        return (filenames[1][index],)

class Unbatch:
    class Any(str):
        def __ne__(self, other):
            return False
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"batched": ("*",)}}
    RETURN_TYPES = (Any('*'),)
    INPUT_IS_LIST = True
    RETURN_NAMES =("unbatched",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "unbatch"
    def unbatch(self, batched):
        if isinstance(batched[0], torch.Tensor):
            return (torch.cat(batched),)
        if isinstance(batched[0], dict):
            out = batched[0].copy()
            if 'samples' in out:
                out['samples'] = torch.cat([x['samples'] for x in batched])
            if 'waveform' in out:
                out['waveform'] = torch.cat([x['waveform'] for x in batched])
            out.pop('batch_index', None)
            return (out,)
        return (functools.reduce(lambda x,y: x+y, batched),)
    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

class SelectLatest:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"filename_prefix": ("STRING", {'default': 'output/AnimateDiff', 'vhs_path_extensions': []}),
                             "filename_postfix": ("STRING", {"placeholder": ".webm"})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES =("Filename",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "select_latest"
    EXPERIMENTAL = True

    def select_latest(self, filename_prefix, filename_postfix):
        assert False, "Not Reachable"

# Custom nodes - only available if dependencies are installed
if HAS_CUSTOM_FEATURES:
    def get_audio_duration(audio_file):
        """
        Get the duration of an audio file in seconds.

        Args:
            audio_file: The absolute path to the audio file.

        Returns:
            The duration of the audio file in seconds.
        """
        ffmpeg_cmd = [ffmpeg_path or 'ffmpeg', '-i', audio_file]
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
        CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/audio"
        FUNCTION = "merge_audio"

        @classmethod
        def merge_audio(cls, audio_file_1, audio_file_2, output_file_name):
            # Get the absolute path of the output file
            _datetime = datetime.datetime.now().strftime("%Y%m%d")
            _datetime = _datetime + datetime.datetime.now().strftime("%H%M%S%f")

            output_file_abs = os.path.join(folder_paths.get_output_directory(), f"audio_{_datetime}.mp3")

            # Construct FFmpeg command
            ffmpeg_cmd = [ffmpeg_path or 'ffmpeg', '-i', audio_file_1, '-i', audio_file_2, '-filter_complex',
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
            return hash_path(output_file_name)

        @classmethod
        def VALIDATE_INPUTS(cls, audio_file_1, audio_file_2, **kwargs):
            if not validate_path(audio_file_1, allow_none=True) or not validate_path(audio_file_2, allow_none=True):
                return False
            return True

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
                    "video_path": ("STRING",{"default": "", "tooltip": "输入视频路径（支持相对/绝对路径）"}),
                    "output_filename": ("STRING",{"default": "_captioned", "tooltip": "输出文件名前缀"}),
                    "is_vertical": ("BOOLEAN",{"default": True, "tooltip": "是否按竖屏流程处理（裁剪/缩放策略）"}),
                    "add_subscription_anim": ("BOOLEAN",{"default": False, "tooltip": "预留参数，当前版本未启用"}),
                    "notify_all": ("BOOLEAN",{"default":True, "tooltip": "完成后发送通知"})
                },
                "optional": {
                    "audio_path": ("STRING",{"default":"", "tooltip": "可选音频路径；为空时使用视频原音轨"}),
                    "water_mark": ("STRING",{"default":"OnePieOne", "tooltip": "预留参数，当前版本未启用"}),
                    "caption_json_param": ("STRING",{"default":f"{default_template_para}","multiline":True, "tooltip": "字幕样式参数（JSON/Python字典）"}),
                    "notify_message": ("STRING",{"default":"====Caption====", "tooltip": "通知消息内容"})       
                },
                "hidden": {},
            }

        RETURN_TYPES = ("STRING", "INT")
        RETURN_NAMES = ("Video Path", "frames")
        OUTPUT_NODE = True
        CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
        FUNCTION = "add_captions"

        def add_captions(self, video_path, output_filename, audio_path,is_vertical,add_subscription_anim,water_mark,notify_all,caption_json_param,notify_message):
            if not HAS_GENTLE_CAPTION:
                raise RuntimeError("VideoCaptions 需要 caption 模块依赖")

            video_path = resolve_media_path(video_path, must_exist=True)
            audio_path = resolve_media_path(audio_path, must_exist=True) if str(audio_path or "").strip() else ""
            output_filename = str(output_filename or "_captioned").strip() or "_captioned"

            cp = GentleCaption()
            params = cp.parse_caption_params(caption_json_param)
            _real_filename, output_path, frames = cp.make_video(
                bg_video_path=video_path,
                bg_audio_path=audio_path,
                output_filename=output_filename,
                extra_para=params,
                is_vertical=is_vertical,
            )
            if notify_all:
                notifyAll(output_path, f"{notify_message}")
            return {
                "ui": {
                    "video": [{"filename": os.path.basename(output_path), "subfolder": "", "type": "output"}],
                    "frames": [frames],
                },
                "result": (output_path, frames),
            }

        def _time_captions(self, audio_path, is_vertical=True):
            if not HAS_SHORTGPT:
                return []
            whisper_analysis = audio_utils.audioToText(audio_path)
            max_len = 30 if is_vertical else 15
            return captions.getCaptionsWithTime(whisper_analysis, maxCaptionSize=max_len)

    class VideoGentleCaptions:
        def __init__(self) -> None:
            self.cp = None
        
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
                    "video_path": ("STRING",{"default": "", "tooltip": "输入视频路径（支持相对/绝对路径）"}),
                    "output_filename": ("STRING",{"default": "_captioned", "tooltip": "输出文件名前缀"}),
                    "is_vertical": ("BOOLEAN",{"default": True, "tooltip": "是否按竖屏流程处理（裁剪/缩放策略）"}),
                    "notify_all": ("BOOLEAN",{"default":True, "tooltip": "完成后发送通知"})
                },
                "optional": {
                    "audio_path": ("STRING",{"default":"", "tooltip": "可选音频路径；为空时使用视频原音轨"}),
                    "caption_json_param": ("STRING",{"default":f"{default_template_para}","multiline":True, "tooltip": "字幕样式参数（JSON/Python字典）"}),
                    "notify_message": ("STRING",{"default":"====Caption====", "tooltip": "通知消息内容"})   
                },
                "hidden": {},
            }

        RETURN_TYPES = ("STRING", "INT")
        RETURN_NAMES = ("Video Path", "frames")
        OUTPUT_NODE = True
        CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
        FUNCTION = "add_gentle_captions"

        def add_gentle_captions(self, video_path, output_filename, audio_path,is_vertical,notify_all,caption_json_param,notify_message):
            if not HAS_GENTLE_CAPTION:
                raise RuntimeError("VideoGentleCaptions 需要 caption 模块依赖")

            video_path = resolve_media_path(video_path, must_exist=True)
            audio_path = resolve_media_path(audio_path, must_exist=True) if str(audio_path or "").strip() else ""
            output_filename = str(output_filename or "_captioned").strip() or "_captioned"

            if self.cp is None:
                self.cp = GentleCaption()

            params = self.cp.parse_caption_params(caption_json_param)
            real_filename, video_result, frames = self.cp.make_video(
                bg_video_path=video_path,
                bg_audio_path=audio_path,
                output_filename=output_filename,
                extra_para=params,
                is_vertical=is_vertical,
            )

            if notify_all:
                notifyAll(video_result, f"{notify_message}")
            
            previews = [
                {
                    "filename": real_filename,
                    "subfolder": "",
                    "type": "output",
                    "format": "video/h264-mp4",
                }
            ]
            return {
                "ui": {
                    "gifs": previews,
                    "video": [{"filename": real_filename, "subfolder": "", "type": "output"}],
                    "frames": [frames],
                },
                "result": (video_result, frames),
            }

    class CaptionStylePreset:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "preset": (
                        [
                            "classic_white",
                            "karaoke_glow",
                            "minimal_bottom",
                            "bold_vertical",
                            "custom_only",
                        ],
                        {"default": "classic_white"},
                    ),
                    "is_vertical": ("BOOLEAN", {"default": True}),
                },
                "optional": {
                    "custom_override_json": (
                        "STRING",
                        {
                            "default": "",
                            "multiline": True,
                            "tooltip": "可选：覆盖预置参数，支持 JSON/Python 字典风格字符串",
                        },
                    ),
                },
            }

        RETURN_TYPES = ("STRING", "STRING")
        RETURN_NAMES = ("caption_json_param", "preset_name")
        CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/caption"
        FUNCTION = "build_preset"

        @staticmethod
        def _preset_dict(preset: str, is_vertical: bool):
            base = {
                "Fontname": "Roboto-Bold",
                "Alignment": 5 if is_vertical else 2,
                "BorderStyle": "1",
                "Outline": "2",
                "Shadow": "1",
                "Blur": "6",
                "Fontsize": 26 if is_vertical else 20,
                "MarginL": "0",
                "MarginR": "0",
                "word_level": True,
                "segment_level": True,
                "vad": True,
            }
            if preset == "karaoke_glow":
                base.update(
                    {
                        "Fontname": "Lemon-Regular",
                        "Blur": "18",
                        "Outline": "1",
                        "Shadow": "3",
                        "highlight_color": "white",
                        "karaoke": True,
                    }
                )
            elif preset == "minimal_bottom":
                base.update(
                    {
                        "Alignment": 2,
                        "Blur": "0",
                        "Outline": "1",
                        "Shadow": "0",
                        "Fontsize": 18 if is_vertical else 16,
                        "word_level": False,
                    }
                )
            elif preset == "bold_vertical":
                base.update(
                    {
                        "Alignment": 5,
                        "Fontsize": 30,
                        "Outline": "3",
                        "Shadow": "2",
                        "Blur": "4",
                    }
                )
            elif preset == "custom_only":
                base = {}
            return base

        def build_preset(self, preset, is_vertical=True, custom_override_json=""):
            if not HAS_GENTLE_CAPTION:
                raise RuntimeError("CaptionStylePreset 需要 caption 模块依赖")
            cp = GentleCaption()
            style = self._preset_dict(preset, is_vertical)
            custom_text = str(custom_override_json or "").strip()
            if custom_text:
                custom = cp.parse_caption_params(custom_text)
                if not isinstance(custom, dict):
                    raise ValueError("custom_override_json 必须是对象字典")
                style.update(custom)
            return (json.dumps(style, ensure_ascii=False, indent=2), preset)

    class VideoGentleCaptionsPro:
        def __init__(self) -> None:
            self.cp = None

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "video_path": ("STRING", {"default": "", "tooltip": "输入视频路径（支持相对/绝对路径）"}),
                    "output_filename": ("STRING", {"default": "_captioned_pro", "tooltip": "输出文件名前缀"}),
                    "is_vertical": ("BOOLEAN", {"default": True, "tooltip": "竖屏/横屏策略"}),
                    "style_preset": (
                        [
                            "classic_white",
                            "karaoke_glow",
                            "minimal_bottom",
                            "bold_vertical",
                            "custom_only",
                        ],
                        {"default": "classic_white"},
                    ),
                    "sync_tolerance_ms": ("INT", {"default": 120, "min": 0, "max": 5000, "step": 1}),
                    "show_progress": ("BOOLEAN", {"default": True}),
                    "notify_all": ("BOOLEAN", {"default": True, "tooltip": "完成后发送通知"}),
                },
                "optional": {
                    "audio_path": ("STRING", {"default": "", "tooltip": "可选音频路径；为空时使用视频原音轨"}),
                    "caption_json_param": (
                        "STRING",
                        {
                            "default": "",
                            "multiline": True,
                            "tooltip": "可选：基础字幕参数（JSON/Python字典）",
                        },
                    ),
                    "custom_override_json": (
                        "STRING",
                        {
                            "default": "",
                            "multiline": True,
                            "tooltip": "可选：覆盖参数（优先级最高）",
                        },
                    ),
                    "notify_message": ("STRING", {"default": "====Caption Pro====", "tooltip": "通知消息内容"}),
                },
                "hidden": {},
            }

        RETURN_TYPES = ("STRING", "INT", "FLOAT", "BOOLEAN", "STRING")
        RETURN_NAMES = ("Video Path", "frames", "sync_drift_ms", "sync_ok", "applied_caption_json")
        OUTPUT_NODE = True
        CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/caption"
        FUNCTION = "add_gentle_captions_pro"

        def add_gentle_captions_pro(
            self,
            video_path,
            output_filename,
            is_vertical,
            style_preset,
            sync_tolerance_ms,
            show_progress,
            notify_all,
            audio_path="",
            caption_json_param="",
            custom_override_json="",
            notify_message="====Caption Pro====",
        ):
            if not HAS_GENTLE_CAPTION:
                raise RuntimeError("VideoGentleCaptionsPro 需要 caption 模块依赖")

            if self.cp is None:
                self.cp = GentleCaption()

            pbar = ProgressBar(5) if show_progress else None

            video_path = resolve_media_path(video_path, must_exist=True)
            if pbar is not None:
                pbar.update(1)

            resolved_audio_path = resolve_media_path(audio_path, must_exist=True) if str(audio_path or "").strip() else ""
            output_filename = str(output_filename or "_captioned_pro").strip() or "_captioned_pro"

            params = CaptionStylePreset._preset_dict(style_preset, is_vertical)
            if str(caption_json_param or "").strip():
                params.update(self.cp.parse_caption_params(caption_json_param))
            if str(custom_override_json or "").strip():
                params.update(self.cp.parse_caption_params(custom_override_json))
            if pbar is not None:
                pbar.update(1)

            real_filename, video_result, frames = self.cp.make_video(
                bg_video_path=video_path,
                bg_audio_path=resolved_audio_path,
                output_filename=output_filename,
                extra_para=params,
                is_vertical=is_vertical,
            )
            if pbar is not None:
                pbar.update(2)

            output_video_info = self.cp.get_media_info(video_result, kind="video")
            ref_audio_path = resolved_audio_path if resolved_audio_path else video_path
            ref_audio_info = self.cp.get_media_info(ref_audio_path, kind="audio")
            drift_ms = abs(float(output_video_info["duration"]) - float(ref_audio_info["duration"])) * 1000.0
            sync_ok = drift_ms <= float(sync_tolerance_ms)
            if pbar is not None:
                pbar.update(1)

            if notify_all:
                notifyAll(video_result, f"{notify_message} | sync_ok={sync_ok} | drift_ms={drift_ms:.2f}")

            previews = [
                {
                    "filename": real_filename,
                    "subfolder": "",
                    "type": "output",
                    "format": "video/h264-mp4",
                }
            ]
            applied_json = json.dumps(params, ensure_ascii=False, indent=2)
            return {
                "ui": {
                    "gifs": previews,
                    "video": [{"filename": real_filename, "subfolder": "", "type": "output"}],
                    "frames": [frames],
                    "sync_drift_ms": [drift_ms],
                    "sync_ok": [sync_ok],
                },
                "result": (video_result, frames, drift_ms, sync_ok, applied_json),
            }

    def _srt_ts_to_sec(raw):
        x = str(raw).strip().replace(".", ",")
        hh, mm, ss_ms = x.split(":")
        ss, ms = ss_ms.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms[:3].ljust(3, "0")) / 1000.0

    class MovisSubtitleFromSRT:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "timeline": ("MOVIS_TIMELINE",),
                    "srt_path": ("STRING", {"default": "", "tooltip": "SRT 文件路径"}),
                    "font_size": ("FLOAT", {"default": 54.0, "min": 8.0, "max": 300.0}),
                    "font_family": ("STRING", {"default": "Sans Serif"}),
                    "color": ("STRING", {"default": "white"}),
                    "position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                    "position_y": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                    "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                    "start_offset": ("FLOAT", {"default": 0.0, "min": -86400.0, "max": 86400.0}),
                    "replace_existing": ("BOOLEAN", {"default": False}),
                }
            }

        RETURN_TYPES = ("MOVIS_TIMELINE", "INT")
        RETURN_NAMES = ("timeline", "subtitle_count")
        CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
        FUNCTION = "add_srt"

        def add_srt(self, timeline, srt_path, font_size, font_family, color, position_x, position_y, opacity, start_offset, replace_existing):
            t = json.loads(json.dumps(timeline))
            real_srt = resolve_media_path(srt_path, must_exist=True)

            content = None
            for enc in ("utf-8-sig", "utf-8", "gbk"):
                try:
                    with open(real_srt, "r", encoding=enc) as f:
                        content = f.read()
                    break
                except Exception:
                    continue
            if content is None:
                raise RuntimeError("无法读取 SRT 文件编码")

            if replace_existing:
                t["text_tracks"] = []
            elif "text_tracks" not in t or not isinstance(t.get("text_tracks"), list):
                t["text_tracks"] = []

            block_re = re.compile(
                r"(?ms)^\\s*\\d+\\s*\\n\\s*(\\d{2}:\\d{2}:\\d{2}[,.]\\d{1,3})\\s*-->\\s*(\\d{2}:\\d{2}:\\d{2}[,.]\\d{1,3}).*?\\n(.*?)(?=\\n\\s*\\n|\\Z)"
            )
            count = 0
            for m in block_re.finditer(content):
                st = _srt_ts_to_sec(m.group(1)) + float(start_offset)
                et = _srt_ts_to_sec(m.group(2)) + float(start_offset)
                txt = re.sub(r"\r?\n", " ", m.group(3)).strip()
                if not txt:
                    continue
                dur = max(0.01, et - st)
                t["text_tracks"].append(
                    {
                        "text": txt,
                        "start": max(0.0, st),
                        "duration": dur,
                        "font_size": float(font_size),
                        "font_family": str(font_family),
                        "color": str(color),
                        "position_x": float(position_x),
                        "position_y": float(position_y),
                        "opacity": float(opacity),
                    }
                )
                count += 1
            return (t, count)

    class MovisAutoCaptionTimeline:
        def __init__(self) -> None:
            self.cp = None
            self._has_audio_cache = {}

        def _resolve_if_valid(self, raw_path: str):
            p = str(raw_path or "").strip()
            if not p:
                return None
            try:
                return resolve_media_path(p, must_exist=True)
            except Exception as e:
                logger.warn(f"[MovisAutoCaptionTimeline] 路径不可用，已忽略: {p} | {e}")
                return None

        def _infer_video_path_from_timeline(self, timeline):
            tracks = timeline.get("video_tracks", []) if isinstance(timeline, dict) else []
            if not isinstance(tracks, list):
                return None
            for clip in tracks:
                if not isinstance(clip, dict):
                    continue
                path = self._resolve_if_valid(clip.get("path", ""))
                if path:
                    return path
            return None

        def _infer_audio_track_path_from_timeline(self, timeline):
            if not isinstance(timeline, dict):
                return None

            audio_tracks = timeline.get("audio_tracks", [])
            if isinstance(audio_tracks, list):
                for track in audio_tracks:
                    if not isinstance(track, dict):
                        continue
                    path = self._resolve_if_valid(track.get("path", ""))
                    if path:
                        return path

            return None

        def _infer_bgm_path_from_timeline(self, timeline):
            if not isinstance(timeline, dict):
                return None

            bgm = timeline.get("bgm", {})
            if isinstance(bgm, dict):
                path = self._resolve_if_valid(bgm.get("path", ""))
                if path:
                    return path

            return None

        def _media_has_audio_stream(self, media_path: str) -> bool:
            path = str(media_path or "").strip()
            if not path:
                return False
            if path in self._has_audio_cache:
                return self._has_audio_cache[path]

            has_audio = False
            try:
                ffprobe_exe = getattr(self.cp, "ffprobe_exe", None)
                if not ffprobe_exe:
                    ffprobe_exe = "ffprobe"
                probe_cmd = [
                    ffprobe_exe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path,
                ]
                ret = subprocess.run(probe_cmd, capture_output=True, text=True)
                has_audio = ret.returncode == 0 and bool(str(ret.stdout or "").strip())
            except Exception as e:
                logger.warn(f"[MovisAutoCaptionTimeline] 音轨探测失败，按无音频处理: {path} | {e}")
                has_audio = False

            self._has_audio_cache[path] = has_audio
            return has_audio

        def _clip_can_use_source_audio(self, clip):
            if not isinstance(clip, dict):
                return False
            if not bool(clip.get("use_source_audio", True)):
                return False
            cpath = str(clip.get("path") or "").strip()
            if not cpath:
                return False
            return self._media_has_audio_stream(cpath)

        def _extract_video_clips(self, timeline):
            clips = []
            tracks = timeline.get("video_tracks", []) if isinstance(timeline, dict) else []
            if not isinstance(tracks, list):
                return clips
            for idx, clip in enumerate(tracks):
                if not isinstance(clip, dict):
                    continue
                path = self._resolve_if_valid(clip.get("path", ""))
                start = float(clip.get("start", 0.0) or 0.0)
                duration = float(clip.get("duration", 0.0) or 0.0)
                source_start = float(clip.get("source_start", 0.0) or 0.0)
                if duration <= 0:
                    continue
                clips.append({
                    "idx": idx,
                    "path": path,
                    "start": start,
                    "duration": duration,
                    "end": start + duration,
                    "source_start": source_start,
                    "use_source_audio": bool(clip.get("use_source_audio", True)),
                })
            clips.sort(key=lambda x: (x["start"], x["idx"]))
            return clips

        def _extract_audio_tracks(self, timeline):
            tracks_out = []
            tracks = timeline.get("audio_tracks", []) if isinstance(timeline, dict) else []
            if not isinstance(tracks, list):
                return tracks_out
            for idx, track in enumerate(tracks):
                if not isinstance(track, dict):
                    continue
                path = self._resolve_if_valid(track.get("path", ""))
                duration = float(track.get("duration", 0.0) or 0.0)
                start = float(track.get("start", 0.0) or 0.0)
                source_start = float(track.get("source_start", 0.0) or 0.0)
                if duration <= 0 or not path:
                    continue
                tracks_out.append({
                    "idx": idx,
                    "path": path,
                    "start": start,
                    "duration": duration,
                    "end": start + duration,
                    "source_start": source_start,
                })
            tracks_out.sort(key=lambda x: (x["start"], x["idx"]))
            return tracks_out

        def _timeline_visible_window(self, video_clips, audio_tracks):
            """返回字幕可见时间窗（优先视频时间窗）。"""
            if isinstance(video_clips, list) and len(video_clips) > 0:
                s = min(float(c.get("start", 0.0)) for c in video_clips)
                e = max(float(c.get("end", 0.0)) for c in video_clips)
                if e - s > 0.01:
                    return s, e
            if isinstance(audio_tracks, list) and len(audio_tracks) > 0:
                s = min(float(a.get("start", 0.0)) for a in audio_tracks)
                e = max(float(a.get("end", 0.0)) for a in audio_tracks)
                if e - s > 0.01:
                    return s, e
            return 0.0, 10**9

        def _timeline_canvas_size(self, timeline):
            canvas = timeline.get("canvas", {}) if isinstance(timeline, dict) else {}
            try:
                w = max(64.0, float(canvas.get("width", 1080) or 1080))
                h = max(64.0, float(canvas.get("height", 1920) or 1920))
            except Exception:
                w, h = 1080.0, 1920.0
            return w, h

        def _adapt_caption_layout(self, timeline, position_x, position_y, font_size, auto_layout=True, layout_mode="auto_safe", safe_margin_ratio=None, font_scale=1.0):
            """按画布尺寸/长宽比做字幕布局自适配，同时允许保留手工控制。"""
            px = max(0.0, min(1.0, float(position_x)))
            py = max(0.0, min(1.0, float(position_y)))
            fs = max(8.0, float(font_size))
            mode = str(layout_mode or "auto_safe").strip().lower()
            if not bool(auto_layout) or mode == "manual":
                return px, py, fs

            cw, ch = self._timeline_canvas_size(timeline)
            short_edge = max(1.0, min(cw, ch))
            # 以 1080 短边为参考，自适配字体尺寸（默认“智能自动”）
            scale = max(0.65, min(1.35, short_edge / 1080.0))
            aspect = cw / max(1.0, ch)
            # 极端宽屏场景下适当收小字幕，避免压画面
            if aspect > 2.0:
                scale *= max(0.78, 1.0 - (aspect - 2.0) * 0.08)
            try:
                scale *= max(0.5, min(2.0, float(font_scale)))
            except Exception:
                pass
            fs = max(12.0, min(220.0, fs * scale))

            # 根据字体像素高度推导安全边距，避免超宽/超窄时字幕出框
            if safe_margin_ratio is None:
                safe_band = max(0.04, min(0.22, (fs * 1.6) / max(64.0, ch)))
            else:
                try:
                    safe_band = max(0.02, min(0.35, float(safe_margin_ratio)))
                except Exception:
                    safe_band = max(0.04, min(0.22, (fs * 1.6) / max(64.0, ch)))
            # 当用户仍用默认底部位置（0.9）时，自动微调到底部安全区
            if mode == "auto_bottom":
                py = 1.0 - safe_band
            elif abs(float(position_y) - 0.9) < 1e-6:
                py = min(py, 1.0 - safe_band)
            py = max(safe_band * 0.7, min(1.0 - safe_band, py))
            return px, py, fs

        def _normalize_audio_strategy(self, raw):
            v = str(raw or "auto").strip().lower()
            if v in {"auto", "mixed_priority", "explicit_only", "audio_tracks_only", "bgm_only", "video_only"}:
                return v
            return "auto"

        def _normalize_layout_mode(self, raw, auto_layout):
            if str(raw or "").strip() == "":
                return "auto_safe" if bool(auto_layout) else "manual"
            v = str(raw).strip().lower()
            if v in {"auto_safe", "auto_bottom", "manual"}:
                return v
            return "auto_safe" if bool(auto_layout) else "manual"

        def _safe_float(self, value, default=None):
            try:
                return float(value)
            except Exception:
                return default

        def _override_resolved_path(self, override_dict, key):
            if not isinstance(override_dict, dict):
                return None
            return self._resolve_if_valid(override_dict.get(key, ""))

        def _overlap(self, a_start: float, a_end: float, b_start: float, b_end: float) -> float:
            return max(0.0, min(a_end, b_end) - max(a_start, b_start))

        def _pick_audio_for_clip(self, clip, audio_tracks, used_audio_idx):
            if not audio_tracks:
                return None
            c_start, c_end = float(clip["start"]), float(clip["end"])
            c_dur = max(0.01, c_end - c_start)

            best = None
            best_key = None
            for t in audio_tracks:
                ov = self._overlap(c_start, c_end, float(t["start"]), float(t["end"]))
                start_diff = abs(float(t["start"]) - c_start)
                used_penalty = 100000.0 if t["idx"] in used_audio_idx else 0.0
                key = (
                    -ov,
                    start_diff + used_penalty,
                    abs(float(t["duration"]) - c_dur),
                    t["idx"],
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = t

            if best is None:
                return None

            ov = self._overlap(c_start, c_end, float(best["start"]), float(best["end"]))
            max_start_diff = max(2.0, c_dur * 0.25)
            if ov <= 0 and abs(float(best["start"]) - c_start) > max_start_diff:
                return None
            return best

        def _append_segment_text(self, t, seg, source_start, timeline_start, timeline_end, start_offset, position_x, position_y, default_font_size, default_font_family, default_color, default_opacity):
            if isinstance(seg, dict):
                st = float(seg.get("start", 0.0))
                et = float(seg.get("end", st + 0.01))
                txt = str(seg.get("text", "")).strip()
            else:
                st = float(getattr(seg, "start", 0.0))
                et = float(getattr(seg, "end", st + 0.01))
                txt = str(getattr(seg, "text", "")).strip()
            if not txt:
                return 0

            out_st = float(timeline_start) + (st - float(source_start)) + float(start_offset)
            out_et = float(timeline_start) + (et - float(source_start)) + float(start_offset)

            if out_et <= float(timeline_start) or out_st >= float(timeline_end):
                return 0

            out_st = max(float(timeline_start), out_st)
            out_et = min(float(timeline_end), out_et)
            if out_et - out_st < 0.01:
                return 0

            t["text_tracks"].append(
                {
                    "text": txt,
                    "start": max(0.0, out_st),
                    "duration": max(0.01, out_et - out_st),
                    "font_size": float(default_font_size),
                    "font_family": str(default_font_family),
                    "color": str(default_color),
                    "position_x": float(position_x),
                    "position_y": float(position_y),
                    "opacity": float(default_opacity),
                }
            )
            return 1

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "timeline": ("MOVIS_TIMELINE",),
                    "video_path": ("STRING", {"default": "", "tooltip": "视频路径（用于参考时长；可不与时间线视频一致）"}),
                    "audio_path": ("STRING", {"default": "", "tooltip": "可选；为空时使用视频音轨"}),
                    "style_preset": (
                        ["classic_white", "karaoke_glow", "minimal_bottom", "bold_vertical", "custom_only"],
                        {"default": "classic_white"},
                    ),
                    "position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                    "position_y": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                    "start_offset": ("FLOAT", {"default": 0.0, "min": -86400.0, "max": 86400.0}),
                    "sync_tolerance_ms": ("INT", {"default": 120, "min": 0, "max": 5000}),
                    "replace_existing": ("BOOLEAN", {"default": False}),
                    "show_progress": ("BOOLEAN", {"default": True}),
                    "auto_layout": ("BOOLEAN", {"default": True, "tooltip": "根据时间线尺寸与宽高比自动适配字幕字号与安全区；关闭后严格使用手动位置/字号"}),
                },
                "optional": {
                    "layout_mode": (["auto_safe", "auto_bottom", "manual"], {"default": "auto_safe", "tooltip": "字幕布局策略：安全自适配/底部贴边自适配/完全手动"}),
                    "audio_source_strategy": (["auto", "mixed_priority", "explicit_only", "audio_tracks_only", "bgm_only", "video_only"], {"default": "auto", "tooltip": "音频识别来源策略"}),
                    "safe_margin_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.35, "step": 0.005, "tooltip": "字幕安全边距比例；0表示自动"}),
                    "font_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": "字幕字号缩放（相对样式字号）"}),
                    "caption_json_param": ("STRING", {"default": "", "multiline": True}),
                    "custom_override_json": ("STRING", {"default": "", "multiline": True}),
                },
            }

        RETURN_TYPES = ("MOVIS_TIMELINE", "INT", "FLOAT", "BOOLEAN", "STRING")
        RETURN_NAMES = ("timeline", "subtitle_count", "sync_drift_ms", "sync_ok", "applied_caption_json")
        CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
        FUNCTION = "auto_caption"

        def _extract_segments(self, transcribe_obj):
            segs = getattr(transcribe_obj, "segments", None)
            if isinstance(segs, list):
                return segs
            if isinstance(transcribe_obj, dict) and isinstance(transcribe_obj.get("segments"), list):
                return transcribe_obj.get("segments")
            return []

        def _transcribe_cached(self, path, cache):
            if not path:
                return []
            if path in cache:
                return cache[path]
            try:
                self.cp._ensure_model()
                transcribe = self.cp.model.transcribe(path, regroup=True, fp16=torch.cuda.is_available())
                segs = self._extract_segments(transcribe)
                cache[path] = segs
                logger.info(f"[MovisAutoCaptionTimeline] 识别完成: {path} | segments={len(segs)}")
                return segs
            except Exception as e:
                logger.warn(f"[MovisAutoCaptionTimeline] 识别失败，已跳过: {path} | {e}")
                cache[path] = []
                return []

        def _transcribe_window_cached(self, path, window_start, window_duration, cache):
            """仅转写音频窗口，减少超长 BGM 的无效识别耗时。"""
            if not path:
                return []
            ws = max(0.0, float(window_start or 0.0))
            wd = max(0.0, float(window_duration or 0.0))
            if wd <= 0.01:
                return []
            cache_key = f"{path}@@window:{ws:.3f}:{wd:.3f}"
            if cache_key in cache:
                return cache[cache_key]

            ffmpeg_exe = getattr(self.cp, "ffmpeg_exe", None) or ffmpeg_path or "ffmpeg"
            tmp_clip = None
            try:
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    prefix="vhs_caption_window_",
                    suffix=".wav",
                    dir=folder_paths.get_temp_directory(),
                    delete=False,
                ) as tf:
                    tmp_clip = tf.name

                cmd = [
                    ffmpeg_exe,
                    "-v",
                    "error",
                    "-ss",
                    f"{ws:.3f}",
                    "-t",
                    f"{wd:.3f}",
                    "-i",
                    path,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    "-y",
                    tmp_clip,
                ]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0:
                    raise RuntimeError((ret.stderr or ret.stdout or "ffmpeg clip failed").strip())

                raw_segs = self._transcribe_cached(tmp_clip, cache)
                shifted = []
                for seg in raw_segs:
                    if isinstance(seg, dict):
                        out = dict(seg)
                        out["start"] = float(out.get("start", 0.0)) + ws
                        out["end"] = float(out.get("end", out.get("start", 0.0))) + ws
                        shifted.append(out)
                    else:
                        st = float(getattr(seg, "start", 0.0)) + ws
                        et = float(getattr(seg, "end", st)) + ws
                        txt = str(getattr(seg, "text", ""))
                        shifted.append({"start": st, "end": et, "text": txt})

                cache[cache_key] = shifted
                logger.info(
                    f"[MovisAutoCaptionTimeline] BGM窗口识别: start={ws:.3f}s dur={wd:.3f}s segments={len(shifted)}"
                )
                return shifted
            except Exception as e:
                logger.warn(f"[MovisAutoCaptionTimeline] BGM窗口识别失败，回退全量识别: {e}")
                segs = self._transcribe_cached(path, cache)
                cache[cache_key] = segs
                return segs
            finally:
                try:
                    if tmp_clip and os.path.exists(tmp_clip):
                        os.remove(tmp_clip)
                except Exception:
                    pass

        def auto_caption(
            self,
            timeline,
            video_path,
            audio_path,
            style_preset,
            position_x,
            position_y,
            start_offset,
            sync_tolerance_ms,
            replace_existing,
            show_progress,
            auto_layout,
            layout_mode="auto_safe",
            audio_source_strategy="auto",
            safe_margin_ratio=0.0,
            font_scale=1.0,
            caption_json_param="",
            custom_override_json="",
        ):
            if not HAS_GENTLE_CAPTION:
                raise RuntimeError("MovisAutoCaptionTimeline 需要 caption 模块依赖")

            if self.cp is None:
                self.cp = GentleCaption()
            pbar = ProgressBar(5) if show_progress else None

            t = json.loads(json.dumps(timeline))
            explicit_video = self._resolve_if_valid(video_path)
            explicit_audio = self._resolve_if_valid(audio_path)
            inferred_video = self._infer_video_path_from_timeline(t)
            inferred_track_audio = self._infer_audio_track_path_from_timeline(t)
            inferred_bgm_audio = self._infer_bgm_path_from_timeline(t)
            video_clips = self._extract_video_clips(t)
            audio_tracks = self._extract_audio_tracks(t)
            visible_start, visible_end = self._timeline_visible_window(video_clips, audio_tracks)
            bgm_meta = t.get("bgm") if isinstance(t.get("bgm"), dict) else {}
            bgm_source_start = float(bgm_meta.get("source_start", 0.0) or 0.0)
            bgm_trim_to_video_length = bool(bgm_meta.get("trim_to_video_length", True))

            valid_audio_tracks = []
            for tr in audio_tracks:
                tpath = tr.get("path")
                if tpath and self._media_has_audio_stream(tpath):
                    valid_audio_tracks.append(tr)
                elif tpath:
                    logger.warn(f"[MovisAutoCaptionTimeline] audio_track#{tr.get('idx', -1)} 无音频流，已忽略: {tpath}")

            clips_with_audio_source = [c for c in video_clips if self._clip_can_use_source_audio(c)]

            vpath = explicit_video or inferred_video
            apath = explicit_audio or inferred_track_audio or inferred_bgm_audio or vpath

            if explicit_video:
                logger.info(f"[MovisAutoCaptionTimeline] 使用显式 video_path: {explicit_video}")
            elif inferred_video:
                logger.info(f"[MovisAutoCaptionTimeline] video_path 为空，已从 timeline.video_tracks 自动推断: {inferred_video}")
            else:
                logger.warn("[MovisAutoCaptionTimeline] video_path 为空，且 timeline 中无可用视频路径；将跳过时长对齐检测")

            if explicit_audio:
                logger.info(f"[MovisAutoCaptionTimeline] 使用显式 audio_path: {explicit_audio}")
            elif inferred_track_audio:
                logger.info(f"[MovisAutoCaptionTimeline] audio_path 为空，已从 timeline.audio_tracks 自动推断: {inferred_track_audio}")
            elif inferred_bgm_audio:
                logger.info(f"[MovisAutoCaptionTimeline] audio_path 为空，已从 timeline.bgm 自动推断: {inferred_bgm_audio}")
            elif vpath:
                logger.info("[MovisAutoCaptionTimeline] audio_path 为空，且 timeline 无独立音频，回退使用视频音轨")
            else:
                logger.warn("[MovisAutoCaptionTimeline] 无可用音频来源（audio_path/video_path/timeline 均缺失）；将跳过自动字幕，不阻断流程")
            if pbar is not None:
                pbar.update(1)

            style = CaptionStylePreset._preset_dict(style_preset, True)
            if str(caption_json_param or "").strip():
                style.update(self.cp.parse_caption_params(caption_json_param))
            override_dict = {}
            if str(custom_override_json or "").strip():
                maybe_dict = self.cp.parse_caption_params(custom_override_json)
                if isinstance(maybe_dict, dict):
                    override_dict = dict(maybe_dict)
                    # 覆盖优先级最高（默认空字符串则不生效）
                    style.update(override_dict)
            if pbar is not None:
                pbar.update(1)

            if replace_existing:
                t["text_tracks"] = []
            elif "text_tracks" not in t or not isinstance(t.get("text_tracks"), list):
                t["text_tracks"] = []

            transcribe_cache = {}
            if pbar is not None:
                pbar.update(1)

            default_font_size = float(style.get("Fontsize", 54))
            default_font_family = str(style.get("Fontname", "Sans Serif"))
            default_color = str(style.get("highlight_color", "white"))
            default_opacity = 1.0

            override_video = self._override_resolved_path(override_dict, "force_video_path")
            override_audio = self._override_resolved_path(override_dict, "force_audio_path")
            if override_video:
                vpath = override_video
            if override_audio:
                explicit_audio = override_audio
                apath = override_audio

            override_audio_strategy = override_dict.get("audio_source_strategy", override_dict.get("audio_strategy", ""))
            input_audio_strategy = self._normalize_audio_strategy(audio_source_strategy)
            audio_source_strategy = self._normalize_audio_strategy(override_audio_strategy) if str(override_audio_strategy).strip() else input_audio_strategy

            input_layout_mode = self._normalize_layout_mode(layout_mode, auto_layout)
            layout_mode = self._normalize_layout_mode(override_dict.get("layout_mode", ""), auto_layout) if str(override_dict.get("layout_mode", "")).strip() else input_layout_mode
            if "auto_layout" in override_dict:
                try:
                    auto_layout = bool(override_dict.get("auto_layout"))
                except Exception:
                    pass

            safe_margin_ratio_input = self._safe_float(safe_margin_ratio, 0.0)
            if safe_margin_ratio_input is not None and safe_margin_ratio_input <= 0.0:
                safe_margin_ratio_input = None
            safe_margin_ratio = self._safe_float(override_dict.get("safe_margin_ratio", safe_margin_ratio_input), safe_margin_ratio_input)

            font_scale_input = self._safe_float(font_scale, 1.0)
            font_scale = self._safe_float(override_dict.get("font_scale", font_scale_input), font_scale_input)

            override_px = self._safe_float(override_dict.get("position_x", None), None)
            override_py = self._safe_float(override_dict.get("position_y", None), None)
            override_fs = self._safe_float(override_dict.get("font_size", override_dict.get("Fontsize", None)), None)
            if override_px is not None:
                position_x = max(0.0, min(1.0, override_px))
            if override_py is not None:
                position_y = max(0.0, min(1.0, override_py))
            if override_fs is not None:
                default_font_size = max(8.0, override_fs)

            layout_x, layout_y, layout_font_size = self._adapt_caption_layout(
                t,
                position_x,
                position_y,
                default_font_size,
                auto_layout=auto_layout,
                layout_mode=layout_mode,
                safe_margin_ratio=safe_margin_ratio,
                font_scale=font_scale,
            )
            logger.info(
                f"[MovisAutoCaptionTimeline] 字幕布局: auto_layout={bool(auto_layout)} "
                f"canvas={self._timeline_canvas_size(t)[0]:.0f}x{self._timeline_canvas_size(t)[1]:.0f} "
                f"layout_mode={layout_mode} pos=({layout_x:.3f},{layout_y:.3f}) fontsize={layout_font_size:.1f}"
            )
            logger.info(f"[MovisAutoCaptionTimeline] 音频策略: {audio_source_strategy}")

            count = 0
            used_audio_idx = set()
            allow_explicit = audio_source_strategy in ("auto", "mixed_priority", "explicit_only")
            allow_tracks = audio_source_strategy in ("auto", "mixed_priority", "audio_tracks_only")
            allow_bgm = audio_source_strategy in ("auto", "mixed_priority", "bgm_only")
            allow_video = audio_source_strategy in ("auto", "mixed_priority", "video_only")
            allow_fallback = audio_source_strategy in ("auto", "mixed_priority")

            if allow_explicit and explicit_audio:
                segs = self._transcribe_cached(explicit_audio, transcribe_cache)
                for seg in segs:
                    count += self._append_segment_text(
                        t, seg,
                        source_start=0.0,
                        timeline_start=0.0,
                        timeline_end=10**9,
                        start_offset=start_offset,
                        position_x=layout_x,
                        position_y=layout_y,
                        default_font_size=layout_font_size,
                        default_font_family=default_font_family,
                        default_color=default_color,
                        default_opacity=default_opacity,
                    )
                logger.info(f"[MovisAutoCaptionTimeline] 模式=显式全局音频，字幕条数={count}")
            elif allow_tracks and len(video_clips) > 0 and len(valid_audio_tracks) > 0:
                logger.info(
                    f"[MovisAutoCaptionTimeline] 模式=智能多轨匹配，"
                    f"video_clips={len(video_clips)}, audio_tracks={len(valid_audio_tracks)}"
                )
                for clip in video_clips:
                    chosen_audio = self._pick_audio_for_clip(clip, valid_audio_tracks, used_audio_idx)
                    segs = []
                    source_start = 0.0
                    source_desc = ""

                    if chosen_audio is not None:
                        used_audio_idx.add(chosen_audio["idx"])
                        segs = self._transcribe_cached(chosen_audio["path"], transcribe_cache)
                        source_start = float(chosen_audio.get("source_start", 0.0))
                        source_desc = f"audio_track#{chosen_audio['idx']}"
                    elif self._clip_can_use_source_audio(clip):
                        segs = self._transcribe_cached(clip.get("path"), transcribe_cache)
                        source_start = float(clip.get("source_start", 0.0))
                        source_desc = f"video_source#{clip['idx']}"
                    else:
                        logger.warn(
                            f"[MovisAutoCaptionTimeline] clip#{clip['idx']} 无可用音频来源"
                            f"（无匹配 audio_track，或源视频无音频流/禁用源音轨），跳过"
                        )
                        continue

                    clip_added = 0
                    for seg in segs:
                        clip_added += self._append_segment_text(
                            t, seg,
                            source_start=source_start,
                            timeline_start=float(clip["start"]),
                            timeline_end=float(clip["end"]),
                            start_offset=start_offset,
                            position_x=layout_x,
                            position_y=layout_y,
                            default_font_size=layout_font_size,
                            default_font_family=default_font_family,
                            default_color=default_color,
                            default_opacity=default_opacity,
                        )
                    count += clip_added
                    logger.info(
                        f"[MovisAutoCaptionTimeline] clip#{clip['idx']} matched={source_desc or 'none'} "
                        f"window=({clip['start']:.3f},{clip['end']:.3f}) added={clip_added}"
                    )
            elif allow_tracks and len(valid_audio_tracks) > 0:
                logger.info(
                    f"[MovisAutoCaptionTimeline] 模式=音轨全局识别，audio_tracks={len(valid_audio_tracks)}"
                )
                for tr in valid_audio_tracks:
                    segs = self._transcribe_cached(tr["path"], transcribe_cache)
                    tr_added = 0
                    for seg in segs:
                        tr_added += self._append_segment_text(
                            t, seg,
                            source_start=float(tr.get("source_start", 0.0)),
                            timeline_start=float(tr["start"]),
                            timeline_end=float(tr["end"]),
                            start_offset=start_offset,
                            position_x=layout_x,
                            position_y=layout_y,
                            default_font_size=layout_font_size,
                            default_font_family=default_font_family,
                            default_color=default_color,
                            default_opacity=default_opacity,
                        )
                    count += tr_added
                    logger.info(
                        f"[MovisAutoCaptionTimeline] audio_track#{tr['idx']} "
                        f"window=({tr['start']:.3f},{tr['end']:.3f}) added={tr_added}"
                    )
            elif allow_bgm and inferred_bgm_audio:
                window_duration = max(0.0, float(visible_end) - float(visible_start))
                if bgm_trim_to_video_length and window_duration > 0.01 and window_duration < 10**8:
                    segs = self._transcribe_window_cached(
                        inferred_bgm_audio,
                        window_start=bgm_source_start,
                        window_duration=window_duration,
                        cache=transcribe_cache,
                    )
                else:
                    segs = self._transcribe_cached(inferred_bgm_audio, transcribe_cache)
                for seg in segs:
                    count += self._append_segment_text(
                        t, seg,
                        source_start=bgm_source_start,
                        timeline_start=visible_start,
                        timeline_end=visible_end,
                        start_offset=start_offset,
                        position_x=layout_x,
                        position_y=layout_y,
                        default_font_size=layout_font_size,
                        default_font_family=default_font_family,
                        default_color=default_color,
                        default_opacity=default_opacity,
                    )
                logger.info(
                    f"[MovisAutoCaptionTimeline] 模式=BGM全局识别，字幕条数={count}, "
                    f"source_start={bgm_source_start:.3f}, window=({visible_start:.3f},{visible_end:.3f})"
                )
            elif allow_video and len(video_clips) > 0:
                logger.info(
                    f"[MovisAutoCaptionTimeline] 模式=视频源音轨识别，"
                    f"video_clips={len(video_clips)}, 可用源音轨clip={len(clips_with_audio_source)}"
                )
                for clip in video_clips:
                    if not self._clip_can_use_source_audio(clip):
                        logger.warn(f"[MovisAutoCaptionTimeline] clip#{clip['idx']} 源视频无音频流或已禁用源音轨，跳过")
                        continue
                    segs = self._transcribe_cached(clip.get("path"), transcribe_cache)
                    clip_added = 0
                    for seg in segs:
                        clip_added += self._append_segment_text(
                            t, seg,
                            source_start=float(clip.get("source_start", 0.0)),
                            timeline_start=float(clip["start"]),
                            timeline_end=float(clip["end"]),
                            start_offset=start_offset,
                            position_x=layout_x,
                            position_y=layout_y,
                            default_font_size=layout_font_size,
                            default_font_family=default_font_family,
                            default_color=default_color,
                            default_opacity=default_opacity,
                        )
                    count += clip_added
                    logger.info(
                        f"[MovisAutoCaptionTimeline] clip#{clip['idx']} matched=video_source#{clip['idx']} "
                        f"window=({clip['start']:.3f},{clip['end']:.3f}) added={clip_added}"
                    )
            elif allow_fallback and apath:
                segs = self._transcribe_cached(apath, transcribe_cache)
                for seg in segs:
                    count += self._append_segment_text(
                        t, seg,
                        source_start=0.0,
                        timeline_start=0.0,
                        timeline_end=10**9,
                        start_offset=start_offset,
                        position_x=layout_x,
                        position_y=layout_y,
                        default_font_size=layout_font_size,
                        default_font_family=default_font_family,
                        default_color=default_color,
                        default_opacity=default_opacity,
                    )
                logger.info(f"[MovisAutoCaptionTimeline] 模式=回退单源识别，字幕条数={count}")
            else:
                logger.warn("[MovisAutoCaptionTimeline] 缺少可识别音频来源，未生成字幕")
            if pbar is not None:
                pbar.update(1)

            drift_ms = -1.0
            sync_ok = False
            if vpath and apath:
                try:
                    vinfo = self.cp.get_media_info(vpath, kind="video")
                    ainfo = self.cp.get_media_info(apath, kind="audio")
                    drift_ms = abs(float(vinfo["duration"]) - float(ainfo["duration"])) * 1000.0
                    sync_ok = drift_ms <= float(sync_tolerance_ms)
                except Exception as e:
                    logger.warn(f"[MovisAutoCaptionTimeline] 同步检测失败，已跳过: {e}")
            else:
                logger.warn("[MovisAutoCaptionTimeline] 缺少视频或音频参考，未执行同步检测（sync_drift_ms=-1）")

            if count == 0:
                logger.warn("[MovisAutoCaptionTimeline] 本次未生成任何字幕（可能无有效音频/识别失败/语音为空）")
            if pbar is not None:
                pbar.update(1)

            return (t, count, drift_ms, sync_ok, json.dumps(style, ensure_ascii=False, indent=2))

# Node mappings
NODE_CLASS_MAPPINGS = {
    # Original nodes
    "VHS_VideoCombine": VideoCombine,
    "VHS_LoadVideo": LoadVideoUpload,
    "VHS_LoadVideoPath": LoadVideoPath,
    "VHS_LoadVideoFFmpeg": LoadVideoFFmpegUpload,
    "VHS_LoadVideoFFmpegPath": LoadVideoFFmpegPath,
    "VHS_LoadImagePath": LoadImagePath,
    "VHS_LoadImages": LoadImagesFromDirectoryUpload,
    "VHS_LoadImagesPath": LoadImagesFromDirectoryPath,
    "VHS_LoadAudio": LoadAudio,
    "VHS_LoadAudioUpload": LoadAudioUpload,
    "VHS_AudioToVHSAudio": AudioToVHSAudio,
    "VHS_VHSAudioToAudio": VHSAudioToAudio,
    "VHS_PruneOutputs": PruneOutputs,
    "VHS_BatchManager": BatchManager,
    "VHS_VideoInfo": VideoInfo,
    "VHS_VideoInfoSource": VideoInfoSource,
    "VHS_VideoInfoLoaded": VideoInfoLoaded,
    "VHS_SelectFilename": SelectFilename,
    # Batched Nodes
    "VHS_VAEEncodeBatched": VAEEncodeBatched,
    "VHS_VAEDecodeBatched": VAEDecodeBatched,
    # Latent and Image nodes
    "VHS_SplitLatents": SplitLatents,
    "VHS_SplitImages": SplitImages,
    "VHS_SplitMasks": SplitMasks,
    "VHS_MergeLatents": MergeLatents,
    "VHS_MergeImages": MergeImages,
    "VHS_MergeMasks": MergeMasks,
    "VHS_GetLatentCount": GetLatentCount,
    "VHS_GetImageCount": GetImageCount,
    "VHS_GetMaskCount": GetMaskCount,
    "VHS_DuplicateLatents": RepeatLatents,
    "VHS_DuplicateImages": RepeatImages,
    "VHS_DuplicateMasks": RepeatMasks,
    "VHS_SelectEveryNthLatent": SelectEveryNthLatent,
    "VHS_SelectEveryNthImage": SelectEveryNthImage,
    "VHS_SelectEveryNthMask": SelectEveryNthMask,
    "VHS_SelectLatents": SelectLatents,
    "VHS_SelectImages": SelectImages,
    "VHS_SelectMasks": SelectMasks,
    "VHS_Unbatch": Unbatch,
    "VHS_SelectLatest": SelectLatest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Original nodes
    "VHS_VideoCombine": "Video Combine 🎥🅥🅗🅢",
    "VHS_LoadVideo": "Load Video (Upload) 🎥🅥🅗🅢",
    "VHS_LoadVideoPath": "Load Video (Path) 🎥🅥🅗🅢",
    "VHS_LoadVideoFFmpeg": "Load Video FFmpeg (Upload) 🎥🅥🅗🅢",
    "VHS_LoadVideoFFmpegPath": "Load Video FFmpeg (Path) 🎥🅥🅗🅢",
    "VHS_LoadImagePath": "Load Image (Path) 🎥🅥🅗🅢",
    "VHS_LoadImages": "Load Images (Upload) 🎥🅥🅗🅢",
    "VHS_LoadImagesPath": "Load Images (Path) 🎥🅥🅗🅢",
    "VHS_LoadAudio": "Load Audio (Path)🎥🅥🅗🅢",
    "VHS_LoadAudioUpload": "Load Audio (Upload)🎥🅥🅗🅢",
    "VHS_AudioToVHSAudio": "Audio to legacy VHS_AUDIO🎥🅥🅗🅢",
    "VHS_VHSAudioToAudio": "Legacy VHS_AUDIO to Audio🎥🅥🅗🅢",
    "VHS_PruneOutputs": "Prune Outputs 🎥🅥🅗🅢",
    "VHS_BatchManager": "Meta Batch Manager 🎥🅥🅗🅢",
    "VHS_VideoInfo": "Video Info 🎥🅥🅗🅢",
    "VHS_VideoInfoSource": "Video Info (Source) 🎥🅥🅗🅢",
    "VHS_VideoInfoLoaded": "Video Info (Loaded) 🎥🅥🅗🅢",
    "VHS_SelectFilename": "Select Filename 🎥🅥🅗🅢",
    # Batched Nodes
    "VHS_VAEEncodeBatched": "VAE Encode Batched 🎥🅥🅗🅢",
    "VHS_VAEDecodeBatched": "VAE Decode Batched 🎥🅥🅗🅢",
    # Latent and Image nodes
    "VHS_SplitLatents": "Split Latents 🎥🅥🅗🅢",
    "VHS_SplitImages": "Split Images 🎥🅥🅗🅢",
    "VHS_SplitMasks": "Split Masks 🎥🅥🅗🅢",
    "VHS_MergeLatents": "Merge Latents 🎥🅥🅗🅢",
    "VHS_MergeImages": "Merge Images 🎥🅥🅗🅢",
    "VHS_MergeMasks": "Merge Masks 🎥🅥🅗🅢",
    "VHS_GetLatentCount": "Get Latent Count 🎥🅥🅗🅢",
    "VHS_GetImageCount": "Get Image Count 🎥🅥🅗🅢",
    "VHS_GetMaskCount": "Get Mask Count 🎥🅥🅗🅢",
    "VHS_DuplicateLatents": "Repeat Latents 🎥🅥🅗🅢",
    "VHS_DuplicateImages": "Repeat Images 🎥🅥🅗🅢",
    "VHS_DuplicateMasks": "Repeat Masks 🎥🅥🅗🅢",
    "VHS_SelectEveryNthLatent": "Select Every Nth Latent 🎥🅥🅗🅢",
    "VHS_SelectEveryNthImage": "Select Every Nth Image 🎥🅥🅗🅢",
    "VHS_SelectEveryNthMask": "Select Every Nth Mask 🎥🅥🅗🅢",
    "VHS_SelectLatents": "Select Latents 🎥🅥🅗🅢",
    "VHS_SelectImages": "Select Images 🎥🅥🅗🅢",
    "VHS_SelectMasks": "Select Masks 🎥🅥🅗🅢",
    "VHS_Unbatch":  "Unbatch 🎥🅥🅗🅢",
    "VHS_SelectLatest": "Select Latest 🎥🅥🅗🅢",
}

# Add custom nodes only if dependencies are available
if HAS_CUSTOM_FEATURES:
    if 'MergeAudio' in globals():
        NODE_CLASS_MAPPINGS.update({"VHS_MergeAudio": MergeAudio})
        NODE_DISPLAY_NAME_MAPPINGS.update({"VHS_MergeAudio": "Merge Audio 🎥🅥🅗🅢"})
    if 'VideoCaptions' in globals():
        NODE_CLASS_MAPPINGS.update({"VHS_VideoCaptions": VideoCaptions})
        NODE_DISPLAY_NAME_MAPPINGS.update({"VHS_VideoCaptions": "Video Captions 🎥🅥🅗🅢"})
    if 'VideoGentleCaptions' in globals():
        NODE_CLASS_MAPPINGS.update({"VHS_VideoGentleCaptions": VideoGentleCaptions})
        NODE_DISPLAY_NAME_MAPPINGS.update({"VHS_VideoGentleCaptions": "Video Gentle Captions 🎥🅥🅗🅢"})
    if 'CaptionStylePreset' in globals():
        NODE_CLASS_MAPPINGS.update({"VHS_CaptionStylePreset": CaptionStylePreset})
        NODE_DISPLAY_NAME_MAPPINGS.update({"VHS_CaptionStylePreset": "Caption Style Preset 🎨🎥🅥🅗🅢"})
    if 'VideoGentleCaptionsPro' in globals():
        NODE_CLASS_MAPPINGS.update({"VHS_VideoGentleCaptionsPro": VideoGentleCaptionsPro})
        NODE_DISPLAY_NAME_MAPPINGS.update({"VHS_VideoGentleCaptionsPro": "Video Gentle Captions Pro 🧠🎥🅥🅗🅢"})
    if 'MovisSubtitleFromSRT' in globals():
        NODE_CLASS_MAPPINGS.update({"VHS_MOVIS_SubtitleFromSRT": MovisSubtitleFromSRT})
        NODE_DISPLAY_NAME_MAPPINGS.update({"VHS_MOVIS_SubtitleFromSRT": "Movis Subtitle From SRT 🎬📝🎥🅥🅗🅢"})
    if 'MovisAutoCaptionTimeline' in globals():
        NODE_CLASS_MAPPINGS.update({"VHS_MOVIS_AutoCaptionTimeline": MovisAutoCaptionTimeline})
        NODE_DISPLAY_NAME_MAPPINGS.update({"VHS_MOVIS_AutoCaptionTimeline": "Movis Auto Caption Timeline 🎬🤖📝🎥🅥🅗🅢"})
    
    # Try to add video_ops nodes if available
    try:
        NODE_CLASS_MAPPINGS.update({
            "VHS_MOVIS_CreateTimeline": MovisCreateTimeline,
            "VHS_MOVIS_AddVideoTrack": MovisAddVideoTrack,
            "VHS_MOVIS_AddImageTrack": MovisAddImageTrack,
            "VHS_MOVIS_MergeTimeline": MovisMergeTimeline,
            "VHS_MOVIS_AddImageTrackSequence": MovisAddImageSequenceTrack,
            "VHS_MOVIS_AddImageMotionTrack": MovisAddImageMotionTrack,
            "VHS_MOVIS_AddVideoMotionTrack": MovisAddVideoMotionTrack,
            "VHS_MOVIS_AddAudioTrack": MovisAddAudioTrack,
            "VHS_MOVIS_SetBGM": MovisSetBGM,
            "VHS_MOVIS_AddTextOverlay": MovisAddTextOverlay,
            "VHS_MOVIS_RenderTimeline": MovisRenderTimeline,
            "VHS_MOVIS_Assemble": MovisAssemble,
            "VHS_MOVIS_UniversalStudio": MovisUniversalStudio,
            "VHS_MOVIS_SmartMerge": MovisSmartMerge,
            "VHS_MOVIS_SetGlobalTransition": MovisSetGlobalTransition,
            "VHS_MOVIS_SetClipTransition": MovisSetClipTransition,
            "VHS_MOVIS_AddClipFX": MovisAddClipFX,
            "VHS_MOVIS_ApplyFXPreset": MovisApplyFXPreset,
            "VHS_MOVIS_EnableLayeredFXEngine": MovisEnableLayeredFXEngine,
            "VHS_MOVIS_AddClipFXLayered": MovisAddClipFXLayered,
            "VHS_MOVIS_ApplyFXPresetLayered": MovisApplyFXPresetLayered,
            "VHS_MOVIS_GPUShaderRender": MovisGPUShaderRender,
            "VHS_MOVIS_SetClipShader": MovisSetClipShader,
            "VHS_MOVIS_TrimClip": MovisTrimClip,
            "VHS_MOVIS_DeleteClip": MovisDeleteClip,
            "VHS_MOVIS_TimelinePro": MovisTimelinePro,
            "VHS_MOVIS_COMPOSITE": CompositeMedia,
            "VHS_MOVIS_MultiVideo": CompositeMultiVideo,
            "VHS_MOVIS_BuildFX": MovisBuildFX,
            "VHS_MOVIS_BuildFXPreset": MovisBuildFXPreset,
            "VHS_MOVIS_ChainFX": MovisChainFX,
            "VHS_MOVIS_BuildShader": MovisBuildShader,
            "VHS_MOVIS_BuildAudio": MovisBuildAudio,
            "VHS_MOVIS_ChainAudio": MovisChainAudio,
            "VHS_MOVIS_BatchAddVideoTracks": MovisBatchAddVideoTracks,
            "VHS_MOVIS_QuickBuildTimeline": MovisQuickBuildTimeline,
        })
        NODE_DISPLAY_NAME_MAPPINGS.update({
            "VHS_MOVIS_CreateTimeline": "Movis Create Timeline 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_AddVideoTrack": "Movis Add Video Track 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_AddImageTrack": "Movis Add Image (Path) 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_MergeTimeline": "Movis Merge Timeline 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_AddImageTrackSequence": "Movis Add Image Sequence (Tensor) 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_AddImageMotionTrack": "Movis Add Image Motion 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_AddVideoMotionTrack": "Movis Add Video Motion 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_AddAudioTrack": "Movis Add Audio Track 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_SetBGM": "Movis Set BGM 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_AddTextOverlay": "Movis Add Text Overlay 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_RenderTimeline": "Movis Render Timeline 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_Assemble": "Movis Assemble (All-in-One) 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_UniversalStudio": "Movis Universal Studio (Pro) 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_SmartMerge": "Movis Smart Merge 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_SetGlobalTransition": "Movis Set Global Transition 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_SetClipTransition": "Movis Set Clip Transition 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_AddClipFX": "Movis Add Clip FX (Shader-like) 🎬✨🎥🅥🅗🅢",
            "VHS_MOVIS_ApplyFXPreset": "Movis Apply FX Preset 🎬⚡🎥🅥🅗🅢",
            "VHS_MOVIS_EnableLayeredFXEngine": "Movis Enable Layered FX Engine 🎬🧠🎥🅥🅗🅢",
            "VHS_MOVIS_AddClipFXLayered": "Movis Add Clip FX Layered 🎬✨🧠🎥🅥🅗🅢",
            "VHS_MOVIS_ApplyFXPresetLayered": "Movis Apply FX Preset Layered 🎬⚡🧠🎥🅥🅗🅢",
            "VHS_MOVIS_GPUShaderRender": "Movis GPU Shader Render (GLSL) 🎬🧠🎥🅥🅗🅢",
            "VHS_MOVIS_SetClipShader": "Movis Set Clip Shader (GLSL) 🎬🧠✨🎥🅥🅗🅢",
            "VHS_MOVIS_TrimClip": "Movis Trim Clip 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_DeleteClip": "Movis Delete Clip 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_TimelinePro": "Movis Timeline Pro 🎬🎥🅥🅗🅢",
            "VHS_MOVIS_COMPOSITE": "Movis Composite (Deprecated) 🎥🅥🅗🅢",
            "VHS_MOVIS_MultiVideo": "Movis Multi-Video (Deprecated) 🎥🅥🅗🅢",
            "VHS_MOVIS_BuildFX": "Movis Build FX 🎬✨🎥🅥🅗🅢",
            "VHS_MOVIS_BuildFXPreset": "Movis Build FX Preset 🎬⚡🎥🅥🅗🅢",
            "VHS_MOVIS_ChainFX": "Movis Chain FX 🎬🔗🎥🅥🅗🅢",
            "VHS_MOVIS_BuildShader": "Movis Build Shader 🎬🧠🔧🎥🅥🅗🅢",
            "VHS_MOVIS_BuildAudio": "Movis Build Audio 🎬🎵🎥🅥🅗🅢",
            "VHS_MOVIS_ChainAudio": "Movis Chain Audio 🎬🔗🎶🎥🅥🅗🅢",
            "VHS_MOVIS_BatchAddVideoTracks": "Movis Batch Add Video Tracks 🎬📦🎥🅥🅗🅢",
            "VHS_MOVIS_QuickBuildTimeline": "Movis Quick Build Timeline 🎬⚡🔒🎥🅥🅗🅢",
        })
    except NameError:
        print("Movis nodes not available")
    
    # Try to add depth generator and video preview nodes if available
    try:
        NODE_CLASS_MAPPINGS.update({
            "VHS_DepthFlow_Generator": DepthFlowGenerator,
            "VHS_VideoPreview": VideoPreview,
            "VHS_VideoPathInput": VideoPathInput,
            "VHS_VideoCacheManager": VideoCacheManager,
        })
        NODE_DISPLAY_NAME_MAPPINGS.update({
            "VHS_DepthFlow_Generator": "DepthFlow Generator 🌊🎥🅥🅗🅢",
            "VHS_VideoPreview": "Video Preview 🎬🎥🅥🅗🅢",
            "VHS_VideoPathInput": "Video Path Input 📁🎥🅥🅗🅢",
            "VHS_VideoCacheManager": "Video Cache Manager 🗑️🎥🅥🅗🅢",
        })
    except NameError as e:
        print(f"DepthFlow/Video nodes not available: {e}")