import datetime
import os
import movis as mv
import pandas as pd
import numpy as np
from notifier import notify
from PIL import Image

import folder_paths

m_output_folder = folder_paths.get_output_directory()



def scale_to_cover(target_size, current_size):
    tw, th = target_size
    cw, ch = current_size
    width_ratio = tw / cw
    height_ratio = th / ch
    return max(width_ratio, height_ratio)

def tensor_to_int(tensor, bits):
    #TODO: investigate benefit of rounding by adding 0.5 before clip/cast
    tensor = tensor.cpu().numpy() * (2**bits-1)
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)


##TODO - fullly port movis basic ops， not just this simple use

class CompositeMedia:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "images": ("IMAGE", ),
                "audios": ("STRING", {"validate": "is_file"},),
                "is_vertical":("BOOLEAN",{"default":True}),
                "output_file_prefix": ("STRING", {"default": "composite_output_"}),
                "notify_all": ("BOOLEAN",{"default":True})
            },
            "optional": {
                "bgm": ("STRING",{"default":""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_file_path",)
    CATEGORY = "Video Helper Suite 🎥VHS"
    FUNCTION = "composite_media"
    OUTPUT_NODE = True

    @classmethod
    def composite_media(self, images, audios, bgm, is_vertical, output_file_prefix, notify_all):
        
        _datetime = datetime.datetime.now().strftime("%Y%m%d")
        _datetime = _datetime + datetime.datetime.now().strftime("%H%M%S%f")

        image_paths = []
        filename = f"composite_media_{_datetime}"
        counter = 0
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None

            file = f"{filename}_{counter:05}_.png"
            file_path = os.path.join(folder_paths.get_output_directory(), file)
            img.savef(file_path,compress_level=4)

            image_paths.append(file_path)
            counter += 1
        
        audio_paths = audios.split(",")

        print(audio_paths)
        print("=======")
        print(image_paths)
        # TODO -不一定音频需要
        # Ensure that the number of images and audios match
        # if len(image_paths) != len(audio_paths):
        #     raise ValueError("The number of images and audios must be the same.")

        size = mv.layer.Image(image_paths[0]).size
        timeline = pd.DataFrame()
        for i, (image_path, audio_path) in enumerate(zip(image_paths, audio_paths)):
            timeline = timeline.append({
                'duration': mv.layer.media.Audio(audio_path).duration, 'image': f'{image_path}',
                'title': '', 'title_position': 'center','audio': audio_path}, ignore_index=True)

        # Create transitions list with 0.5 for each transition
        transitions = [0.5] * (len(images) - 1)

        total_time = timeline['duration'].sum() + sum(transitions) +1
        print(f"total time {total_time}")
        scene = mv.layer.Composition(size=size, duration=total_time)
        scene.add_layer(mv.layer.Rectangle(size=size, color='#202020', duration=scene.duration), name='bg')

        print("----scene-----")
        time = 0.
        prev_transitions = [0.] + transitions
        next_transitions = transitions + [0.]
        for (i, row), t_prev, t_next in zip(timeline.iterrows(), prev_transitions, next_transitions):
            T = row['duration']
            image_layer = mv.layer.Image(row['image'], duration=T + t_prev + t_next)
            image = scene.add_layer(image_layer, offset=time - t_prev)
            
            if row['audio'] != "":
                scene.add_layer(mv.layer.media.Audio(row['audio']),offset=time - t_prev)

            if i == 0:
                # Add fadein effect
                image.opacity.enable_motion().extend(keyframes=[0.0, 0.5], values=[0.0, 1.0])
            elif i == len(timeline) - 1:
                # Add fadeout effect
                t = image.duration
                image.opacity.enable_motion().extend(keyframes=[t - 0.5, t], values=[1.0, 0.0])

            # kwargs_dict = {
            #     'center': {'position': (size[0] / 2, size[1] / 2), 'origin_point': mv.Direction.CENTER},
            # #     'bottom_right': {'position': (size[0] - 50, size[1] - 50), 'origin_point': mv.Direction.BOTTOM_RIGHT}}
            # position = kwargs_dict[row['title_position']]['position']
            # origin_point = kwargs_dict[row['title_position']]['origin_point']
            # scene.add_layer(
            #     make_logo(row['title'], duration=T, font_size=64),
            #     offset=time, position=position, origin_point=origin_point)

            if 0 < i:
                # Add fade effects
                image.opacity.enable_motion().extend(keyframes=[0.0, t_prev], values=[0.0, 1.0])

            # Add scale effect
            values = [1.15, 1.25] if i % 2 == 0 else [1.25, 1.15]
            image.scale.enable_motion().extend(
                keyframes=[0.0, T + t_prev + t_next], values=values)
            time += (T + t_next)
            
        # Get the absolute path of the output file
        _datetime = datetime.datetime.now().strftime("%Y%m%d")
        _datetime = _datetime + datetime.datetime.now().strftime("%H%M%S%f")
        
        
        file_name = output_file_prefix + _datetime +".mp4"
        file_pth = os.path.join(folder_paths.get_output_directory(),file_name)
    
        if bgm != "":
            scene.add_layer(mv.layer.media.Audio(bgm),start_time=0,end_time=scene.duration)
            
        scene.write_video(file_pth)

        if notify_all:
            notify.notifyAll(file_pth,"video_composite")
            
        return {"ui":{"video_path":file_pth},"result": (file_pth,)}
    

# class MakeMedia:
#     @classmethod
#     def INPUT_TYPES(self):
#         return {
#             "required": {
#                 "movis_composite": ("MOVIS_COMPOSITE",),
#                 "output_file_prefix": ("STRING", {"default": "composite_output_"}),
#                 "notify_all": ("BOOLEAN",{"default":True})
#             },
#             "optional": {
#                 "bgm": ("STRING",{"default":""}),
#             },
#         }

#     RETURN_TYPES = ("STRING",)
#     RETURN_NAMES = ("video_file_path",)
#     CATEGORY = "Video Helper Suite 🎥VHS"
#     FUNCTION = "make"
#     OUTPUT_NODE = True

#     @classmethod
#     def make(self,bgm,output_file_prefix,movis_composite,notify_all):

#         # Get the absolute path of the output file
#         _datetime = datetime.datetime.now().strftime("%Y%m%d")
#         _datetime = _datetime + datetime.datetime.now().strftime("%H%M%S%f")
        
        
#         file_name = output_file_prefix + _datetime +".mp4"
#         file_pth = os.path.join(folder_paths.get_output_directory(),file_name)
    
#         if bgm != "":
#             movis_composite.add_layer(mv.layer.media.Audio(bgm),start_time=0,end_time=movis_composite.duration)
            
#         movis_composite.write_video(file_pth)

#         if notify_all:
#             notify.notifyAll(file_pth,"video_composite")
            
#         return {"ui":{"video_path":file_pth},"result": (file_pth,)}
    
