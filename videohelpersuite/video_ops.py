import datetime
import os
import movis as mv
import pandas as pd

from notifier import notify

import folder_paths

m_output_folder = folder_paths.get_output_directory()


##TODO - fullly port movis basic ops， not just this simple use

class CompositeMedia:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image_1": ("STRING", {"validate": "is_file"},),
                "image_2": ("STRING", {"validate": "is_file"},),
                "image_3": ("STRING", {"validate": "is_file"},),
                "image_4": ("STRING", {"validate": "is_file"},),
                "audio_1": ("STRING", {"validate": "is_file"},),
                "audio_2": ("STRING", {"validate": "is_file"},),
                "audio_3": ("STRING", {"validate": "is_file"},),
                "audio_4": ("STRING", {"validate": "is_file"},),
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
    def composite_media(self,image_1,image_2,image_3,image_4,audio_1,audio_2,audio_3,audio_4,bgm,is_vertical,output_file_prefix,notify_all):
        size =  (1080,1920) if is_vertical else (1920, 1080) 
        timeline = pd.DataFrame([
            {
                'duration': mv.layer.media.Audio(audio_1).duration, 'image': f'{image_1}',
                'title': '', 'title_position': 'center'},
            {
                'duration': mv.layer.media.Audio(audio_2).duration, 'image': f'{image_2}',
                'title': '', 'title_position': 'bottom_right'},
            {
                'duration': mv.layer.media.Audio(audio_3).duration, 'image': f'{image_3}',
                'title': '', 'title_position': 'bottom_right'},
                    {
                'duration': mv.layer.media.Audio(audio_4).duration, 'image': f'{image_4}',
                'title': '', 'title_position': 'bottom_right'}
        ])
        transitions = [2.0, 2.0]

        total_time = timeline['duration'].sum() + sum(transitions) + 1.0
        print(f"total time {total_time}")
        scene = mv.layer.Composition(size=size, duration=total_time)
        scene.add_layer(mv.layer.Rectangle(size=size, color='#202020', duration=scene.duration), name='bg')

        print("----scene-----")
        time = 0.
        prev_transitions = [0.] + transitions
        next_transitions = transitions + [0.]
        for (i, row), t_prev, t_next in zip(timeline.iterrows(), prev_transitions, next_transitions):
            T = row['duration']
            image = scene.add_layer(
                mv.layer.Image(row['image'], duration=T + t_prev + t_next), offset=time - t_prev)
            if i == 0:
                # Add fadein effect
                image.opacity.enable_motion().extend(keyframes=[0.0, 1.0], values=[0.0, 1.0])
            elif i == len(timeline) - 1:
                # Add fadeout effect
                t = image.duration
                image.opacity.enable_motion().extend(keyframes=[t - 1.0, t], values=[1.0, 0.0])

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
    
