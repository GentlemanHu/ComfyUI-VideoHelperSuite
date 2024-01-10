import json
import os
# from shortGPT.audio import audio_utils
# from shortGPT.audio.audio_duration import get_asset_duration
# from shortGPT.config.asset_db import AssetDatabase
# from shortGPT.config.languages import Language
# from shortGPT.editing_framework.editing_engine import EditingEngine, EditingStep
# from shortGPT.editing_utils import captions


# def _editAndRenderShort(ori_path,is_vertical=True,language=Language.ARABIC):

#     outputPath = "rendered_video.mp4"
#     if not (os.path.exists(outputPath)):
#         print("Rendering short: Starting automated editing...")

#         timed_captions = _timeCaptions(ori_path,is_vertical)

#         videoEditor = EditingEngine()
#         videoEditor.addEditingStep(EditingStep.ADD_VOICEOVER_AUDIO, {
#                                     'url': ori_path})
        
#         videoEditor.addEditingStep(EditingStep.ADD_SUBSCRIBE_ANIMATION, {'url': AssetDatabase.get_asset_link('subscribe animation')})

#         # if (self._db_background_music_url):
#         #     videoEditor.addEditingStep(EditingStep.ADD_BACKGROUND_MUSIC, {'url': self._db_background_music_url,
#         #                                                                     'loop_background_music': self._db_voiceover_duration,
#         #                                                                     "volume_percentage": 0.08})
#         _, vid_length = get_asset_duration(ori_path)
#         videoEditor.addEditingStep(EditingStep.ADD_BACKGROUND_VIDEO, {'url': ori_path,
#                                                                             'set_time_start': 0,
#                                                                             'set_time_end': vid_length})
        
#         videoEditor.addEditingStep(EditingStep.CROP_1920x1080,{'url': ori_path})
        
#         if (is_vertical):
#             caption_type = EditingStep.ADD_CAPTION_SHORT_ARABIC if language == Language.ARABIC.value else EditingStep.ADD_CAPTION_SHORT
#         else:
#             caption_type = EditingStep.ADD_CAPTION_LANDSCAPE_ARABIC if language == Language.ARABIC.value else EditingStep.ADD_CAPTION_LANDSCAPE

#         for (t1, t2), text in timed_captions:
#             videoEditor.addEditingStep(caption_type, {'text': text.upper(),
#                                                         'set_time_start': t1,
#                                                         'set_time_end': t2})

#         videoEditor.renderVideo(outputPath,None)




# def _timeCaptions(ori_audio_path,is_vertical=True):
#     whisper_analysis = audio_utils.audioToText(ori_audio_path)
#     max_len = 15

#     if(is_vertical):
#         max_len = 30

#     result = captions.getCaptionsWithTime(
#         whisper_analysis, maxCaptionSize=max_len)
    
#     print(result)
#     return result



# _editAndRenderShort("/Users/huyilin/Downloads/tk/AnimateDiff_00001-audio.mp4")
import io

json_string = """
{
  "fontsize": 100,
  "font": "Lemon-Bold",
  "color": "white",
  "stroke_width": 4,
  "stroke_color": "black",
  "method": "caption"
}
"""

string_io = io.StringIO(json_string)
template_dict = json.load(string_io)

# Add the text, start time, and end time to the dictionary
template_dict["text"] = "NNjdklafjkldjflakd".upper()
template_dict["set_time_start"] = 0
template_dict["set_time_end"] = 99

print(template_dict)