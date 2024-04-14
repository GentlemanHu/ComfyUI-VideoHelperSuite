

import datetime
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import ffmpeg

import torch
import folder_paths

import stable_whisper as whisper

global probe

class GentleCaption:
    def __init__(self) -> None:
        
        self.model = whisper.load_model("base")
        self.output_dir = os.path.join(folder_paths.get_output_directory(), 'captioned')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        pass

    def srt_create(self,whisper_model,word_dict:dict, path: str = None, filename: str=None,**kwargs) -> bool:

        if str is None or "":
            path = self.output_dir

        _datetime = datetime.datetime.now().strftime("%Y%m%d")
        _datetime = _datetime + datetime.datetime.now().strftime("%H%M%S%f")
        
        srt_path = f"{path}{os.sep}"
        srt_filename = f"{srt_path}{_datetime}.srt"
        ass_filename = f"{srt_path}{_datetime}.ass"

        absolute_srt_path = Path(srt_filename).absolute()
        absolute_ass_path = Path(ass_filename).absolute()

        transcribe = whisper_model.transcribe(
            filename, regroup=True, fp16=torch.cuda.is_available())
        transcribe.split_by_gap(0.5).split_by_length(
            38).merge_by_gap(0.15, max_words=2)
        transcribe.to_srt_vtt(str(absolute_srt_path), word_level=True)
        transcribe.to_ass(str(absolute_ass_path),**word_dict)
        return ass_filename



    def get_info(filename: str, kind: str):

        if kind == 'video':
            result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_entries', 'stream=width,height,duration', '-of', 'csv=p=0', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise RuntimeError(f"Error getting video info: {result.stderr.decode('utf-8')}")
            width, height, duration = result.stdout.decode('utf-8').split(',')
            return {'width': int(width), 'height': int(height), 'duration': float(duration)}

        elif kind == 'audio':
            result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=duration', '-of', 'csv=p=0', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise RuntimeError(f"Error getting audio info: {result.stderr.decode('utf-8')}")
            duration = result.stdout.decode('utf-8')
            return {'duration': float(duration)}

        else:
            raise ValueError(f"Invalid kind: {kind}")
            


    def convert_time(self,time_in_seconds):
        """
        Converts time in seconds to a string in the format "hh:mm:ss.mmm".

        Args:
            time_in_seconds (float): The time in seconds to be converted.

        Returns:
            str: The time in the format "hh:mm:ss.mmm".
        """
        hours = int(time_in_seconds // 3600)
        minutes = int((time_in_seconds % 3600) // 60)
        seconds = int(time_in_seconds % 60)
        milliseconds = int((time_in_seconds - int(time_in_seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def makeVideo(self,bg_video_path:str,bg_audio_path:str,output_filename:str,extra_para:dict):
    
        print(extra_para)
        video_info = self.get_info(kind= 'video',filename= bg_video_path)
        video_duration = int(round(video_info.get('duration'), 0))
        
        
        if bg_audio_path is None or bg_audio_path == "":
            bg_audio_path = bg_video_path
            
        print(bg_audio_path)
        
        ass_path = self.srt_create(self.model,extra_para,path=self.output_dir,filename=bg_audio_path)
        m_type ='audio' if bg_video_path != bg_audio_path  else 'video'
        audio_info = self.get_info(bg_audio_path,m_type)
        audio_duration = int(round(audio_info.get('duration'), 0))
        
        print(f"video_duration {video_duration} ---- audio_duration {audio_duration}")
        ss = random.randint(0, abs(video_duration-audio_duration))
        audio_duration = self.convert_time(audio_duration)
        
        
        _datetime = datetime.datetime.now().strftime("%Y%m%d")
        _datetime = _datetime + datetime.datetime.now().strftime("%H%M%S%f")

        output_path = os.path.join(folder_paths.get_output_directory(), output_filename+f"_{_datetime}.mp4")
        
        args = [
            "ffmpeg",
        "-ss", str(ss),
        "-t", str(audio_duration),
        "-i", bg_video_path,
        "-i", bg_audio_path,
        "-map", "0:v",
        "-map", "1:a",
        # TODO - 可以控制不同比例尺寸
        "-filter:v", f"crop=ih/16*9:ih, scale=w=1080:h=1920:flags=lanczos, gblur=sigma=2, ass={ass_path}",
        "-c:v", "libx264",
        "-crf", "23",
        "-c:a", "aac",
        "-ac", "2",
        "-b:a", "192K",
        f"{output_path}",
        "-y",
        "-threads", f"{multiprocessing.cpu_count()}"
        ]

        subprocess.run(args, check=True)
        
        return output_path
