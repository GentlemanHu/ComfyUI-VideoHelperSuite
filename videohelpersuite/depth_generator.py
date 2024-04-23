import os
import subprocess
from datetime import datetime
from PIL import Image
import numpy as np

import folder_paths
# Assuming folder_paths is available from ComfyUI or a similar module
# Replace with the appropriate import or implementation if needed
# from comfyui import folder_paths 

class DepthFlowGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "images": ("IMAGE",),  # Input is a list of images
                "output_path": ("STRING", {"default": "depthflow_output"}),
            },
            "optional": {
                "isVertical": ("BOOLEAN", {"default": True}),
                "quality": ("INT", {"default": 100, "min": 0, "max": 100}),
                "time": ("FLOAT", {"default": 8.0, "min": 0.1}),
                "fps": ("INT", {"default": 25, "min": 1}),
                "ssaa": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0}),
                "format": (
                    "STRING",
                    {"default": "mp4", "choices": ["mp4", "mkv"]},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Video Result Path",)
    OUTPUT_NODE = True
    FUNCTION = "run_depthflow"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    def run_depthflow(
        self,
        images,  # Input is a list of images (we'll take the first one)
        output_path,
        isVertical=True,
        quality=100,
        time=8,
        fps=25,
        ssaa=0.25,
        format="mp4",
    ):
        """
        Runs Depthflow on the first image in the input list and returns the output video path.
        """

        # Take the first image from the list
        image = images[0]

        _datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"depth_composite_media_{_datetime}"

        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        file = f"{filename}_.png"
        file_path = os.path.join(folder_paths.get_output_directory(), file)
        img.save(file_path, compress_level=4)

        input_path = file_path  # Use the saved image path as input

        # Generate a unique filename based on timestamp and custom suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_suffix = "_depthflow" 
        filename = f"{timestamp}{custom_suffix}.{format}"

        # Get the output directory and construct the full output path
        output_dir = os.path.join(folder_paths.get_output_directory(), "depthflow_videos") 
        os.makedirs(output_dir, exist_ok=True) 
        output_path = os.path.join(output_dir, filename)

        # Construct the depthflow command
        if isVertical:
            width = 1080
            height = 1920
        else:
            width = 1920
            height = 1080

        command = [
            "xvfb-run",  # Uncomment if running on a headless system
            "depthflow",
            "input",
            "-i",
            input_path,
            "main",
            "-w",
            str(width),
            "-h",
            str(height),
            "-q",
            str(quality),
            "-t",
            str(time),
            "-f",
            str(fps),
            "-s",
            str(ssaa),
            "-r",
            "-o",
            output_path,
            "--format",
            format,
        ]
        os.environ["SHADERFLOW_BACKEND"] = "headless"

        # Run the depthflow command
        subprocess.run(command)

        final_path = output_path + f".{format}" 
        return {"ui":{"video_path":final_path},"result": (final_path,)}  # Return the absolute path to the output video