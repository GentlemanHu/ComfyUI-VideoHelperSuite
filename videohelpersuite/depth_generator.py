import os
import subprocess

def run_depthflow(input_path, output_path, isVertical:bool=True, quality=100, time=6, format="mp4"):
  """Runs Depthflow on an image and returns the absolute path to the output video file.

  Args:
    input_path: The path to the input image.
    output_path: The path to the output video.
    aspect_ratio: The aspect ratio of the output video.
    quality: The quality of the output video (0-100).
    time: The length of the output video in seconds.
    format: The format of the output video.

  Returns:
    The absolute path to the output video file.
  """

  if isVertical:
      width = 1080
      height = 1920
  else:
      width = 1920
      height = 1080

  command = [
      "xvfb-run",
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
      "-r",
      "-o",
      output_path,
      "--format",
      format,
  ]
  os.environ["SHADERFLOW_BACKEND"] = "headless"
  subprocess.run(command)

  return os.path.abspath(output_path)


print(run_depthflow("/Users/gentlemanhu/Documents/Codes/AI/photo_2023-10-28 23.36.17.jpeg","/Users/gentlemanhu/Downloads/Telegram Desktop/cache/depth_999.mp4"))

