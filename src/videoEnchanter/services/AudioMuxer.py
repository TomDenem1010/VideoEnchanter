import shutil
import subprocess
from pathlib import Path


class AudioMuxer:

    def merge_audio(self, source_video_path: str, processed_video_path: str, output_path: str):
        output_file = Path(output_path)

        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError:
            shutil.move(processed_video_path, output_path)
            return

        ffmpeg_executable = get_ffmpeg_exe()

        command = [
            ffmpeg_executable,
            "-y",
            "-i",
            processed_video_path,
            "-i",
            source_video_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_file)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            return

        if output_file.exists():
            output_file.unlink()

        shutil.move(processed_video_path, output_path)