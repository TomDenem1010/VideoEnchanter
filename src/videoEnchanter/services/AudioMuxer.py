import shutil
import subprocess
from pathlib import Path


class AudioMuxer:

    def _build_mux_commands(self, ffmpeg_executable: str, source_video_path: str, processed_video_path: str, output_path: str):
        common_prefix = [
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
            "-shortest",
            "-movflags",
            "+faststart"
        ]

        return [
            common_prefix + [
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ar",
                "48000",
                output_path
            ],
            common_prefix + [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "24",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ar",
                "48000",
                output_path
            ]
        ]

    def _run_command(self, command):
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )

    def merge_audio(self, source_video_path: str, processed_video_path: str, output_path: str):
        output_file = Path(output_path)

        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError:
            shutil.move(processed_video_path, output_path)
            return

        ffmpeg_executable = get_ffmpeg_exe()
        last_error = None

        for command in self._build_mux_commands(
            ffmpeg_executable,
            source_video_path,
            processed_video_path,
            str(output_file)
        ):
            result = self._run_command(command)

            if result.returncode == 0:
                return

            last_error = result.stderr.strip() or result.stdout.strip() or "Ismeretlen ffmpeg hiba"

            if output_file.exists():
                output_file.unlink()

        raise RuntimeError(f"Nem sikerült visszatenni a hangsávot: {last_error}")