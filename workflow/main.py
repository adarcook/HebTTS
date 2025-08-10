from tokens_split import process_hebrew_text
from infer import infer_texts_override
import subprocess
from fs_utils import empty_folder
import argparse

output_foler = "out"


def wav_to_mp3(wav_path, mp3_path, bitrate="192k"):
    """
    Convert WAV file to MP3 using ffmpeg.

    :param wav_path: Path to input WAV file
    :param mp3_path: Path to output MP3 file
    :param bitrate: Bitrate for MP3 (default 192k)
    """
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output file if exists
        "-i", wav_path,
        "-b:a", bitrate,
        mp3_path
    ]

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Process some text and generate audio.")
    parser.add_argument('--text', type=str, required=True, help="Text to process")
    
    args = parser.parse_args()
    
    print(f"Parameter value: {args.text}")

    empty_folder(output_foler)
    chunks = process_hebrew_text(args.text, output_dir=output_foler)
    print(chunks)
    infer_texts_override(chunks)
    # Define your paths
    txt_file = f"{output_foler}/0.txt"
    output_file = f"{output_foler}/0_full.wav"

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", txt_file,
        "-c", "copy",
        output_file
    ]

    # Run command
    subprocess.run(cmd, check=True)
    wav_to_mp3(output_file, f"{output_foler}/0_full.mp3")


if __name__ == "__main__":
    main()