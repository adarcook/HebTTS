from tokens_split import process_hebrew_text
from infer import infer_texts_override
import subprocess
from fs_utils import empty_folder
import argparse

output_foler = "out"


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


if __name__ == "__main__":
    main()