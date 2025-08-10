import re
import os
import glob

MAX_WORDS = 10  # max words per chunk for TTS

def split_hebrew_text(text):
    """Split Hebrew text into short TTS-friendly chunks."""
    text = re.sub(r'\s+', ' ', text.strip())
    text = text.replace('!', '')  # Remove exclamation marks only
    sentences = re.split(r'(?<=[.?!])\s+', text)  # keep end punctuation

    final_chunks = []
    for sentence in sentences:
        sub_parts = re.split(r',\s*', sentence)
        for part in sub_parts:
            part = part.strip()
            if not part:
                continue
            words = part.split()
            if len(words) > MAX_WORDS:
                start = 0
                while start < len(words):
                    chunk = " ".join(words[start:start + MAX_WORDS]).strip()
                    if chunk:
                        final_chunks.append(chunk)
                    start += MAX_WORDS
            else:
                final_chunks.append(part)
    return final_chunks

def get_next_index(output_dir):
    """Return next free index based on existing N.wav files."""
    os.makedirs(output_dir, exist_ok=True)
    existing = glob.glob(os.path.join(output_dir, "*.wav"))
    indices = []
    for f in existing:
        base = os.path.splitext(os.path.basename(f))[0]
        if base.isdigit():
            indices.append(int(base))
    return max(indices) + 1 if indices else 0

def save_ffmpeg_txt(tuples_list, output_dir, txt_index):
    """Create TXT file listing WAV files for ffmpeg concat."""
    txt_filename = os.path.join(output_dir, f"{txt_index}.txt")
    with open(txt_filename, "w", encoding="utf-8") as f:
        for filename, _ in tuples_list:
            f.write(f"file '{filename}'\n")
    return txt_filename

def process_hebrew_text(hebrew_text, output_dir="output", reset_index=True, create_txt=True):
    """
    Returns list of tuples: [('0.wav', 'text1'), ('1.wav', 'text2'), ...]
    - reset_index=True  -> start from 0
    - reset_index=False -> continue from existing wav files in output_dir
    - create_txt=True   -> generate ffmpeg-friendly .txt file
    """
    chunks = split_hebrew_text(hebrew_text)
    start_index = 0 if reset_index else get_next_index(output_dir)
    tuples_list = [(f"{i}.wav", chunk) for i, chunk in enumerate(chunks, start=start_index)]

    if create_txt:
        save_ffmpeg_txt(tuples_list, output_dir, start_index)

    return tuples_list

# Example usage
if __name__ == "__main__":
    text = "הייתי עכשיו בבריכה עם הילדים. הם מאוד נהנו וגם אני. אשמח ללכת איתם שוב."
    result = process_hebrew_text(text, output_dir="output", reset_index=True, create_txt=True)
    print(result)
