import os
import torch
import torchaudio
from omegaconf import OmegaConf
import argparse
from pathlib import Path
from types import SimpleNamespace


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import AttributeDict


from valle.data import AudioTokenizer, tokenize_audio
from valle.data.collation import get_text_token_collater
from valle.models import get_model
from valle.data.hebrew_root_tokenizer import AlefBERTRootTokenizer, replace_chars


def load_model(checkpoint, device):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device)

    args = AttributeDict(checkpoint)
    model = get_model(args)

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.to(device)
    model.eval()

    text_tokens = args.text_tokens_path

    return model, text_tokens


def prepare_inference(checkpoint_path, args, prompt_audio):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    model, text_tokens = load_model(checkpoint_path, device)
    text_collater = get_text_token_collater(args.tokens_file)
    audio_tokenizer = AudioTokenizer(mbd=args.mbd)
    alef_bert_tokenizer = AlefBERTRootTokenizer(vocab_file=args.vocab_file)

    audio_prompts = []
    encoded_frames = tokenize_audio(audio_tokenizer, prompt_audio)
    audio_prompts.append(encoded_frames[0][0])
    audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1).to(device)

    return device, model, text_collater, audio_tokenizer, alef_bert_tokenizer, audio_prompts

def infer_texts(
    texts_with_filenames,
    output_dir,
    prompt_text,
    device,
    model,
    text_collater,
    audio_tokenizer,
    alef_bert_tokenizer,
    audio_prompts,
    top_k=50,
    temperature=1,
    args=None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for filename, text in texts_with_filenames:
            text_without_space = [replace_chars(f"{prompt_text} {text}").strip().replace(" ", "_")]
            tokens = alef_bert_tokenizer._tokenize(text_without_space)
            prompt_text_without_space = [replace_chars(f"{prompt_text}").strip().replace(" ", "_")]
            prompt_tokens = alef_bert_tokenizer._tokenize(prompt_text_without_space)

            text_tokens, text_tokens_lens = text_collater([tokens])
            _, enroll_x_lens = text_collater([prompt_tokens])

            encoded_frames = model.inference(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=top_k,
                temperature=temperature,
            )

            audio_path = Path(output_dir) / f"{filename}"

            if args.mbd:
                samples = audio_tokenizer.mbd_decode(encoded_frames.transpose(2, 1))
            else:
                samples = audio_tokenizer.decode([(encoded_frames.transpose(2, 1), None)])

            torchaudio.save(audio_path.as_posix(), samples[0].cpu().detach(), 24000)

def infer(checkpoint_path, output_dir, texts, prompt_text, prompt_audio, top_k=50, temperature=1, args=None):

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    model, text_tokens = load_model(checkpoint_path, device)
    text_collater = get_text_token_collater(args.tokens_file)

    audio_tokenizer = AudioTokenizer(mbd=args.mbd)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    alef_bert_tokenizer = AlefBERTRootTokenizer(vocab_file=args.vocab_file)
    texts = texts.split("|")

    audio_prompts = list()
    encoded_frames = tokenize_audio(audio_tokenizer, prompt_audio)
    audio_prompts.append(encoded_frames[0][0])
    audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
    audio_prompts = audio_prompts.to(device)

    for n, text in enumerate(texts):
        text_without_space = [replace_chars(f"{prompt_text} {text}").strip().replace(" ", "_")]
        tokens = alef_bert_tokenizer._tokenize(text_without_space)
        prompt_text_without_space = [replace_chars(f"{prompt_text}").strip().replace(" ", "_")]
        prompt_tokens = alef_bert_tokenizer._tokenize(prompt_text_without_space)

        text_tokens, text_tokens_lens = text_collater(
            [
                tokens
            ]
        )
        _, enroll_x_lens = text_collater(
            [
                prompt_tokens
            ]
        )

        # synthesis
        encoded_frames = model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=top_k,
            temperature=temperature,
        )

        audio_path = f"{output_dir}/sample_{n}.wav"

        if args.mbd:
            samples = audio_tokenizer.mbd_decode(
                encoded_frames.transpose(2, 1)
            )
        else:
            samples = audio_tokenizer.decode(
                [(encoded_frames.transpose(2, 1), None)]
            )

        torchaudio.save(audio_path, samples[0].cpu(), 24000)



def infer_texts_override(texts_with_filenames, speaker="omer"):
    """
    This function is a wrapper to allow overriding the infer_texts function
    with a custom implementation if needed.
    """
    args = { 
        "checkpoint": "checkpoint.pt",
        "output_dir": "out",
        "speaker_yaml": "speakers/speakers.yaml",
        "vocab_file": "tokenizer/vocab.txt",
        "tokens_file": "tokenizer/unique_words_tokens_all.k2symbols",
        "mbd": False,
        "top_k": 40,
    }
    args = SimpleNamespace(**args)  # Convert dict to SimpleNamespace for compatibility    
    spekers_yaml_path = args.speaker_yaml
    speaker_yaml = OmegaConf.load(spekers_yaml_path)
    try:
        speaker = speaker_yaml[speaker]
    except Exception:
        print(f"Invalid speaker {speaker}. Should be defined at speakers.yaml.")

    audio_prompt = str(Path(spekers_yaml_path).parent / speaker["audio-prompt"])

    device, model, text_collater, audio_tokenizer, alef_bert_tokenizer, audio_prompts = prepare_inference(
        args.checkpoint, args, audio_prompt
    )


    return infer_texts(
        texts_with_filenames,
        "out",
        speaker["text-prompt"],
        device,
        model,
        text_collater,
        audio_tokenizer,
        alef_bert_tokenizer,
        audio_prompts,
        top_k=40,
        temperature=1,
        args=args,
    )


