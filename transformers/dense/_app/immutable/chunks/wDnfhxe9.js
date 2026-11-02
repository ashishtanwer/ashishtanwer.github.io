const e=`# Whisper base (Radford et al. 2022), openai/whisper-base, as plain PyTorch: how the decoder's predictions become a
# transcript. The network itself is the Transformer Explainer's Speech drawer (pytorch/speech.py) and its 30 s
# front end the Tokenization Explainer's Whisper drawer (pytorch/tokenization_whisper.py); here \`step\` stands for
# it: step(ids) → the logits for the next token, or step(list of id lists) → one row per beam. The decoder starts
# from special tokens (start of transcript, language, task), its first prediction is the language, Whisper's rules
# remove tokens the format forbids (timestamps come in pairs and never go back), greedy decoding keeps the likeliest
# token and beam search the five likeliest sequences, and audio longer than 30 s is read window by window, each one
# starting where the last complete segment ended. The rules and the beam follow transformers' WhisperForConditional-
# Generation.generate, so its outputs can be compared token for token.
# Each "# %% [id]" section is one step of the Whisper tab's diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

INF = float("inf")


@dataclass
class Config:
    sample_rate: int = 16_000
    n_fft: int = 400  # 25 ms windows
    hop: int = 160  # a log-mel frame every 10 ms
    window_frames: int = 3_000  # the encoder always reads 30 s
    input_stride: int = 2  # frames per encoder position, the spacing of the timestamp tokens (20 ms)
    seconds_per_timestamp: float = 0.02
    vocab_size: int = 51_865
    eot: int = 50257  # <|endoftext|>
    sot: int = 50258  # <|startoftranscript|>
    # Language tokens follow <|startoftranscript|> in this order: <|en|> = 50259, <|zh|> = 50260, ...
    languages: str = (
        "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt "
        "la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc "
        "ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su"
    )
    translate: int = 50358
    transcribe: int = 50359
    no_timestamps: int = 50363
    timestamp_begin: int = 50364  # <|0.00|>; the last, 51864, is <|30.00|>
    max_initial_timestamp: int = 50  # a window's first timestamp is at most <|1.00|>
    begin_suppress: tuple = (220, 50257)  # " " and <|endoftext|> cannot come first
    # generation_config.suppress_tokens: symbols and the special tokens that never appear in a transcript
    suppress: tuple = (
        1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542,
        873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667,
        6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618,
        16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470,
        36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362,
    )
    max_length: int = 448  # prompt + generated tokens
    beams: int = 5
    length_penalty: float = 1.0


# %% [log-mel] Log-mel frames of a whole clip
def log_mel(audio, filters, c: Config):
    """audio (samples,) at 16 kHz → (80, samples // 160). For long-form decoding the clip is not padded to 30 s:
    every 10 ms frame of it, log10 floored 8 below the loudest cell of the whole clip, scaled to about −1 … 1.
    filters: the (201, 80) mel filterbank of the 30 s front end."""
    window = torch.hann_window(c.n_fft)
    power = torch.stft(audio, c.n_fft, c.hop, window=window, return_complex=True)[..., :-1].abs() ** 2
    log = (filters.T @ power).clamp(min=1e-10).log10()
    return (torch.maximum(log, log.max() - 8.0) + 4.0) / 4.0


# %% [prompt] The prompt: start of transcript, language, task
def prompt(c: Config, language="en", task="transcribe", timestamps=True):
    """<|startoftranscript|> <|en|> <|transcribe|>, then <|notimestamps|> unless the text is to be bracketed by
    timestamp tokens. <|translate|> asks for English text whatever the language spoken."""
    ids = [c.sot, c.sot + 1 + c.languages.split().index(language), getattr(c, task)]
    return ids if timestamps else ids + [c.no_timestamps]


# %% [language] Language detection: the first prediction
def detect_language(logits, c: Config):
    """logits (vocab,): the decoder's prediction right after <|startoftranscript|>. Only the language tokens are
    kept, so its softmax over them is a distribution over the 99 languages; the likeliest one is the detection."""
    codes = c.languages.split()
    p = logits[c.sot + 1: c.sot + 1 + len(codes)].float().softmax(-1)
    return codes[int(p.argmax())], p


# %% [rules] Whisper's rules: what may come next
def rules(generated, scores, c: Config, timestamps=True):
    """generated: the ids written after the prompt; scores (vocab,): logits (greedy) or log-probabilities (beam).
    Returns the scores with every forbidden token at −∞, in transformers' order."""
    s = scores.clone()
    if not generated:  # the very first token can be neither a space nor the end
        s[list(c.begin_suppress)] = -INF
    s[list(c.suppress)] = -INF
    if not timestamps:
        return s
    tb = c.timestamp_begin
    s[c.no_timestamps] = -INF
    last = len(generated) >= 1 and generated[-1] >= tb
    penultimate = len(generated) < 2 or generated[-2] >= tb
    if last:
        if penultimate:  # a segment just closed (or the first timestamp): text comes next
            s[tb:] = -INF
        else:  # text then a timestamp: the next is a timestamp too, or the end
            s[: c.eot] = -INF
    stamps = [i for i in generated if i >= tb]
    if stamps:  # time never goes back; a closing timestamp may repeat the one before it
        s[tb: stamps[-1] if last and not penultimate else stamps[-1] + 1] = -INF
    if not generated:  # start with a timestamp, at most <|1.00|>
        s[:tb] = -INF
        s[tb + c.max_initial_timestamp + 1:] = -INF
    log_p = F.log_softmax(s.float(), dim=-1)
    if log_p[tb:].logsumexp(-1) > log_p[:tb].max():  # all timestamps together beat every text token
        s[:tb] = -INF
    return s


# %% [greedy] Greedy decoding
def greedy(step, start, c: Config, timestamps=True):
    """Writes the likeliest allowed token until <|endoftext|> or 448 tokens. Returns the ids and, per step, the
    scores after the rules."""
    ids, scores = list(start), []
    while len(ids) < c.max_length:
        s = rules(ids[len(start):], step(ids).float(), c, timestamps)
        scores.append(s)
        ids.append(int(s.argmax()))
        if ids[-1] == c.eot:
            break
    return ids, scores


# %% [beam] Beam search: the five likeliest sequences
def beam_search(step, start, c: Config, timestamps=True):
    """Keeps \`beams\` running sequences scored by their summed log-probabilities. Every step extends each one by
    every allowed token and keeps the 2 × beams best extensions; those ending in <|endoftext|> (among the first
    \`beams\`) become finished hypotheses scored by sum / length^penalty, the rest are the next running beams. It stops
    when no running beam can still beat the worst of \`beams\` finished ones. Returns the best finished sequence and
    the search: per step the running beams as (parent beam, token, score) and the finished ones."""
    n, keep, length = c.beams, 2 * c.beams, len(start)
    running = [list(start)] * n
    running_scores = torch.tensor([0.0] + [-1e9] * (n - 1))
    done_ids, done_scores, done = [list(start)] * n, torch.full((n,), -1e9), torch.zeros(n, dtype=torch.bool)
    open_heuristic, history = True, []
    first_n = torch.arange(keep) < n
    while True:
        log_p = F.log_softmax(step(running).float(), dim=-1)  # (beams, vocab)
        log_p = torch.stack([rules(r[length:], row, c, timestamps) for r, row in zip(running, log_p)])
        total = (log_p + running_scores[:, None]).reshape(-1)
        top_scores, top = total.topk(keep)
        parents, tokens = (top // log_p.shape[-1]).tolist(), (top % log_p.shape[-1]).tolist()
        candidates = [running[b] + [t] for b, t in zip(parents, tokens)]
        hits = torch.tensor([t == c.eot or len(s) >= c.max_length for t, s in zip(tokens, candidates)])
        # running beams for the next step: the best extensions that did not finish
        live = top_scores + hits.float() * -1e9
        chosen = live.topk(n).indices.tolist()
        # finished hypotheses: merge the new ones (only among the first \`beams\`) with the old, keep the best
        new = top_scores / ((len(running[0]) + 1 - length) ** c.length_penalty)
        new = new + (not open_heuristic) * -1e9 + (~(hits & first_n)).float() * -1e9
        merged = torch.cat([done_scores, new])
        best = merged.topk(n).indices.tolist()
        all_ids, all_done = done_ids + candidates, torch.cat([done, hits & first_n])
        done_ids, done_scores, done = [all_ids[i] for i in best], merged[best], all_done[best]
        history.append({"running": [(parents[i], tokens[i], float(live[i])) for i in chosen],
                        "finished": [(done_ids[k][length:], float(done_scores[k])) for k in range(n) if done[k]]})
        running, running_scores = [candidates[i] for i in chosen], live[chosen]
        # can a running beam still beat the worst finished one? (transformers' early-stop heuristic)
        best_possible = running_scores[0] / ((len(running[0]) - length) ** c.length_penalty)
        worst = torch.where(done, done_scores.min(), torch.tensor(-1e9))
        open_heuristic = open_heuristic and bool((best_possible > worst).any())
        if not open_heuristic or bool(hits.all()):
            return done_ids[0], history


# %% [segments] Timestamp tokens → segments, and where the next window starts
def segments(ids, offset, frames, c: Config):
    """ids: one window's output after the prompt (without <|endoftext|>); offset: the window's start in seconds;
    frames: how many of its 3,000 frames are audio. Each pair of adjacent timestamps closes one segment and opens
    the next. If the output ends with a lone timestamp, the window had no more speech and the next one starts after
    it; otherwise the last segment may be cut off mid-word, so it is dropped and the next window starts at the last
    pair. Returns [(start s, end s, ids)] and how many frames to move on."""
    tb = c.timestamp_begin
    stamp = [i >= tb for i in ids]
    single_ending = stamp[-2:] == [False, True]
    pairs = [k + 1 for k in range(len(ids) - 1) if stamp[k] and stamp[k + 1]]
    seconds = lambda i: offset + (i - tb) * c.seconds_per_timestamp  # noqa: E731
    if not pairs:  # one segment for the whole window, ending at its last timestamp if it has one
        last = [i for i in ids if i >= tb]
        end = last[-1] - tb if last and last[-1] != tb else int(frames * c.hop / c.sample_rate
                                                                / c.seconds_per_timestamp)
        return [(offset, offset + end * c.seconds_per_timestamp, ids)], frames
    cuts = pairs + [len(ids)] if single_ending else pairs[:-1] + [pairs[-1] + 1]
    found, begin = [], 0
    for k, cut in enumerate(cuts):
        piece = ids[begin:cut]
        closing = piece[-1] if k < len(cuts) - 1 or single_ending else piece[-2]
        found.append((seconds(piece[0]), seconds(closing), piece))
        begin = cut
    advance = frames if single_ending else (ids[begin - 2] - tb) * c.input_stride
    return found, advance


# %% [long-form] Long-form transcription: 30 s windows
def transcribe(features, decode, c: Config):
    """features (80, frames): the clip's log-mel frames; decode(window (80, 3,000)) → the window's ids after the
    prompt, <|endoftext|> removed. Each window is the next 30 s from \`seek\` (zero-padded at the end of the clip);
    its segments are kept and \`seek\` moves to where the last complete one ended."""
    seek, total, windows = 0, features.shape[-1], []
    while seek < total:
        frames = min(c.window_frames, total - seek)
        chunk = features[:, seek: seek + frames]
        ids = decode(F.pad(chunk, (0, c.window_frames - chunk.shape[-1])))
        found, advance = segments(ids, seek * c.seconds_per_timestamp / c.input_stride, frames, c)
        windows.append({"seek": seek, "frames": frames, "ids": ids, "segments": found, "advance": advance})
        seek += advance
    return windows
`;export{e as default};
