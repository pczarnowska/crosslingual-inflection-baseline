'''
Decode model
'''
import argparse
from functools import partial

import torch
import regex as re

from model import decode_beam_search, decode_greedy
from util import maybe_mkdir, get_device, BOS, EOS, UNK_IDX, BOS_IDX, EOS_IDX
from tqdm import tqdm
tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')


def setup_inference(max_len=30, beam_size=3, decode='greedy', nonorm=False):
    decode_fn = None
    if decode == 'greedy':
        decode_fn = partial(decode_greedy, max_len=max_len)
    elif decode == 'beam':
        decode_fn = partial(
            decode_beam_search,
            max_len=max_len,
            nb_beam=beam_size,
            norm=not nonorm)
    return decode_fn



def encode(model, lemma, tags, device):
    tag_shift = model.src_vocab_size - len(model.attr_c2i)

    src = []
    src.append(model.src_c2i[BOS])
    for char in lemma:
        src.append(model.src_c2i.get(char, UNK_IDX))
    src.append(model.src_c2i[EOS])

    attr = [0] * (len(model.attr_c2i) + 1)
    for tag in tags:
        if tag in model.attr_c2i:
            attr_idx = model.attr_c2i[tag] - tag_shift
        else:
            attr_idx = -1
        if attr[attr_idx] == 0:
            attr[attr_idx] = model.attr_c2i.get(tag, 0)

    return (torch.tensor(src, device=device).view(len(src), 1),
            torch.tensor(attr, device=device).view(1, len(attr)))


def reinflect_form(model, device, decode_fn, tag, pos, lemma):
    lemma_feats = {"V": "NFIN",
                    "ADJ": "SG+MASC+NOM",
                    "V.PTCP": "SG+MASC+NOM",
                    "N": "SG+NOM"}

    trg_i2c = {i: c for c, i in model.trg_c2i.items()}
    decode_trg = lambda seq: [trg_i2c[i] for i in seq]

    if pos in lemma_feats and all([y.lower() in tag.lower() for y in lemma_feats[pos].split("+")]):
        if pos == "V.PTCP" and len(lemma) > 0 and lemma[-1] == "ć":
            pass
        else:
            return lemma

    tag_splits = tag.split(';')
    src = encode(model, lemma, tag_splits, device)
    pred, _ = decode_fn(model, src)
    pred_out = ''.join(decode_trg(pred))
    return pred_out


def reinflect(model_source, lemmas, tags, poses, multi=False):
    decode_fn = setup_inference()
    device = get_device()
    model = torch.load(open(model_source, mode='rb'), map_location=device)
    model = model.to(device)

    forms = []

    for i, (lemma, tag) in enumerate(zip(lemmas, tags)):
        pos = poses[i]
        if multi or isinstance(tag, list): # we need to inflect many times
            preds = set()
            for t in tag:
                pred_out = reinflect_form(model, device, decode_fn, t, pos, lemma)
                preds.add(pred_out)
            forms.append(list(preds))
        else:
            pred_out = reinflect_form(model, device, decode_fn, tag, pos, lemma)
            forms.append(f"{pred_out}")
    return forms


