import torch
import numpy as np
import pyworld as pw
import pysptk
from fastdtw import fastdtw

def get_mgc(audio, sample_rate, frame_period, fft_size=512, mcep_size=60, alpha=0.65):
    if isinstance(audio, torch.Tensor):
        if audio.ndim > 1:
            audio = audio[0]

        audio = audio.numpy()

    _, sp, _ = pw.wav2world(
        audio.astype(np.double), fs=sample_rate, frame_period=frame_period, fft_size=fft_size)
    mgc = pysptk.sptk.mcep(
        sp, order=mcep_size, alpha=alpha, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)

    return mgc


def dB_distance(source, target):
    dB_const = 10.0/np.log(10.0)*np.sqrt(2.0)
    distance = source - target

    return dB_const*np.sqrt(np.inner(distance, distance))


def get_mcd(source, target, sample_rate, frame_period=5, cost_function=dB_distance):
    mgc_source = get_mgc(source, sample_rate, frame_period)
    mgc_target = get_mgc(target, sample_rate, frame_period)

    length = min(mgc_source.shape[0], mgc_target.shape[0])
    mgc_source = mgc_source[:length]
    mgc_target = mgc_target[:length]

    mcd, _ = fastdtw(mgc_source[..., 1:], mgc_target[..., 1:], dist=cost_function)
    mcd = mcd/length

    return mcd, length


def get_f0(audio, sample_rate, frame_period=5, method='dio'):
    if isinstance(audio, torch.Tensor):
        if audio.ndim > 1:
            audio = audio[0]

        audio = audio.numpy()

    hop_size = int(frame_period*sample_rate/1000)
    if method == 'dio':
        f0, _ = pw.dio(audio.astype(np.double), sample_rate, frame_period=frame_period)
    elif method == 'harvest':
        f0, _ = pw.harvest(audio.astype(np.double), sample_rate, frame_period=frame_period)
    elif method == 'swipe':
        f0 = pysptk.sptk.swipe(audio.astype(np.double), sample_rate, hopsize=hop_size)
    elif method == 'rapt':
        f0 = pysptk.sptk.rapt(audio.astype(np.double), sample_rate, hopsize=hop_size)
    else:
        raise ValueError(f'No such f0 extract method, {method}.')

    f0 = torch.from_numpy(f0)
    vuv = 1*(f0 != 0.0)

    return f0, vuv


def get_f0_rmse(source, target, sample_rate, frame_period=5, method='dio'):
    length = min(source.shape[-1], target.shape[-1])

    source_f0, source_v = get_f0(source[...,:length], sample_rate, frame_period, method)
    target_f0, target_v = get_f0(target[...,:length], sample_rate, frame_period, method)

    source_uv = 1 - source_v
    target_uv = 1 - target_v
    tp_mask = source_v*target_v

    length = tp_mask.sum().item()

    f0_rmse = 1200.0*torch.abs(torch.log2(target_f0 + target_uv) - torch.log2(source_f0 + source_uv))
    f0_rmse = tp_mask*f0_rmse
    f0_rmse = f0_rmse.sum()/length

    return f0_rmse.item(), length