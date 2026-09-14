#!/usr/bin/env python3
"""
01_preprocesamiento.py (versión extendida con extracción de features)

Este script:
 - lee ficheros PCAP/PCAPNG por sitio,
 - construye vectores de dirección (1 / -1) y timestamps,
 - Calcula un conjunto muy amplio de features: bursts, inter-arrival times (IAT), CUMUL, TAM, n-grams,
  entropía, ratios, espectro, distribuciones, etc.
 - guarda un CSV por sitio con las features (los arrays complejos se serializan como JSON).

Requisitos:
  pip install scapy numpy pandas tqdm

Uso:
  python3 01_preprocesamiento.py \
  --pcap-root "/data/Datos/" \
  --outdir "~/modelos/output" \
  --logdir "~/modelos/logs" \
  --max-packets 0

Salida:
 - CSV por sitio en outdir (cada celda con arrays complejos serializados en JSON)
 - Logs por pcap en logdir

"""
from pathlib import Path                    # Para manejo de rutas de archivos
import argparse                             # Para CLI / argumentos de línea de comandos
import json                                 # Para serializar arrays complejos a JSON
from collections import Counter             # Contador de elementos (frecuencias)
import numpy as np                          # Operaciones numéricas vectoriales
import pandas as pd                         # Manejo de dataframes / CSV
from scapy.all import PcapReader, IP, TCP   # type: ignore
from tqdm import tqdm                       # type: ignore
import math                                 # Funciones matemáticas básicas
from src.utils.paths import data_path, result_path

# Estadísticas avanzadas: skew, kurtosis (asimetría y curtosis)
try:
    from scipy.stats import skew, kurtosis
except Exception:
    skew = None
    kurtosis = None

# ---------------------------------------
# Utilities / utilidades pequeñas
# ---------------------------------------

def shannon_entropy(probs):
    """
    Calcula la entropía de Shannon a partir de probabilidades.
    - probs: lista de probabilidades de cada evento.
    - Ignora probabilidades 0.
    """
    probs = np.array([p for p in probs if p > 0.0])
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())

def ngrams_counts(seq, n):
    """
    Cuenta ocurrencias de n-grams en una secuencia.
    - seq: lista de elementos
    - n: tamaño del n-gram
    """
    cnt = Counter()
    for i in range(len(seq) - n + 1):
        tup = tuple(seq[i:i+n])
        cnt[tup] += 1
    return cnt


# ------------------------------
# Feature extraction (completa)
# ------------------------------
def extract_features_from_vector(vec, times, sizes=None, interp_points=100, tam_intervals=(1,2,3)):
    """
    Función principal de extracción de features.

    Entrada:
    - vec: lista de 1/-1 (dirección de paquetes)
    - times: timestamps de paquetes
    - sizes: tamaño en bytes de cada paquete (opcional)
    - interp_points: número de puntos para interpolación de CUMUL
    - tam_intervals: ventanas para TAM

    Salida:
    - diccionario con ~130 features.
    """
    feat = {}
    n = len(vec)
    feat['vector_len'] = n  # 3 -- longitud del vector de dirección

    if sizes is None:
        sizes = [0]*n

    # --------------------------
    # Features del vector sizes
    # --------------------------
    if sizes is not None and len(sizes) > 0:
        sizes_arr = np.array(sizes)
        
        # Estadísticas básicas
        feat['sizes_mean'] = float(np.mean(sizes_arr))
        feat['sizes_std'] = float(np.std(sizes_arr))
        feat['sizes_min'] = int(np.min(sizes_arr))
        feat['sizes_max'] = int(np.max(sizes_arr))
        feat['sizes_median'] = float(np.median(sizes_arr))
        feat['sizes_sum'] = int(np.sum(sizes_arr))
        
        # Percentiles
        p25, p50, p75, p90 = np.percentile(sizes_arr, [25, 50, 75, 90])
        feat['sizes_p25'] = float(p25)
        feat['sizes_p50'] = float(p50)
        feat['sizes_p75'] = float(p75)
        feat['sizes_p90'] = float(p90)
        
        # Skewness y kurtosis
        feat['sizes_skew'] = float(skew(sizes_arr)) # type: ignore
        feat['sizes_kurtosis'] = float(kurtosis(sizes_arr)) # type: ignore
        
        # Histograma personalizado
        hist_bins = [0, 500, 1000, 1500, np.inf]  # puedes ajustar bins
        hist_counts, _ = np.histogram(sizes_arr, bins=hist_bins)
        feat['sizes_hist_1_500'] = int(hist_counts[0])
        feat['sizes_hist_501_1000'] = int(hist_counts[1])
        feat['sizes_hist_1001_1500'] = int(hist_counts[2])
        feat['sizes_hist_gt1500'] = int(hist_counts[3])
    else:
        # Si el vector está vacío
        for key in ['sizes_mean','sizes_std','sizes_min','sizes_max','sizes_median',
                    'sizes_sum','sizes_p25','sizes_p50','sizes_p75','sizes_p90',
                    'sizes_skew','sizes_kurtosis','sizes_hist_1_500','sizes_hist_501_1000',
                    'sizes_hist_1001_1500','sizes_hist_gt1500']:
            feat[key] = None

    # --------------------------
    # Features del vector times
    # --------------------------
    if times is not None and len(times) > 1:
        times_arr = np.array(times)
        feat['times_duration'] = float(times_arr[-1] - times_arr[0])
        feat['times_mean'] = float(np.mean(times_arr))
        feat['times_std'] = float(np.std(times_arr))
        feat['times_min'] = float(np.min(times_arr))
        feat['times_max'] = float(np.max(times_arr))
        feat['times_median'] = float(np.median(times_arr))
    else:
        for key in ['times_duration','times_mean','times_std','times_min','times_max','times_median']:
            feat[key] = None

    # Variables básicas (1..5)
    arr = np.array(vec) if n>0 else np.array([])
    total_out = int((arr == 1).sum()) if n>0 else 0
    total_in  = int((arr == -1).sum()) if n>0 else 0
    feat['total_in'] = total_in   # 4
    feat['total_out'] = total_out # 5

    # Si no hay paquetes
    if n == 0:
        # rellenar defaults
        defaults = {
            'burst_max': 0, 'burst_count': 0, 'burst_mean': 0.0, 'burst_std': 0.0,
            'bursts_lengths': [], 'bursts_durations': [],
            'iat_min': None, 'iat_max': None, 'iat_mean': None, 'iat_median': None, 'iat_var': None, 'iat_percentiles': [],
            'duration': 0.0, 'burst_durations_mean': 0.0,
            'packets_per_window': {}, 'cumul_in': [], 'cumul_out': [], 'cumul_interp_in': [], 'cumul_interp_out': [],
            'TAM': {}, 'burst_histogram': {}, 'ngrams_top': {}, 'entropy': 0.0, 'concentration_25pct': 0.0,
            'DoA': 0.0
        }
        feat.update(defaults)
        return feat

    # -------------------------
    # Ráfagas (bursts) 6..8.10
    # -------------------------
    bursts_lengths = []
    bursts_durations = []
    cur = arr[0]
    cur_len = 1
    cur_start_time = times[0] if times else 0.0
    for i in range(1, n):
        if arr[i] == cur:
            cur_len += 1
        else:
            bursts_lengths.append(cur_len)
            bursts_durations.append(times[i-1] - cur_start_time if times else 0.0)
            cur = arr[i]
            cur_len = 1
            cur_start_time = times[i]
    bursts_lengths.append(cur_len)
    bursts_durations.append(times[-1] - cur_start_time if times else 0.0)

    bursts_lengths = np.array(bursts_lengths, dtype=int)
    bursts_durations = np.array(bursts_durations, dtype=float)

    # Calcula estadísticas de bursts:
    feat['burst_max'] = int(bursts_lengths.max()) if bursts_lengths.size>0 else 0  # 6
    feat['burst_count'] = int(bursts_lengths.size)  # 7
    # percentiles de tamaño de ráfagas (25, 50, 75, 90) — separados en variables
    if bursts_lengths.size > 0:
        p25, p50, p75, p90 = np.percentile(bursts_lengths, [25, 50, 75, 90]).tolist()
        burst_size_p25 = float(p25)
        burst_size_p50 = float(p50)
        burst_size_p75 = float(p75)
        burst_size_p90 = float(p90)
    else:
        # No hay ráfagas: variables individuales a None y lista vacía
        burst_size_p25 = burst_size_p50 = burst_size_p75 = burst_size_p90 = None
    # Guardar en feat (variables separadas + la lista original para compatibilidad)
    feat['burst_size_p25'] = burst_size_p25
    feat['burst_size_p50'] = burst_size_p50
    feat['burst_size_p75'] = burst_size_p75
    feat['burst_size_p90'] = burst_size_p90
    feat['burst_mean'] = float(bursts_lengths.mean()) if bursts_lengths.size>0 else 0.0
    feat['burst_std'] = float(bursts_lengths.std(ddof=0)) if bursts_lengths.size>0 else 0.0

    bd = bursts_durations  # ndarray float

    # Defaults
    feat['burst_durations_count'] = int(bd.size) if bd is not None else 0

    if bd.size == 0:
        feat['burst_durations_sum'] = 0.0
        feat['burst_durations_mean'] = 0.0
        feat['burst_durations_median'] = None
        feat['burst_durations_min'] = None
        feat['burst_durations_max'] = None
        feat['burst_durations_std'] = None
        feat['burst_durations_var'] = None
        feat['burst_durations_cv'] = None
        feat['burst_durations_iqr'] = None
        feat['burst_durations_p25'] = None
        feat['burst_durations_p50'] = None
        feat['burst_durations_p75'] = None
        feat['burst_durations_p90'] = None
        feat['burst_durations_skew'] = None
        feat['burst_durations_kurtosis'] = None
        feat['burst_durations_entropy'] = None
        feat['burst_durations_prop_gt_1s'] = 0.0
        feat['burst_durations_prop_gt_2s'] = 0.0
        feat['burst_durations_prop_gt_5s'] = 0.0
        feat['burst_durations_longest_ratio'] = None
    else:
        # básicos
        s_sum = float(bd.sum())
        s_mean = float(bd.mean())
        s_median = float(np.median(bd))
        s_min = float(bd.min())
        s_max = float(bd.max())
        s_var = float(bd.var(ddof=0))
        s_std = float(np.std(bd, ddof=0))
        # coeficiente de variación (std/mean) - cuidado con mean ~ 0
        s_cv = float(s_std / s_mean) if s_mean != 0 else None

        # percentiles e IQR
        p25, p50, p75, p90 = np.percentile(bd, [25, 50, 75, 90]).tolist()
        iqr = float(p75 - p25)

        # skew / kurtosis (si scipy disponible)
        try:
            s_skew = float(skew(bd)) if skew is not None else None
        except Exception:
            s_skew = None
        try:
            s_kurt = float(kurtosis(bd)) if kurtosis is not None else None
        except Exception:
            s_kurt = None

        # Entropía: construimos histograma y calculamos Shannon sobre la distribución de bins
        # usamos bins automáticos pero con un mínimo para estabilidad
        try:
            nbins = min(max(5, int(np.sqrt(bd.size))), 50)  # heurística: sqrt(N), cap 50
            hist, edges = np.histogram(bd, bins=nbins, density=False)
            probs = hist.astype(float) / (hist.sum() if hist.sum() > 0 else 1.0)
            # eliminar ceros y calcular entropía
            probs_nonzero = probs[probs > 0]
            bd_entropy = float(-(probs_nonzero * np.log2(probs_nonzero)).sum()) if probs_nonzero.size>0 else 0.0
        except Exception:
            bd_entropy = None

        # Proporciones de ráfagas "largas" (umbrales en segundos)
        prop_gt_1 = float((bd > 1.0).sum()) / float(bd.size)
        prop_gt_2 = float((bd > 2.0).sum()) / float(bd.size)
        prop_gt_5 = float((bd > 5.0).sum()) / float(bd.size)

        # Relación del burst más largo respecto al total de duración de ráfagas
        longest_ratio = float(s_max) / float(s_sum) if s_sum > 0 else None

        # Guardar en feat (nombres claros)
        feat['burst_durations_sum'] = s_sum
        feat['burst_durations_mean'] = s_mean
        feat['burst_durations_median'] = s_median
        feat['burst_durations_min'] = s_min
        feat['burst_durations_max'] = s_max
        feat['burst_durations_std'] = s_std
        feat['burst_durations_var'] = s_var
        feat['burst_durations_cv'] = s_cv

        feat['burst_durations_p25'] = float(p25)
        feat['burst_durations_p50'] = float(p50)
        feat['burst_durations_p75'] = float(p75)
        feat['burst_durations_p90'] = float(p90)
        feat['burst_durations_iqr'] = iqr

        feat['burst_durations_skew'] = s_skew
        feat['burst_durations_kurtosis'] = s_kurt
        feat['burst_durations_entropy'] = bd_entropy

        feat['burst_durations_prop_gt_1s'] = prop_gt_1
        feat['burst_durations_prop_gt_2s'] = prop_gt_2
        feat['burst_durations_prop_gt_5s'] = prop_gt_5

        feat['burst_durations_longest_ratio'] = longest_ratio

    # histograma bins 8.4
    bins_map = {'1':0, '2':0, '3-5':0, '6-10':0, 'gt10':0}
    for bl in bursts_lengths:
        if bl == 1:
            bins_map['1'] += 1
        elif bl == 2:
            bins_map['2'] += 1
        elif 3 <= bl <= 5:
            bins_map['3-5'] += 1
        elif 6 <= bl <= 10:
            bins_map['6-10'] += 1
        else:
            bins_map['gt10'] += 1
    # Guardar cada bin como feature separada
    feat['burst_histogram_1'] = bins_map['1']
    feat['burst_histogram_2'] = bins_map['2']
    feat['burst_histogram_3_5'] = bins_map['3-5']
    feat['burst_histogram_6_10'] = bins_map['6-10']
    feat['burst_histogram_gt10'] = bins_map['gt10']

    # in/out bursts stats 8.5..8.10
    in_bursts = []
    out_bursts = []
    cur = arr[0]
    cur_len = 1
    for i in range(1, n):
        if arr[i] == cur:
            cur_len += 1
        else:
            if cur == -1:
                in_bursts.append(cur_len)
            else:
                out_bursts.append(cur_len)
            cur = arr[i]
            cur_len = 1
    # última
    if cur == -1:
        in_bursts.append(cur_len)
    else:
        out_bursts.append(cur_len)

    feat['num_bursts_in'] = int(len(in_bursts))
    feat['num_bursts_out'] = int(len(out_bursts))
    feat['mean_burst_size_in'] = float(np.mean(in_bursts)) if in_bursts else 0.0
    feat['mean_burst_size_out'] = float(np.mean(out_bursts)) if out_bursts else 0.0
    feat['max_burst_in'] = int(np.max(in_bursts)) if in_bursts else 0
    feat['max_burst_out'] = int(np.max(out_bursts)) if out_bursts else 0

    # -------------------------
    # IAT (9..10)
    # -------------------------
    if len(times) >= 2:
        iats = np.diff(np.array(times, dtype=float))
        feat['iat_min'] = float(iats.min())
        feat['iat_max'] = float(iats.max())
        feat['iat_mean'] = float(iats.mean())
        feat['iat_median'] = float(np.median(iats))
        feat['iat_var'] = float(iats.var(ddof=0))
        if iats.size > 0:
            iat_p25, iat_p50, iat_p75, iat_p90 = np.percentile(iats, [25, 50, 75, 90]).tolist()
            iat_size_p25 = float(iat_p25)
            iat_size_p50 = float(iat_p50)
            iat_size_p75 = float(iat_p75)
            iat_size_p90 = float(iat_p90)
        else:
            # No hay ráfagas: variables individuales a None y lista vacía
            iat_size_p25 = iat_size_p50 = iat_size_p75 = iat_size_p90 = None
        # Guardar en feat (variables separadas + la lista original para compatibilidad)
        feat['iat_size_p25'] = iat_size_p25
        feat['iat_size_p50'] = iat_size_p50
        feat['iat_size_p75'] = iat_size_p75
        feat['iat_size_p90'] = iat_size_p90

        # mean/std IAT por entrada/salida
        in_iats = [iats[j] for j in range(len(iats)) if arr[j]==-1]
        out_iats = [iats[j] for j in range(len(iats)) if arr[j]==1]
        feat['iat_mean_in'] = float(np.mean(in_iats)) if in_iats else None
        feat['iat_std_in'] = float(np.std(in_iats)) if in_iats else None
        feat['iat_mean_out'] = float(np.mean(out_iats)) if out_iats else None
        feat['iat_std_out'] = float(np.std(out_iats)) if out_iats else None
    else:
        feat['iat_min'] = feat['iat_max'] = feat['iat_mean'] = feat['iat_median'] = feat['iat_var'] = None
        feat['iat_size_p25'] = feat['iat_size_p50'] = feat['iat_size_p75'] = feat['iat_size_p90'] = None
        feat['iat_mean_in'] = feat['iat_std_in'] = feat['iat_mean_out'] = feat['iat_std_out'] = None

    # -------------------------
    # Tiempo y duración (11..13)
    # -------------------------
    if times:
        duration = float(times[-1] - times[0])
    else:
        duration = 0.0
    feat['duration'] = duration
    feat['DoA'] = duration
    # time to first response (heurística: primer paquete entrante)
    first_in_idx = next((i for i,v in enumerate(arr) if v==-1), None)
    feat['time_to_first_response'] = float(times[first_in_idx]) if (first_in_idx is not None) else None

    # -------------------------
    # 14. Paquetes por segundo en diferentes ventanas de tiempo (features individuales)
    # -------------------------
    start_t = times[0] if times else 0.0
    windows = [1, 2, 5, 10]

    for w in windows:
        # Si no hay duración o no hay paquetes, rellenar con defaults
        if duration == 0 or len(times) == 0:
            feat[f'pps_w{w}_total_packets'] = 0
            feat[f'pps_w{w}_mean'] = 0.0
            feat[f'pps_w{w}_max'] = 0.0
            feat[f'pps_w{w}_min'] = 0.0
            feat[f'pps_w{w}_std'] = 0.0
            feat[f'pps_w{w}_burst_count'] = 0
            feat[f'pps_w{w}_in_out'] = None
            continue

        # número de intervalos de longitud w
        nbins = int(np.ceil(duration / w))
        counts = [0] * nbins           # paquetes por bin (bin length = w segundos)
        counts_in = [0] * nbins        # paquetes entrantes por bin
        counts_out = [0] * nbins       # paquetes salientes por bin

        for t, d in zip(times, arr):
            idx = int((t - start_t) // w)
            if idx < 0:
                idx = 0
            if idx >= nbins:
                idx = nbins - 1
            counts[idx] += 1
            if d == -1:
                counts_in[idx] += 1
            else:
                counts_out[idx] += 1

        # convertir a paquetes por segundo (pps) en cada bin
        pps = [c / float(w) for c in counts] if nbins > 0 else []

        # métricas solicitadas (calculadas sobre la serie pps)
        total_packets = int(sum(counts))                # 14.x.1 -> número total de paquetes (suma de bins)
        mean_pps = float(np.mean(pps)) if pps else 0.0  # 14.x.2 -> media de pps por bin
        max_pps = float(max(pps)) if pps else 0.0      # 14.x.3 -> máximo pps en un bin
        min_pps = float(min(pps)) if pps else 0.0      # 14.x.4 -> mínimo pps en un bin
        std_pps = float(np.std(pps)) if pps else 0.0   # 14.x.5 -> std de pps
        burst_count = int(sum(1 for c in counts if c > 0))  # 14.x.6 -> número de bins con >0 paquetes

        # in/out: totales en esta ventana (suma sobre bins de in/out) -> ratio total_in / total_out
        total_in = int(sum(counts_in))
        total_out = int(sum(counts_out))
        in_out_ratio = None
        if total_out > 0:
            in_out_ratio = float(total_in) / float(total_out)
        else:
            in_out_ratio = None

        # guardar features individuales con nombres claros
        feat[f'pps_w{w}_total_packets'] = total_packets
        feat[f'pps_w{w}_mean'] = mean_pps
        feat[f'pps_w{w}_max'] = max_pps
        feat[f'pps_w{w}_min'] = min_pps
        feat[f'pps_w{w}_std'] = std_pps
        feat[f'pps_w{w}_burst_count'] = burst_count
        feat[f'pps_w{w}_in_out'] = in_out_ratio

    # -------------------------
    # CUMUL y TAM (15..16)
    # -------------------------
    cumul_in = np.cumsum((arr == -1).astype(int))
    cumul_out = np.cumsum((arr == 1).astype(int))
    x_orig = np.arange(n)
    x_new = np.linspace(0, n-1, interp_points)
    cumul_in_interp = np.interp(x_new, x_orig, cumul_in)
    cumul_out_interp = np.interp(x_new, x_orig, cumul_out)


    # -------------------------
    # Representación numérica de CUMULs (resumenes de vectores)
    # -------------------------
    def _vector_summary(prefix, vec):
        """
        Calcula y asigna a feat[...] un conjunto de resumenes numéricos para la lista/array `vec`.
        Prefijo ejemplo: 'cumul_in' -> guardará feat['cumul_in_len'], feat['cumul_in_last'], ...
        """
        try:
            arr = np.array(vec, dtype=float)
        except Exception:
            arr = np.array([], dtype=float)

        # defaults
        feat[f'{prefix}_len'] = int(arr.size)
        if arr.size == 0:
            feat[f'{prefix}_last'] = None
            feat[f'{prefix}_mean'] = None
            feat[f'{prefix}_max'] = None
            feat[f'{prefix}_min'] = None
            feat[f'{prefix}_std'] = None
            feat[f'{prefix}_auc'] = None
            feat[f'{prefix}_slope'] = None
            feat[f'{prefix}_slope_norm'] = None
            feat[f'{prefix}_diffs_mean'] = None
            feat[f'{prefix}_diffs_median'] = None
            feat[f'{prefix}_diffs_std'] = None
            feat[f'{prefix}_diffs_p25'] = None
            feat[f'{prefix}_diffs_p50'] = None
            feat[f'{prefix}_diffs_p75'] = None
            feat[f'{prefix}_diffs_p90'] = None
            feat[f'{prefix}_pct_increasing'] = None
            feat[f'{prefix}_skew'] = None
            feat[f'{prefix}_kurtosis'] = None
            return

        # básicos
        feat[f'{prefix}_last'] = float(arr[-1])
        feat[f'{prefix}_mean'] = float(np.mean(arr))
        feat[f'{prefix}_max'] = float(np.max(arr))
        feat[f'{prefix}_min'] = float(np.min(arr))
        feat[f'{prefix}_std'] = float(np.std(arr))

        # AUC (approx integral / area under curve) usando suma simple
        feat[f'{prefix}_auc'] = float(np.trapz(arr)) if arr.size >= 2 else float(arr.sum())

        # slope: ajuste lineal arr ~ a*x + b (x = 0..len-1)
        if arr.size >= 2:
            x = np.arange(arr.size, dtype=float)
            # polyfit puede fallar en casos extraños; capturamos excepciones
            try:
                a, b = np.polyfit(x, arr, 1)
                feat[f'{prefix}_slope'] = float(a)
                # pendiente normalizada: pendiente / último valor (evita división por 0)
                feat[f'{prefix}_slope_norm'] = float(a / arr[-1]) if arr[-1] != 0 else None
            except Exception:
                feat[f'{prefix}_slope'] = None
                feat[f'{prefix}_slope_norm'] = None
        else:
            feat[f'{prefix}_slope'] = None
            feat[f'{prefix}_slope_norm'] = None

        # diferencias (incrementos entre puntos consecutivos)
        if arr.size >= 2:
            diffs = np.diff(arr)
            feat[f'{prefix}_diffs_mean'] = float(np.mean(diffs))
            feat[f'{prefix}_diffs_median'] = float(np.median(diffs))
            feat[f'{prefix}_diffs_std'] = float(np.std(diffs))
            p25, p50, p75, p90 = np.percentile(diffs, [25, 50, 75, 90]).tolist()
            feat[f'{prefix}_diffs_p25'] = float(p25)
            feat[f'{prefix}_diffs_p50'] = float(p50)
            feat[f'{prefix}_diffs_p75'] = float(p75)
            feat[f'{prefix}_diffs_p90'] = float(p90)
            # porcentaje de incrementos positivos (tendencia puntual)
            feat[f'{prefix}_pct_increasing'] = float((diffs > 0).sum()) / float(diffs.size)
        else:
            feat[f'{prefix}_diffs_mean'] = None
            feat[f'{prefix}_diffs_median'] = None
            feat[f'{prefix}_diffs_std'] = None
            feat[f'{prefix}_diffs_p25'] = None
            feat[f'{prefix}_diffs_p50'] = None
            feat[f'{prefix}_diffs_p75'] = None
            feat[f'{prefix}_diffs_p90'] = None
            feat[f'{prefix}_pct_increasing'] = None

        # skew / kurtosis (si scipy disponible)
        try:
            if 'skew' in globals() and skew is not None:
                feat[f'{prefix}_skew'] = float(skew(arr)) if arr.size>0 else None
            else:
                feat[f'{prefix}_skew'] = None
        except Exception:
            feat[f'{prefix}_skew'] = None

        try:
            if 'kurtosis' in globals() and kurtosis is not None:
                feat[f'{prefix}_kurtosis'] = float(kurtosis(arr)) if arr.size>0 else None
            else:
                feat[f'{prefix}_kurtosis'] = None
        except Exception:
            feat[f'{prefix}_kurtosis'] = None

    # llamar resúmenes para los 4 vectores
    _vector_summary('cumul_in', cumul_in)
    _vector_summary('cumul_out', cumul_out)
    _vector_summary('cumul_interp_in', cumul_in_interp)
    _vector_summary('cumul_interp_out', cumul_out_interp)
    
    # -------------------------
    # TAM -> expandido a features independientes por ventana (tam_intervals)
    # -------------------------
    # tam_intervals debe estar definido (ej. (1,2,3,4,5))
    for s in tam_intervals:
        key_prefix = f"tam_{s}s"

        # defaults si no hay duración o no hay paquetes
        if duration == 0 or len(times) == 0:
            feat[f'{key_prefix}_in_mean'] = None
            feat[f'{key_prefix}_in_max'] = None
            feat[f'{key_prefix}_in_min'] = None
            feat[f'{key_prefix}_in_std'] = None

            feat[f'{key_prefix}_out_mean'] = None
            feat[f'{key_prefix}_out_max'] = None
            feat[f'{key_prefix}_out_min'] = None
            feat[f'{key_prefix}_out_std'] = None
            continue

        # número de bins de tamaño s en la duración total
        nbins = int(np.ceil(duration / s))
        in_counts = [0] * nbins
        out_counts = [0] * nbins

        # rellenar conteos por bin
        for t, d in zip(times, arr):
            idx = int((t - start_t) // s)
            if idx < 0:
                idx = 0
            if idx >= nbins:
                idx = nbins - 1
            if d == -1:
                in_counts[idx] += 1
            else:
                out_counts[idx] += 1

        # preparar arrays para estadísticos (garantizar al menos un elemento para evitar errores en max/min)
        rows_in = in_counts if in_counts else [0]
        rows_out = out_counts if out_counts else [0]

        # estadísticas in
        try:
            feat[f'{key_prefix}_in_mean'] = float(np.mean(rows_in))
        except Exception:
            feat[f'{key_prefix}_in_mean'] = None
        try:
            feat[f'{key_prefix}_in_max'] = int(max(rows_in)) if rows_in else None
        except Exception:
            feat[f'{key_prefix}_in_max'] = None
        try:
            feat[f'{key_prefix}_in_min'] = int(min(rows_in)) if rows_in else None
        except Exception:
            feat[f'{key_prefix}_in_min'] = None
        try:
            feat[f'{key_prefix}_in_std'] = float(np.std(rows_in)) if rows_in else None
        except Exception:
            feat[f'{key_prefix}_in_std'] = None

        # estadísticas out
        try:
            feat[f'{key_prefix}_out_mean'] = float(np.mean(rows_out))
        except Exception:
            feat[f'{key_prefix}_out_mean'] = None
        try:
            feat[f'{key_prefix}_out_max'] = int(max(rows_out)) if rows_out else None
        except Exception:
            feat[f'{key_prefix}_out_max'] = None
        try:
            feat[f'{key_prefix}_out_min'] = int(min(rows_out)) if rows_out else None
        except Exception:
            feat[f'{key_prefix}_out_min'] = None
        try:
            feat[f'{key_prefix}_out_std'] = float(np.std(rows_out)) if rows_out else None
        except Exception:
            feat[f'{key_prefix}_out_std'] = None

    # -------------------------
    # Distribuciones y percentiles (17..19)
    # -------------------------
    if duration > 0:
        nbins = int(np.ceil(duration))
        sec_counts = [0]*max(1, nbins)
        for t in times:
            idx = int((t - start_t) // 1)
            if idx < 0:
                idx = 0
            if idx >= len(sec_counts):
                idx = len(sec_counts)-1
            sec_counts[idx] += 1

        # calcular percentiles
        percentiles = np.percentile(sec_counts, [25, 50, 75, 90])
        feat['global_packet_count_25'] = float(percentiles[0])
        feat['global_packet_count_50'] = float(percentiles[1])
        feat['global_packet_count_75'] = float(percentiles[2])
        feat['global_packet_count_90'] = float(percentiles[3])

    else:
        feat['global_packet_count_25'] = 0.0
        feat['global_packet_count_50'] = 0.0
        feat['global_packet_count_75'] = 0.0
        feat['global_packet_count_90'] = 0.0

    # primeros/ultimos 30
    first30 = vec[:30]
    last30 = vec[-30:]

    def summarize_30(subvec, prefix):
        total = len(subvec)
        if total == 0:
            feat[f'{prefix}_out_in_ratio'] = None
            feat[f'{prefix}_out_total_ratio'] = None
            feat[f'{prefix}_in_total_ratio'] = None
            return

        out_c = sum(1 for x in subvec if x == 1)
        in_c = sum(1 for x in subvec if x == -1)
        
        feat[f'{prefix}_out_in_ratio'] = float(out_c) / in_c if in_c > 0 else None
        feat[f'{prefix}_out_total_ratio'] = float(out_c) / total
        feat[f'{prefix}_in_total_ratio'] = float(in_c) / total

    # primeros 30 paquetes
    summarize_30(first30, 'first30')
    # últimos 30 paquetes
    summarize_30(last30, 'last30')

    # -------------------------
    # 20. Ratios globales en paquetes
    # -------------------------
    total_packets = len(vec)
    out_count = sum(1 for x in vec if x == 1)
    in_count = sum(1 for x in vec if x == -1)

    feat['ratio_out_total'] = float(out_count) / total_packets if total_packets > 0 else None
    feat['ratio_in_total'] = float(in_count) / total_packets if total_packets > 0 else None
    feat['ratio_out_in'] = float(out_count) / float(in_count) if in_count > 0 else None

    # -------------------------
    # 21. Ratios sobre ráfagas
    # -------------------------
    total_bursts = len(bursts_lengths)  # total de ráfagas
    in_bursts_count = len(in_bursts)    # cantidad de ráfagas entrantes
    out_bursts_count = len(out_bursts)  # cantidad de ráfagas salientes

    # 21.1 Ratio de ráfagas entrantes / total
    feat['ratio_in_bursts_total'] = float(in_bursts_count) / total_bursts if total_bursts > 0 else None
    # 21.2 Ratio de ráfagas salientes / total
    feat['ratio_out_bursts_total'] = float(out_bursts_count) / total_bursts if total_bursts > 0 else None
    # 21.3 Ratio de ráfagas salientes / ráfagas entrantes
    feat['ratio_out_in_bursts'] = float(out_bursts_count) / float(in_bursts_count) if in_bursts_count > 0 else None
    # 21.4 Ratio de primeras N ráfagas (N=50) / total ráfagas
    first_bursts_N = bursts_lengths[:50]  # puede ser ndarray vacío
    # suma de paquetes en las primeras N ráfagas (sumatoria de tamaños de ráfaga = número de paquetes)
    sum_first_bursts_pkts = int(first_bursts_N.sum()) if first_bursts_N.size > 0 else 0
    # suma total de paquetes en todas las ráfagas (suma de bursts_lengths)
    sum_all_bursts_pkts = int(bursts_lengths.sum()) if bursts_lengths.size > 0 else 0
    # Guardar feature: proporción de paquetes en las primeras N ráfagas respecto al total (en paquetes)
    feat['ratio_pkts_firstN_bursts_total'] = float(sum_first_bursts_pkts) / float(sum_all_bursts_pkts) if sum_all_bursts_pkts > 0 else None

    # -------------------------
    # Ordering / n-grams / transiciones (22..23)
    # -------------------------

    # -------------------------
    # 22.2 N-grams de dirección
    # -------------------------
    seq = list(arr.tolist())

    # Nota: si ya tienes 'cnt' (Counter) disponible, podrías usarlo directamente.
    # Aquí recalculamos de forma segura a partir de la secuencia `seq`.
    topK = 5

    for ngram_n in (2, 3, 4):
        cnt = ngrams_counts(seq, ngram_n) if seq is not None else Counter()
        total_occ = int(sum(cnt.values())) if cnt else 0
        unique_ngrams = int(len(cnt))
        # lista de counts ordenada desc
        counts_sorted = sorted([int(v) for v in cnt.values()], reverse=True) if cnt else []

        # Entropía sobre la distribución de counts (convertir a probabilidades)
        if total_occ > 0:
            probs = [c / float(total_occ) for c in counts_sorted]
            ngram_entropy = float(shannon_entropy(probs))
        else:
            ngram_entropy = None

        # estadísticas básicas
        if counts_sorted:
            mean_count = float(np.mean(counts_sorted))
            std_count = float(np.std(counts_sorted))
            max_count = int(counts_sorted[0])
            min_count = int(counts_sorted[-1])
        else:
            mean_count = std_count = None
            max_count = min_count = None

        # Top-K raw + fraction (rellenar con 0 si faltan)
        top_counts = counts_sorted[:topK] + [0] * max(0, topK - len(counts_sorted))
        top_fracs = [(c / float(total_occ)) if total_occ>0 else 0.0 for c in top_counts]
        topk_cumfrac = float(sum(top_counts)) / float(total_occ) if total_occ > 0 else None

        # sparsity: proporción inversa entre ngram distintos y número de windows posibles
        # aproximación: número de posiciones donde un ngram puede empezar = max(1, len(seq)-ngram_n+1)
        possible_windows = max(1, max(0, len(seq) - ngram_n + 1))
        try:
            sparsity = 1.0 - (unique_ngrams / float(possible_windows)) if possible_windows>0 else None
        except Exception:
            sparsity = None

        # Guardar en feat con nombres claros
        prefix = f'ngrams_{ngram_n}'
        feat[f'{prefix}_unique'] = unique_ngrams
        feat[f'{prefix}_total_occurrences'] = total_occ
        feat[f'{prefix}_entropy'] = ngram_entropy
        feat[f'{prefix}_mean_count'] = mean_count
        feat[f'{prefix}_std_count'] = std_count
        feat[f'{prefix}_max_count'] = max_count
        feat[f'{prefix}_min_count'] = min_count
        feat[f'{prefix}_possible_windows'] = int(possible_windows)
        feat[f'{prefix}_sparsity'] = sparsity
        feat[f'{prefix}_topk_cumfrac'] = topk_cumfrac

        for i in range(topK):
            feat[f'{prefix}_top{i+1}_count'] = int(top_counts[i])
            feat[f'{prefix}_top{i+1}_frac'] = float(top_fracs[i])

    # -------------------------
    # 23. Transiciones
    # -------------------------
    transitions = Counter()
    for i in range(len(seq)-1):
        transitions[(seq[i], seq[i+1])] += 1

    # Guardar transiciones específicas
    feat['transition_p1_to_m1'] = int(transitions.get((1, -1), 0))   # 23.1.1  +1 -> -1
    feat['transition_m1_to_p1'] = int(transitions.get((-1, 1), 0))   # 23.1.2  -1 -> +1
    feat['transition_p1_to_p1'] = int(transitions.get((1, 1), 0))    # 23.1.3  +1 -> +1
    feat['transition_m1_to_m1'] = int(transitions.get((-1, -1), 0))  # 23.1.4  -1 -> -1

    # Transition rate: proporción de cambios de dirección
    num_changes = sum(1 for i in range(len(seq)-1) if seq[i] != seq[i+1])
    feat['transition_rate'] = float(num_changes) / max(1, len(seq)-1)

    # -------------------------
    # Entropía & concentración (24..26)
    # -------------------------
    p_out = float(total_out)/n #Nùmero de paquetes salientes +1
    p_in = float(total_in)/n #Nùmero de paquetes entrantes -1
    feat['entropy'] = shannon_entropy([p_out, p_in]) #La entropía mide incertidumbre o aleatoriedad en la distribución de direcciones. H=−i∑​pi​⋅log2​(pi​) 

    def frac_in_time_window(start_frac, end_frac, by_bytes=False):
        if duration==0:
            return 0.0
        s_cut = start_t + start_frac*duration
        e_cut = start_t + end_frac*duration
        if by_bytes:
            total_bytes = sum(sizes)
            if total_bytes==0:
                return 0.0
            bytes_in = sum(sz for t, sz in zip(times, sizes) if t>=s_cut and t<e_cut)
            return float(bytes_in)/total_bytes
        else:
            count = sum(1 for t in times if t>=s_cut and t<e_cut)
            return float(count)/n

    feat['concentration_25pct'] = frac_in_time_window(0.0, 0.25, by_bytes=False)
    feat['concentration_10pct'] = frac_in_time_window(0.0, 0.10, by_bytes=False)
    feat['bytes_first25pct'] = frac_in_time_window(0.0, 0.25, by_bytes=True)
    feat['bytes_first10pct'] = frac_in_time_window(0.0, 0.10, by_bytes=True)
    feat['packets_last25pct'] = frac_in_time_window(0.75, 1.0, by_bytes=False)
    feat['packets_last10pct'] = frac_in_time_window(0.9, 1.0, by_bytes=False)
    feat['bytes_last25pct'] = frac_in_time_window(0.75, 1.0, by_bytes=True)
    feat['bytes_last10pct'] = frac_in_time_window(0.9, 1.0, by_bytes=True)

    # -------------------------
    # Carga/timing específicos (27..28)
    # -------------------------
    big_burst_idx = None
    idx_acc = 0
    for bl in bursts_lengths:
        if bl >= 10:
            big_burst_idx = idx_acc
            break
        idx_acc += bl
    feat['time_to_first_big_burst'] = float(times[big_burst_idx]) if (big_burst_idx is not None and big_burst_idx < len(times)) else None
    # Tiempo hasta el primer burst grande (primer burst con tamaño >= 10).
    burst_starts = []
    cur = arr[0]
    cur_start = 0
    for i in range(1, n):
        if arr[i]!=cur:
            burst_starts.append(times[cur_start])
            cur = arr[i]
            cur_start = i
    burst_starts.append(times[cur_start])
    # Tiempo promedio entre ráfagas.
    if len(burst_starts) >= 2:
        diffs_bs = np.diff(np.array(burst_starts))
        feat['mean_time_between_bursts'] = float(np.mean(diffs_bs))
    else:
        feat['mean_time_between_bursts'] = None

    # -------------------------
    # Metadata y derived (29..33)
    # -------------------------
    # DoA (duration of activity)
    feat['DoA'] = float(duration)


    try:
        serie = np.array(cumul_in - np.mean(cumul_in)) if len(cumul_in)>1 else np.array([0.0])
        fft = np.abs(np.fft.rfft(serie))
        total_energy = float(np.sum(fft**2))
        if total_energy>0:
            third = len(fft)//3
            feat['spectral_energy_low'] = float(np.sum(fft[:third]**2))/total_energy
            feat['spectral_energy_mid'] = float(np.sum(fft[third:2*third]**2))/total_energy
            feat['spectral_energy_high'] = float(np.sum(fft[2*third:]**2))/total_energy
        else:
            feat['spectral_energy_low'] = feat['spectral_energy_mid'] = feat['spectral_energy_high'] = 0.0
    except Exception:
        feat['spectral_energy_low'] = feat['spectral_energy_mid'] = feat['spectral_energy_high'] = None

    # store extras
    #feat['vector'] = vec
    #feat['times'] = times
    #feat['sizes'] = sizes

    return feat


# ------------------------------
# Procesado de PCAP - VERSIÓN FINAL
# ------------------------------
def process_pcap_with_features(pcap_path: Path, log_path: Path, max_packets: int = 0):
    """
    Lee un PCAP y procesa todos los paquetes TCP (sin filtrar por puertos).
    
    Lógica de dirección:
    - La IP origen del primer paquete TCP válido con SYN sin ACK se toma como 'originator_ip'.
    - Si no hay SYN, se usa el primer src observado como fallback.
    - Paquetes:
        src == originator_ip -> 1 (saliente)
        src != originator_ip -> -1 (entrante)
    
    Se guarda un log detallado por pcap y se generan features.
    
    Args:
        pcap_path (Path): ruta al archivo pcap
        log_path (Path): ruta al archivo log
        max_packets (int): límite de paquetes a procesar (0 = todos)
    
    Returns:
        vec, times, sizes, features, stats
    """
    src_sequence = []
    times = []
    sizes = []
    matches = 0
    errors = []

    originator_ip = None
    first_src_seen = None  # fallback seguro

    # Abrir log
    with open(log_path, 'w') as lf:
        lf.write(f"pcap,{pcap_path.name}\n")
        lf.write("line,packet_index,timestamp,src,src_port,dst,dst_port,protocol,decision,size,notes\n")

        try:
            with PcapReader(str(pcap_path)) as rdr:
                for i, pkt in enumerate(rdr):
                    if max_packets and i >= max_packets:
                        break

                    try:
                        if IP not in pkt or TCP not in pkt:
                            continue

                        src = pkt[IP].src
                        dst = pkt[IP].dst
                        tcp_layer = pkt[TCP]

                        # Guardar primer src (fallback)
                        if first_src_seen is None:
                            first_src_seen = src

                        src_sequence.append(src)
                        timestamp = float(getattr(pkt, 'time', 0.0))
                        times.append(timestamp)
                        sizes.append(len(pkt))
                        matches += 1

                        # Detectar originador usando TCP SYN sin ACK
                        if originator_ip is None:
                            flags = int(tcp_layer.flags)
                            is_syn = flags & 0x02
                            is_ack = flags & 0x10
                            if is_syn and not is_ack:
                                originator_ip = src
                                lf.write(f"INFO,Originator detected via SYN: {originator_ip}\n")

                    except Exception as e_pkt:
                        errors.append(str(e_pkt))
                        lf.write(f"ERROR,packet_index={i},error={e_pkt}\n")
                        continue  # continuar con el siguiente paquete

        except Exception as e_file:
            errors.append(str(e_file))
            lf.write(f"ERROR,PCAP_READ,{e_file}\n")

        # Fallback si no se detectó ningún SYN
        if originator_ip is None and first_src_seen is not None:
            originator_ip = first_src_seen
            lf.write(f"INFO,Fallback applied: originator_ip set to first_src_seen {originator_ip}\n")

    # Construir vector final de dirección
    vec = [1 if s == originator_ip else -1 for s in src_sequence]

    # Normalizar tiempos
    if times:
        t0 = times[0]
        times = [float(t - t0) for t in times]

    stats = {
        'pcap': pcap_path.name,
        'matches': matches,
        'vector_len': len(vec),
        'errors': errors
    }

    features = extract_features_from_vector(
        vec,
        times,
        sizes=sizes,
        interp_points=100,
        tam_intervals=(1, 2, 3, 4, 5)
    )

    return vec, times, sizes, features, stats


# ------------------------------
# Buscar pcaps
# ------------------------------
def find_pcaps(site_pcaps_dir: Path):
    return sorted(list(site_pcaps_dir.glob('**/*.pcap')) + list(site_pcaps_dir.glob('**/*.pcapng')))

# ------------------------------
# MAIN CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Filtrar PCAPs por puertos TCP y extraer features (130).")
    ap.add_argument('--pcap-root', required=True, help='Directorio raíz (ej: /datos) que contiene categorías.')
    ap.add_argument('--outdir', default=str(data_path('generated', 'preprocessed')), help='Directorio de salida CSV.')
    ap.add_argument('--logdir', default=str(result_path('diagnostics', 'preprocess_logs')), help='Directorio de logs por pcap.')
    ap.add_argument('--max-packets', type=int, default=0, help='Limitar lectura a N paquetes por pcap (0 = full).')
    ap.add_argument('--min-fraction', type=float, default=0.2, help='Fracción de la media de packets por sitio.')
    ap.add_argument('--min-abs', type=int, default=0, help='Umbral absoluto mínimo de paquetes.')

    args = ap.parse_args()

    root = Path(args.pcap_root) # Esto es /datos
    outdir = Path(args.outdir)
    logdir = Path(args.logdir)
    
    # Crear directorios de salida
    logdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    discard_dir = outdir / "descartados"
    discard_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Iniciando procesamiento con estructura de Categorías -> Sitios.")
    print(f"[INFO] Filtro por tamaño relativo: min_fraction={args.min_fraction}, min_abs={args.min_abs}")

    global_summary = []

    # ---------------------------------------
    # Buffers globales (1 fila = 1 PCAP)
    # ---------------------------------------
    all_features_rows = []
    all_vectors_rows = []
    all_discarded_rows = []

    # Identificador global único de PCAP
    global_pcap_id = 0

    # -> Inicializar el contador global de sitios
    global_site_count = 0

    # ---------------------------------------------------------------------
    # NIVEL 0: Iterar por INSTALACIONES (ej: Inst1, Inst2, Inst3)
    # ---------------------------------------------------------------------

    for inst_dir in sorted(root.iterdir()):
        if not inst_dir.is_dir() or inst_dir.name.startswith('.'):
            continue

        print(f"\n{'='*60}")
        print(f"[INST] Procesando instancia: {inst_dir.name}")
        print(f"{'='*60}")
        # ---------------------------------------------------------------------
        # NIVEL 1: Iterar por CATEGORÍAS (ej: belleza, deportes, politica)
        # ---------------------------------------------------------------------

        for category_dir in sorted(inst_dir.iterdir()):
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue

            print(f"\n{'-'*60}")
            print(f"[CATEGORY] Procesando categoría: {category_dir.name}")
            print(f"{'-'*60}")

            # -----------------------------------------------------------------
            # 1. Construir skip_map ESPECÍFICO para esta categoría
            #    Busca en category_dir/capture_logs
            # -----------------------------------------------------------------
            skip_map = {}   # pcap_basename -> list of (csv_path, note)
            csvs_read_category = []
            captures_logs_dir = category_dir / 'captures_logs'

            if captures_logs_dir.exists() and captures_logs_dir.is_dir():
                for csv_path in sorted(captures_logs_dir.rglob('*.csv')):
                    try:
                        df_csv = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
                    except Exception as e:
                        print(f"[WARN] No se pudo leer CSV {csv_path}: {e}")
                        continue
                    
                    csvs_read_category.append(str(csv_path))
                    
                    # Procesar notas de ignorado
                    if 'pcap_raw' in df_csv.columns and 'notes' in df_csv.columns:
                        for _, row in df_csv.iterrows():
                            notes_val = str(row.get('notes', '')).strip()
                            pcap_raw_val = row.get('pcap_raw', '')
                            if notes_val:
                                try:
                                    pcap_name = Path(str(pcap_raw_val)).name
                                except Exception:
                                    pcap_name = str(pcap_raw_val)
                                if pcap_name:
                                    skip_map.setdefault(pcap_name, []).append((str(csv_path), notes_val))
            
                print(f"[INFO] Categoría '{category_dir.name}': {len(skip_map)} pcaps marcados para ignorar en {len(csvs_read_category)} logs.")
            else:
                print(f"[WARN] No se encontró carpeta 'captures_logs' en {category_dir.name}. No se ignorará ningún pcap por notas.")

            # -----------------------------------------------------------------
            # NIVEL 2: Iterar por SITIOS WEB dentro de la categoría
            # -----------------------------------------------------------------
            for site_dir in sorted(category_dir.iterdir()):
                if not site_dir.is_dir():
                    continue
                
                # Ignorar la carpeta de logs propia de la categoría y carpetas de código
                if site_dir.name == 'captures_logs' or site_dir.name.lower() == 'codigo':
                    continue

                pcaps_dir = site_dir / 'pcaps'
                if not pcaps_dir.exists():
                    # A veces no hay carpeta pcaps, solo archivos sueltos? Asumo estructura estricta.
                    # print(f"[SKIP] {site_dir.name}: no existe carpeta pcaps/") 
                    continue

                # Listar pcaps en disco
                pcaps = find_pcaps(pcaps_dir)
                if not pcaps:
                    print(f"[INFO] {site_dir.name} (Cat: {category_dir.name}): no pcaps encontrados")
                    continue

                # Filtrar usando el skip_map DE ESTA CATEGORÍA
                total_found = len(pcaps)
                filtered_pcaps = []
                skipped_by_notes = []
                
                for p in pcaps:
                    # Chequear tanto nombre completo como stem (sin extension)
                    if p.name in skip_map or p.stem in skip_map:
                        skipped_by_notes.append(p)
                    else:
                        filtered_pcaps.append(p)

                print(f"[SITE] {site_dir.name}: encontrados={total_found}, ignorados_logs={len(skipped_by_notes)}, procesar={len(filtered_pcaps)}")

                # Procesar los pcaps válidos
                rows = []
                
                # Barra de progreso por sitio
                desc_text = f"{category_dir.name}/{site_dir.name}"

                # -------------------------------------------------------------
                # Asignación de etiqueta de sitio ÚNICA (aquí lo necesitamos)
                # -------------------------------------------------------------
                site_label_id = global_site_count
                global_site_count += 1 # Aumentar el contador para el siguiente sitio
                for pcap in tqdm(filtered_pcaps, desc=desc_text, leave=False):

                    

                    # Log específico
                    log_path = logdir / f"{category_dir.name}_{site_dir.name}__{pcap.stem}__filter.log"
                    
                    # LLAMADA A TU FUNCIÓN DE PROCESAMIENTO
                    vec, times, sizes, features, stats = process_pcap_with_features(pcap, log_path, max_packets=args.max_packets)
       
                    rows.append({
                        'pcap_name': pcap.name,
                        'vec': vec,
                        'times': times,
                        'sizes': sizes,
                        'features': features,
                        'stats': stats
                    })

                # -------------------------------------------------------------
                # FILTRADO ESTADÍSTICO (Mismo algoritmo, aplicado por sitio)
                # -------------------------------------------------------------
                lengths = [r.get('stats', {}).get('vector_len', 0) for r in rows]
                mean_len = float(np.mean(lengths)) if lengths else 0.0
                threshold = max(args.min_abs, mean_len * args.min_fraction)
                threshold_int = int(math.ceil(threshold))

                kept_rows = []
                discarded_rows = []
                for r in rows:
                    vlen = int(r.get('stats', {}).get('vector_len', 0))
                    if vlen >= threshold_int:
                        kept_rows.append(r)
                    else:
                        r['reason'] = f"below_threshold({vlen}<{threshold_int})"
                        discarded_rows.append(r)

                for r in discarded_rows:
                    all_discarded_rows.append({
                        'pcap_name': r['pcap_name'],
                        'site_label': site_label_id,
                        'category': category_dir.name,
                        'site': site_dir.name,
                        'vector_len': r['stats']['vector_len'],
                        'reason': r['reason']
                    })
                
                for r in kept_rows:
                    pcap_uid = global_pcap_id
                    global_pcap_id += 1
                    # FEATURES
                    feat_row = {
                        'pcap_uid': pcap_uid,
                        'site_label': site_label_id,
                        'pcap_name': r['pcap_name'],
                        'category': category_dir.name,
                        'site': site_dir.name,
                        'vector_len': r['stats']['vector_len'],
                        'matches': r['stats']['matches'],
                        'n_errors': len(r['stats']['errors'])
                    }
                    feat_row.update(r['features'])
                    all_features_rows.append(feat_row)

                    # VECTORES
                    vec_row = {
                        'pcap_uid': pcap_uid,
                        'site_label': site_label_id,
                        'direction_vector': json.dumps(r['vec']),
                        'time_vector': json.dumps(r['times']),
                        'size_vector': json.dumps(r['sizes']),
                    }
                    all_vectors_rows.append(vec_row)

                # Añadir al resumen global
                global_summary.append({
                    'category': category_dir.name,
                    'site': site_dir.name,
                    'total_disk': total_found,
                    'skipped_logs': len(skipped_by_notes),
                    'kept': len(kept_rows),
                    'discarded_stats': len(discarded_rows)
                })

    # ---------------------------------------
    # GUARDADO FINAL (SOLO 2 CSVs)
    # ---------------------------------------
    df_features = pd.DataFrame(all_features_rows)
    df_vectors = pd.DataFrame(all_vectors_rows)

    # Validaciones duras
    assert len(df_features) == len(df_vectors), "ERROR: Features y vectores no alineados"
    assert (df_features['pcap_uid'].values == df_vectors['pcap_uid'].values).all(), \
        "ERROR: pcap_uid desalineado entre CSVs"

    df_features.to_csv(outdir / "final_features_sites.csv", index=False)
    df_vectors.to_csv(outdir / "final_vectors_sites.csv", index=False)
    if all_discarded_rows:
        df_discarded = pd.DataFrame(all_discarded_rows)
        df_discarded.to_csv(outdir / "discarded_pcaps.csv", index=False)


    print(f"[OK] Generados CSV finales con {len(df_features)} PCAPs")
    
    # ------------------------------
    # RESUMEN FINAL
    # ------------------------------
    print("\n" + "="*60)
    print("[GLOBAL SUMMARY EXECUTION]")
    print("="*60)
    df_summary = pd.DataFrame(global_summary)
    if not df_summary.empty:
        print(df_summary.to_string())
        print("-" * 60)
        print(f"Total Sites Procesados: {len(df_summary)}")
        print(f"Total Pcaps Kept: {df_summary['kept'].sum()}")
    else:
        print("No se procesaron sitios.")

if __name__ == "__main__":
    main()
