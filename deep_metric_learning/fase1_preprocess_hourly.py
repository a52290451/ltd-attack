import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

# ==========================================
# CONFIGURACIÓN DE HIPERPARÁMETROS
# ==========================================
MIN_PACKETS = 50              
MIN_CAPTURES_PER_SITE = 500   
MIN_HOURS_PER_DAY = 12        
MAX_SEQ_LEN = 3000            
NORMALIZATION_DIVISOR = 3000.0 
MAX_IAT_SECONDS = 5.0         
OUTPUT_DIR = "./resultados/"

def parse_pcap_name(pcap_name):
    match = re.search(r'(\d{8}-\d{6})', str(pcap_name))
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d-%H%M%S')
        except ValueError:
            return None
    return None

def extract_metadata_and_filter(features_file):
    print("[1/5] Analizando Metadatos Ligeros (features.csv)...")
    # Cargamos solo la información ligera
    df = pd.read_csv(features_file, usecols=['pcap_uid', 'site_label', 'pcap_name'])
    
    print("      -> Parseando fechas y deduplicando...")
    df['datetime'] = df['pcap_name'].apply(parse_pcap_name)
    df = df.dropna(subset=['datetime'])
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    
    df = df.drop_duplicates(subset=['site_label', 'date', 'hour'], keep='first')
    
    # Creamos un índice rápido para búsquedas O(1)
    # pcap_uid -> (site_label, date, hour)
    valid_pcaps = df.set_index('pcap_uid')[['site_label', 'date', 'hour']].to_dict('index')
    print(f"      -> Capturas únicas listas para buscar en disco: {len(valid_pcaps)}")
    return valid_pcaps, df

def process_vectors_in_streaming(vectors_file, valid_pcaps):
    print("[2/5] Procesando Vectores Gigantes en Streaming (RAM Segura)...")
    
    # Aquí acumularemos los datos truncados (solo 3000 float16 por vector)
    profiles = defaultdict(lambda: defaultdict(dict))
    
    # Estadísticas para el filtro IQR posterior
    seq_lengths = defaultdict(list)
    
    # Leemos el archivo gigante por chunks (bloques)
    chunk_size = 5000
    procesados = 0
    
    for chunk in pd.read_csv(vectors_file, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            uid = row['pcap_uid']
            
            # Si el pcap no sobrevivió a la deduplicación, lo ignoramos sin parsear
            if uid not in valid_pcaps:
                continue
                
            meta = valid_pcaps[uid]
            site, date, hour = meta['site_label'], meta['date'], meta['hour']
            
            # 1. Parsear Dirección (y usarla para medir longitud)
            try:
                dirs_lst = json.loads(row['direction_vector'])
            except:
                clean_s = row['direction_vector'].strip('[] \n\r')
                dirs_lst = [float(x) for x in clean_s.split(',')] if clean_s else []
                
            seq_len = len(dirs_lst)
            if seq_len < MIN_PACKETS:
                continue
                
            # Registrar longitud para el IQR
            seq_lengths[site].append((date, hour, seq_len))
            
            # Truncado INMEDIATO a MAX_SEQ_LEN para no guardar basura en RAM
            dirs_arr = np.array(dirs_lst[:MAX_SEQ_LEN], dtype=np.float16)
            
            # 2. Parsear y Truncar Tamaño
            try:
                sizes_lst = json.loads(row['size_vector'])
            except:
                clean_s = row['size_vector'].strip('[] \n\r')
                sizes_lst = [float(x) for x in clean_s.split(',')] if clean_s else []
            sizes_arr = np.array(sizes_lst[:MAX_SEQ_LEN], dtype=np.float16) / NORMALIZATION_DIVISOR
            
            # 3. Parsear Tiempo y calcular IAT al instante
            try:
                time_lst = json.loads(row['time_vector'])
            except:
                clean_s = row['time_vector'].strip('[] \n\r')
                time_lst = [float(x) for x in clean_s.split(',')] if clean_s else []
                
            t_arr = np.array(time_lst, dtype=np.float32)
            if len(t_arr) > 0:
                iat_arr = np.diff(t_arr)
                iat_arr = np.concatenate(([0.0], iat_arr))
                iat_arr = iat_arr[:MAX_SEQ_LEN]
                iat_arr = np.clip(iat_arr, 0.0, MAX_IAT_SECONDS) / MAX_IAT_SECONDS
            else:
                iat_arr = np.zeros(min(seq_len, MAX_SEQ_LEN), dtype=np.float16)
            
            # Guardamos la tupla truncada
            profiles[site][date][hour] = (dirs_arr, sizes_arr, iat_arr.astype(np.float16))
            
        procesados += len(chunk)
        print(f"      ... Escaneadas {procesados} filas de vectores.")
        gc.collect() # Limpieza de basura forzada

    return profiles, seq_lengths

def apply_iqr_and_build_tensors(profiles, seq_lengths, meta_df):
    print("[3/5] Aplicando IQR y construyendo tensores 3D...")
    
    # Calcular IQR por sitio
    valid_site_date_hours = set()
    site_survivors_count = defaultdict(int)
    
    for site, records in seq_lengths.items():
        lengths = [r[2] for r in records]
        if not lengths: continue
        
        q1 = np.percentile(lengths, 25)
        q3 = np.percentile(lengths, 75)
        iqr = q3 - q1
        lower_bound = max(MIN_PACKETS, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        
        for date, hour, length in records:
            if lower_bound <= length <= upper_bound:
                valid_site_date_hours.add(f"{site}_{date}_{hour}")
                site_survivors_count[site] += 1
                
    # Filtrar minorías
    valid_sites = {s for s, c in site_survivors_count.items() if c >= MIN_CAPTURES_PER_SITE}
    print(f"      -> Sitios que sobrevivieron a estadística IQR y minorías: {len(valid_sites)}")
    
    X_tensor, y_labels = [], []
    empty_hour = np.zeros((MAX_SEQ_LEN, 3), dtype=np.float16)
    
    for site, dates in profiles.items():
        if site not in valid_sites:
            continue
            
        for date, hours_data in dates.items():
            # Limpiar horas que no pasaron el IQR
            valid_hours = {h: data for h, data in hours_data.items() if f"{site}_{date}_{h}" in valid_site_date_hours}
            
            if len(valid_hours) < MIN_HOURS_PER_DAY:
                continue
                
            profile_24h = []
            for h in range(24):
                if h in valid_hours:
                    dirs, sizes, iats = valid_hours[h]
                    
                    if len(dirs) < MAX_SEQ_LEN:
                        pad_len = MAX_SEQ_LEN - len(dirs)
                        dirs = np.pad(dirs, (0, pad_len), 'constant')
                        sizes = np.pad(sizes, (0, pad_len), 'constant')
                        iats = np.pad(iats, (0, pad_len), 'constant')
                        
                    hour_tensor = np.stack([dirs, sizes, iats], axis=-1)
                    profile_24h.append(hour_tensor)
                else:
                    profile_24h.append(empty_hour)
                    
            X_tensor.append(np.array(profile_24h, dtype=np.float16))
            y_labels.append(site)
            
    return np.array(X_tensor, dtype=np.float16), np.array(y_labels)

if __name__ == "__main__":
    FILE_FEATURES = "../../output/final_features_sites.csv" 
    FILE_VECTORS = "../../output/final_vectors_sites.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid_pcaps_dict, df_meta = extract_metadata_and_filter(FILE_FEATURES)
    
    profiles_dict, lengths_dict = process_vectors_in_streaming(FILE_VECTORS, valid_pcaps_dict)
    
    X, y = apply_iqr_and_build_tensors(profiles_dict, lengths_dict, df_meta)

    print("[4/5] Serializando tensores finales para entrenamiento...")
    np.save(os.path.join(OUTPUT_DIR, 'X_hourly_raw.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_hourly_raw.npy'), y)
    
    print(f"=== FASE 1 COMPLETADA ===")
    print(f"-> Tensor X Shape: {X.shape} (Días, Horas, Secuencia, Canales)")
    print(f"-> Tensor Y Shape: {y.shape}")
