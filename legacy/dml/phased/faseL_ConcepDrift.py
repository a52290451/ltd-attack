import pandas as pd
import ast
import numpy as np

# Rutas
FILE_FEAT_PAST = "../../output/final_features_sites.csv"
FILE_VECT_PAST = "../../output/final_vectors_sites.csv"
FILE_FEAT_FUT = "../../output/final_features_sites_concept_drift.csv"
FILE_VECT_FUT = "../../output/final_vectors_sites_concept_drift.csv"

# El Centroide "Agujero Negro" (Cámbialo si tu consola mostró otro ID numérico distinto a 74)
SINKHOLE_ID = 74 

print("[1/3] Cargando datos del PASADO...")
df_f_past = pd.read_csv(FILE_FEAT_PAST, usecols=['pcap_uid', 'site_label'])
df_v_past = pd.read_csv(FILE_VECT_PAST, usecols=['pcap_uid', 'direction_vector'])
df_past = pd.merge(df_f_past, df_v_past, on='pcap_uid')
df_past['seq_len'] = df_past['direction_vector'].apply(lambda x: len(ast.literal_eval(x)))

print("[2/3] Cargando datos del FUTURO...")
df_f_fut = pd.read_csv(FILE_FEAT_FUT, usecols=['pcap_uid', 'site_label'])
df_v_fut = pd.read_csv(FILE_VECT_FUT, usecols=['pcap_uid', 'direction_vector'])
df_fut = pd.merge(df_f_fut, df_v_fut, on='pcap_uid')
df_fut['seq_len'] = df_fut['direction_vector'].apply(lambda x: len(ast.literal_eval(x)))

# Identificar las clases válidas (las que sobrevivieron al filtro inicial)
valid_classes = df_past['site_label'].value_counts()[df_past['site_label'].value_counts() >= 500].index

print("\n[3/3] --- ANALIZANDO LAS MUTACIONES (PASADO vs FUTURO) ---")

mutations = []
for cls in valid_classes:
    past_subset = df_past[df_past['site_label'] == cls]['seq_len']
    fut_subset = df_fut[df_fut['site_label'] == cls]['seq_len']
    
    if len(fut_subset) > 0:
        past_med = past_subset.median()
        fut_med = fut_subset.median()
        shift = ((fut_med - past_med) / past_med) * 100
        
        mutations.append({
            'site': cls,
            'past_median': past_med,
            'fut_median': fut_med,
            'shift_pct': shift,
            'fut_count': len(fut_subset)
        })

df_mut = pd.DataFrame(mutations)

# Analizar al sospechoso principal
print(f"\n==================================================")
print(f" INVESTIGANDO EL CENTROIDE AGUJERO NEGRO (Sitio {SINKHOLE_ID})")
print(f"==================================================")
sinkhole_data = df_mut[df_mut['site'] == SINKHOLE_ID]
if not sinkhole_data.empty:
    print(sinkhole_data.to_string(index=False))
else:
    print("El sitio 74 no parece estar en la lista válida. Revisa los IDs.")

print(f"\n==================================================")
print(f" TOP 10 SITIOS WEB CON MAYOR MUTACIÓN (Concept Drift Severo)")
print(f" (Sitios cuyo tamaño se infló o colapsó drásticamente)")
print(f"==================================================")
# Ordenamos por la mutación absoluta más grande
df_mut['abs_shift'] = df_mut['shift_pct'].abs()
top_mutations = df_mut.sort_values(by='abs_shift', ascending=False).head(10)
print(top_mutations[['site', 'past_median', 'fut_median', 'shift_pct']].to_string(index=False))

print(f"\n==================================================")
print(f" TOP 10 SITIOS WEB MÁS ESTABLES (Concept Drift Leve)")
print(f"==================================================")
stable = df_mut.sort_values(by='abs_shift', ascending=True).head(10)
print(stable[['site', 'past_median', 'fut_median', 'shift_pct']].to_string(index=False))