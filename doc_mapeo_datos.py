import pandas as pd
import json
import os

# --- Configuración ---
# Usamos el archivo de la Fase 1 o el original donde estén ambos datos
INPUT_RAW_CSV = "output/final_features_sites.csv" 
OUTPUT_DIR = "output/reference/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_site_mapping(csv_path):
    print(f"🔍 Cargando datos desde {csv_path}...")
    # Leemos solo las columnas necesarias para ahorrar memoria
    try:
        df = pd.read_csv(csv_path, usecols=['site_label', 'site'])
    except ValueError:
        # Si la columna se llama diferente (ej: 'url', 'domain', 'web_name')
        df = pd.read_csv(csv_path)
        print("⚠️ Columna 'site' no encontrada. Columnas disponibles:", df.columns.tolist())
        return

    # Eliminar duplicados para tener 1 registro por sitio
    mapping_df = df[['site_label', 'site']].drop_duplicates().sort_values('site_label')

    # 1. Guardar como CSV (fácil de leer en Excel)
    csv_out = os.path.join(OUTPUT_DIR, "site_dictionary.csv")
    mapping_df.to_csv(csv_out, index=False)

    # 2. Guardar como JSON (ideal para cargar como diccionario en Python)
    # Formato: { "0": "google.com", "1": "facebook.com", ... }
    site_dict = dict(zip(mapping_df['site_label'].astype(str), mapping_df['site']))
    
    json_out = os.path.join(OUTPUT_DIR, "site_dictionary.json")
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(site_dict, f, indent=4, ensure_ascii=False)

    print(f"✅ Diccionario creado con {len(site_dict)} sitios.")
    print(f"📂 Archivos generados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_site_mapping(INPUT_RAW_CSV)