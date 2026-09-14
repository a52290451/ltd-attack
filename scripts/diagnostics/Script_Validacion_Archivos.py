import pandas as pd
import os

# Rutas de los archivos
files_to_check = {
    "Vectores (Pasado)": "../../output/final_vectors_sites.csv",
    "Metadatos/Features (Pasado)": "../../output/final_features_sites.csv",
    "Vectores (Futuro)": "../../output/final_vectors_sites_concept_drift.csv",
    "Metadatos/Features (Futuro)": "../../output/final_features_sites_concept_drift.csv"
}

print("\n" + "="*80)
print("🔍 INSPECCIÓN DE LA PRIMERA FILA DE CADA DATASET")
print("="*80)

def truncate_string(val, max_len=80):
    """Trunca strings largos (como los vectores crudos) para no saturar la consola."""
    s = str(val)
    if len(s) > max_len:
        return s[:max_len] + "... [TRUNCADO]"
    return s

for name, filepath in files_to_check.items():
    if os.path.exists(filepath):
        try:
            # Leemos solo la primera fila de datos
            df = pd.read_csv(filepath, nrows=1)
            print(f"\n📄 {name} ({filepath}):")
            
            # Iterar sobre las columnas y mostrar el primer valor
            for col in df.columns:
                val = df.iloc[0][col]
                print(f"   - {col}: {truncate_string(val)}")
                
        except Exception as e:
            print(f"\n❌ Error al leer {name}: {e}")
    else:
        print(f"\n⚠️ ARCHIVO NO ENCONTRADO: {name}")

print("\n" + "="*80)