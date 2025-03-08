import os
import pandas as pd
import json
import math

# Répertoire contenant vos fichiers CSV
csv_directory = '/Users/mac/Downloads/KT4'  # Remplacez par le chemin réel du dossier contenant les fichiers CSV

# Lister tous les fichiers CSV dans le répertoire
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

# Initialiser une liste pour stocker tous les DataFrames
df_list = []

# Charger chaque fichier CSV et ajouter son contenu à la liste
for csv_file in csv_files:
    file_path = os.path.join(csv_directory, csv_file)
    df = pd.read_csv(file_path)
    df_list.append(df)

# Concaténer tous les DataFrames en un seul
full_df = pd.concat(df_list, ignore_index=True)

# Convertir le DataFrame en format JSON
json_data = full_df.to_dict(orient="records")

# Calculer le nombre de partitions
num_parts = 15
part_size = math.ceil(len(json_data) / num_parts)

# Sauvegarder les données en plusieurs fichiers JSON
for i in range(num_parts):
    part_data = json_data[i * part_size:(i + 1) * part_size]
    part_output_path = f'./train_data_part{i + 1}.json'
    
    with open(part_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(part_data, json_file, ensure_ascii=False, indent=4)
    
    print(f"Les données ont été sauvegardées dans {part_output_path}")
