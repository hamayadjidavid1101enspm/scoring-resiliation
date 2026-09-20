import json

with open('notebooks/tp_final.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

for cell in d.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell.get('source', []):
            new_source.append(line.replace("'Usage Véhicule': 'Personnel'", "'Usage Véhicule': 'Domicile-Travail'"))
        cell['source'] = new_source

with open('notebooks/tp_final.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1)
