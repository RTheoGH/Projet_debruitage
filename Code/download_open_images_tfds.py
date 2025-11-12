"""
Petit script de démonstration pour télécharger un petit sous-ensemble
d'Open Images via tensorflow_datasets (TFDS) et sauvegarder les images localement.

Usage:
    python Code/download_open_images_tfds.py

Notes:
- TFDS va télécharger les fichiers depuis les serveurs d'Open Images la première fois.
- Le dataset complet est très volumineux; ici on télécharge un petit échantillon.
- Assurez-vous d'avoir installé: tensorflow-datasets, tensorflow, pillow
  (ex: pip install tensorflow-datasets tensorflow pillow)
"""
import os
from typing import Optional

import numpy as np
try:
    import tensorflow_datasets as tfds
except Exception:
    tfds = None
from PIL import Image


def download_samples(num_samples: int = 20, out_dir: str = 'open_images_samples') -> None:
    """Télécharge `num_samples` images depuis Open Images (train split) et les sauvegarde.

    Args:
        num_samples: nombre d'images à télécharger (ex: 10, 50). Ne pas mettre trop haut.
        out_dir: dossier de sortie où les images seront écrites.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Nom du dataset dans TFDS — si cette ligne échoue, vérifiez la liste des builders
    dataset_name = 'open_images_v4'

    if tfds is None:
        raise RuntimeError(
            "tensorflow_datasets n'est pas installé. Installez-le avec: \n"
            "    pip install tensorflow-datasets tensorflow pillow\n"
            "ou utilisez votre gestionnaire d'environnement (poetry/conda)."
        )

    print(f"Chargement de '{dataset_name}' ({num_samples} échantillons) via TFDS — préparation du téléchargement...")
    # On utilise une tranche pour limiter la taille téléchargée
    split = f'train[:{num_samples}]'

    ds, info = tfds.load(dataset_name, split=split, with_info=True, shuffle_files=True)
    print('Dataset chargé. Description rapide:')
    print(info)

    ds = tfds.as_numpy(ds)

    for i, example in enumerate(ds):
        # Rechercher une clé possible pour le nom de fichier
        fname: Optional[str] = None
        for k in example.keys():
            if 'filename' in k.lower() or 'file' in k.lower():
                v = example[k]
                if isinstance(v, (bytes, np.bytes_)):
                    fname = v.decode('utf-8')
                else:
                    fname = str(v)
                break

        if not fname:
            fname = f'image_{i}.png'

        # La clé standard pour l'image dans la plupart des datasets TFDS est 'image'
        img = example.get('image')
        if img is None:
            print(f"Aucune clé 'image' dans l'exemple {i} — clés disponibles: {list(example.keys())}")
            continue

        # img est un array uint8 HWC
        im = Image.fromarray(img)

        out_path = os.path.join(out_dir, fname)
        # Assurer l'extension png pour éviter problèmes
        if not out_path.lower().endswith('.png'):
            out_path = out_path + '.png'

        im.save(out_path)
        print(f"[{i+1}/{num_samples}] Sauvegardé -> {out_path}")


if __name__ == '__main__':
    # Exemple: télécharge 10 images et les met dans Code/open_images_samples/
    download_samples(num_samples=10, out_dir=os.path.join(os.path.dirname(__file__), 'open_images_samples'))
