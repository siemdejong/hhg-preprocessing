# hhg-preprocessing
Preprocessing pipeline specifically for higher harmonic generation images for use with AI.

Currently, this repo implements
- Tissue segmentation

## Install
```
git clone https://github.com/siemdejong/hhg-preprocessing
cd hhg-preprocessing
pip install -e .
```

## Tissue segmentation
To use the reimplemented EntropyMasker tissue segmentation algorithm
```python
import numpy as np
from PIL import Image
from hhg_preprocess.entropy_masker.entropy_masker import entropy_masker
image = np.array(Image.open("path/to/your/image.png"))
mask = entropy_masker(image)
```
or use the CLI:
```
entropy_masker --data <data/data.txt> --output_dir <output_dir>
```
where \<data/data.txt> is a text file with input filenames separated by newlines.
If desired, one can change the default settings in `hhg_preprocess.entropy_masker.config.default.py`.

## Data Version Control
To start a DVC pipeline, populate \<data/data.txt> with filenames and execute
```
dvc repro
```
For more on DVC, refer to the [DVC documentation](https://dvc.org/doc)
