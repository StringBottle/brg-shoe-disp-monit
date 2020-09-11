# Automated Anomaly Detection of Bridge Shoe Displacement 

This repository is for development of automated bridge shoe displacement anomaly detection system.

## Installation

```bash
conda env create -f environment.yml
```

## Usage

```python

from module.utils import imread
from module.disp_measure import convert_by_img

src_img = imread('image path of target at original position')
dest_img = imread('image path of target after moving')
displacement = displacement_measure(dest_img, src_img)  # returns numpy array with [x_axis_disp, y_axis_disp]
```
