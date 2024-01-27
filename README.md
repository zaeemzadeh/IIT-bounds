# Upper bounds for integrated information
Implementation of the bounds and calculations presented in:

Zaeemzadeh, Alireza, and Giulio Tononi. "Upper bounds for integrated information." (2023). [link](https://arxiv.org/pdf/2305.09826.pdf)


## Setup
```bash
conda create -n pyphi python=3.10
conda activate pyphi
python -m pip install -U git+https://github.com/wmayner/pyphi.git@feature/iit-4.0
conda install ipykernel ipywidgets matplotlib=3.7
```
See `bound_figures.ipynb` for examples and results.

## Citing this work
If you use this work in your research, please use the following BibTeX entry.
```
@article{zaeemzadeh2023upper,
  title={Upper bounds for integrated information},
  author={Zaeemzadeh, Alireza and Tononi, Giulio},
  journal={arXiv preprint arXiv:2305.09826},
  year={2023}
}
```