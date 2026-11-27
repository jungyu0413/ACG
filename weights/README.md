# Weights

`scripts/train.py <fold>` writes `weights/acg_<fold>_seed<S>.pth`, and `scripts/generate.py <fold>` loads that file
unless `--weights <path>` is given. Pretrained weights for the published rows will be released on the
[release page](https://github.com/jungyu0413/ACG/releases); place them here under the same names.
