# Observation-Level Causality-Guided Graph Structure Learning

This repository contains the implementation associated with the paper:

> **Observation-level causality-guided graph structure learning for reliable community detection in complex information networks**  
> Yuxian Ke, Qizhang Li, Yi Sun, Chaoyu Yang, Tan Zhang, Hongrui Zhang, Tao Deng, Limengzi Yuan, and Dongqin Zhu.  
> *Information Sciences*, 2026, Article 123850.

The code constructs an inverse-probability-weighted (IPW) graph view, performs community inference using the observed and corrected graph views, and evaluates the inferred communities.

## Project structure

```text
.
├── data/
│   ├── input/
│   │   └── EmailEU/
│   │       └── raw/
│   │           └── EmailEU_raw.pt
│   ├── output/
│   └── make_ipw_layer.py
├── src/
│   ├── model.py
│   └── setting_inference.yaml
├── main_inference.py
├── eval.py
├── requirements.txt
└── README.md
```


## Running the code

Run all commands from the project root directory and strictly follow this order:

```text
make_ipw_layer.py → main_inference.py → eval.py
```

### 1. Construct the IPW-corrected graph layer

```bash
python data/make_ipw_layer.py
```


### 2. Run community inference

```bash
python main_inference.py
```

Here, `K` is the number of communities. The main optional arguments are:

| Argument | Description | Default |
|---|---|---:|
| `-f`, `--in_folder` | Input data directory | `data/input/EmailEU/corrected` |
| `-d`, `--data_file` | Corrected data filename | `EmailEU_ipw.pt` |
| `-K`, `--K` | Number of communities | `42` |
| `--diffusion_alpha` | Diffusion coefficient | Value in the YAML configuration |
| `--obs_weight` | Weight assigned to the observed graph view | Value in the YAML configuration |
| `--lambda1` | Variation sparsity coefficient | Value in the YAML configuration |
| `--lambda2` | Centering constraint coefficient | Value in the YAML configuration |

The remaining settings are defined in `src/setting_inference.yaml`. Results are written to:

```text
data/output/main/EmailEU/corrected/
```

### 3. Evaluate the inferred communities

After the inference step has completed successfully, run:

```bash
python eval.py
```

The evaluation reports:

- Normalized Mutual Information (NMI)
- Adjusted Rand Index (ARI)
- Modularity for each graph layer
- Mean modularity across graph layers

## Citation

If this code is useful in your research, please cite:

```bibtex
@article{ke2026observation,
  title     = {Observation-level causality-guided graph structure learning for reliable community detection in complex information networks},
  author    = {Ke, Yuxian and Li, Qizhang and Sun, Yi and Yang, Chaoyu and Zhang, Tan and Zhang, Hongrui and Deng, Tao and Yuan, Limengzi and Zhu, Dongqin},
  journal   = {Information Sciences},
  pages     = {123850},
  year      = {2026},
  publisher = {Elsevier}
}
```
