# EGAnet-Python

**Exploratory Graph Analysis for Python**

A Python implementation of the [EGAnet R package](https://r-ega.net) for estimating dimensionality in multivariate data using network psychometrics and community detection.

## Overview

EGAnet-Python provides tools for:

- **Exploratory Graph Analysis (EGA)** - Estimate the number of dimensions using network methods
- **Network Estimation** - GLASSO and TMFG network construction
- **Community Detection** - Walktrap, Louvain, Leiden algorithms
- **Bootstrap Analysis** - Assess stability of dimensional structure
- **Information Theory** - Von Neumann entropy, TEFI, Jensen-Shannon divergence
- **Psychometric Tools** - Network loadings, scores, CFA comparison
- **Simulation** - Generate data with known factor structures

## Installation

```bash
pip install eganet
```

Or install from source:

```bash
git clone https://github.com/yourusername/eganet-python.git
cd eganet-python
pip install -e .
```

## Quick Start

```python
import eganet as ega

# Load example data
data = ega.load_wmt2()

# Run Exploratory Graph Analysis
result = ega.EGA(data, model="glasso", algorithm="walktrap")

print(f"Dimensions detected: {result.n_dim}")
print(f"Community memberships: {result.wc}")

# Compute network loadings
loadings = ega.net_loads(result)
print(loadings)

# Assess stability with bootstrap
boot_result = ega.boot_ega(data, n_boots=500)
```

## Core Functions

### Exploratory Graph Analysis

| Function | Description |
|----------|-------------|
| `EGA()` | Main EGA function |
| `boot_ega()` | Bootstrap EGA for stability |
| `hier_ega()` | Hierarchical EGA |
| `dyn_ega()` | Dynamic EGA for time series |
| `ri_ega()` | Random-intercept EGA |

### Network Methods

| Function | Description |
|----------|-------------|
| `glasso()` | Graphical LASSO network |
| `tmfg()` | Triangulated Maximally Filtered Graph |
| `community_detection()` | Community detection algorithms |
| `network_compare()` | Compare networks |

### Psychometrics

| Function | Description |
|----------|-------------|
| `net_loads()` | Network loadings |
| `net_scores()` | Network scores |
| `uva()` | Unique Variable Analysis |
| `cfa()` | Confirmatory Factor Analysis comparison |
| `invariance()` | Measurement invariance |

### Information Theory

| Function | Description |
|----------|-------------|
| `vn_entropy()` | Von Neumann entropy |
| `tefi()` | Total Entropy Fit Index |
| `jsd()` | Jensen-Shannon divergence |
| `ergo_info()` | Ergodicity information |

## Dependencies

- numpy >= 1.21
- scipy >= 1.7
- networkx >= 2.6
- pandas >= 1.3
- scikit-learn >= 1.0
- matplotlib >= 3.4
- seaborn >= 0.11
- joblib >= 1.0

## Citation

If you use EGAnet-Python in your research, please cite both the Python package and the original R package:

### Python Package

```bibtex
@software{eganet_python,
  title = {EGAnet-Python: Exploratory Graph Analysis for Python},
  author = {EGAnet Python Contributors},
  year = {2025},
  url = {https://github.com/yourusername/eganet-python}
}
```

### Original R Package

```bibtex
@manual{EGAnet,
  title = {EGAnet: Exploratory Graph Analysis – A framework for estimating the number of dimensions in multivariate data using network psychometrics},
  author = {Hudson Golino and Alexander P. Christensen},
  year = {2025},
  url = {https://r-ega.net},
  doi = {10.32614/CRAN.package.EGAnet}
}
```

### Key Publications

Golino, H., & Epskamp, S. (2017). Exploratory graph analysis: A new approach for estimating the number of dimensions in psychological research. *PLOS ONE, 12*(6), e0174035. https://doi.org/10.1371/journal.pone.0174035

Golino, H., Shi, D., Christensen, A. P., Garrido, L. E., Nieto, M. D., Sadana, R., ... & Martinez-Molina, A. (2020). Investigating the performance of exploratory graph analysis and traditional techniques to identify the number of latent factors: A simulation and tutorial. *Psychological Methods, 25*(3), 292-320. https://doi.org/10.1037/met0000255

## Credits

This Python implementation is based on the [EGAnet R package](https://github.com/hfgolino/EGAnet) developed by:

- **Hudson Golino** (University of Virginia) - Original author and maintainer
- **Alexander P. Christensen** - Author and contributor

Additional R package contributors:
- Robert Moulder
- Luis E. Garrido
- Laura Jamison
- Dingjing Shi

## License

AGPL-3.0-or-later

This package is licensed under the GNU Affero General Public License v3.0 or later, consistent with the original R package.

## Links

- [Original R Package](https://github.com/hfgolino/EGAnet)
- [R-EGA Website](https://r-ega.net)
- [CRAN Package](https://cran.r-project.org/package=EGAnet)
