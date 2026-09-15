# karpos-downscaling

**Atmospheric reanalysis downscaling with sparse in-situ calibration, for
parametric agricultural insurance.**

`karpos-downscaling` is an open-source Python package
([Apache 2.0](https://github.com/maurinl26/karpos-downscaling/blob/main/LICENSE))
that downscales atmospheric reanalyses (ERA5, CERRA) to kilometer-scale
resolution. It supports three families of methods behind a single
configuration:

1. **Statistical** — lapse-rate + quantile mapping + RBF Sencrop residual
   correction (CPU pipeline)
2. **Deep learning** — U-Net with FiLM conditioning on DEM and synoptic regime
   (PyTorch Lightning)
3. **Foundation model** — Prithvi WxC (NASA/IBM) backbone with adapter layers

The output is calibrated against sparse in-situ sensor networks (Sencrop) and
stratified by synoptic regime (radiative, advective, cold pool, etc.) for
interpretable performance reporting. Applications include parametric
agricultural insurance indices (frost, hail, extreme heat, hydric stress) on
fruit orchards and hillside vineyards.

## Quick links

- [GitHub repository](https://github.com/maurinl26/karpos-downscaling)
- [Issue tracker](https://github.com/maurinl26/karpos-downscaling/issues)
- [Contributing guidelines](https://github.com/maurinl26/karpos-downscaling/blob/main/CONTRIBUTING.md)
- [Code of conduct](https://github.com/maurinl26/karpos-downscaling/blob/main/CODE_OF_CONDUCT.md)

## Documentation

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/installation
getting-started/quickstart
getting-started/examples
```

```{toctree}
:maxdepth: 2
:caption: User guide

user-guide/statistical-pipeline
user-guide/dl-film-pipeline
user-guide/prithvi-pipeline
user-guide/regime-stratification
user-guide/calibration
user-guide/counterfactual
```

```{toctree}
:maxdepth: 2
:caption: Methodology

methodology/README
methodology/chilling-model-baronnies
methodology/karpos-slr-calibration-report
methodology/gmd-paper-draft
methodology/joss-paper-draft
```

```{toctree}
:maxdepth: 1
:caption: Architecture and infrastructure

architecture
infra/infra_pro
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Community

community/contributing
community/code-of-conduct
community/security
community/docstring-conventions
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
