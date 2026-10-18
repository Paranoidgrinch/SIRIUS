# SIRIUS

SIRIUS is the automated ion-beam tuning and transmission optimization system for the
[FLAVIA](https://github.com/Paranoidgrinch/FLAVIA-alpha-1.1-Fully-integrated-Laser-Anion-Variables-for-AMS-)-controlled
laser-AMS beamline at CologneAMS. It tunes the beam stage by stage from the ion source to the last Faraday
cup, with physics models of each section as the starting point and RCDS optimization for the fine tuning.
For the source section alone, see the compact [mcsirius](https://github.com/Paranoidgrinch/mcsirius).

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-in_development-D29922?style=flat-square)
[![License](https://img.shields.io/badge/license-GPL--3.0-394B59?style=flat-square)](LICENSE)

## Design principles

- FLAVIA remains the hardware communication backend.
- SIRIUS performs physics-informed sequential optimization.
- Cup 1 is the source-current reference and represents 100 % transmission.
- The Cup 1 reference state is rechecked every 10 minutes during optimization.
- Final transmission measurements use one frozen final machine state without refocusing between cups.
- All tested settings, measurements, uncertainties, optimizer decisions and machine states are logged.
- Successful SIRIUS states can be exported as FLAVIA-compatible configurations.
- Useful operating regions are learned and stored by ion mass.

## Beamline optimization stages

1. Cup 1: source, extraction, einzel lens and analyzing magnet
2. Cup 2: lens 2 and steerer X1/Y1
3. Cup 3: ion cooler, deceleration/acceleration electrodes, guide field and RFQ
4. Cup 4: quadrupole triplet and steerer X2/Y2
5. Cup 5: ESA
6. Cup 6: lens 4 and steerer X3/Y3

## Getting started

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

SIRIUS drives the machine through an already running FLAVIA backend (`sirius.flavia_adapter`); FLAVIA remains
responsible for every device connection. Run records are written to `data/runs/`, learned operating regions
per ion mass to `data/mass_profiles/`.

## Structure

| Modules | Role |
| --- | --- |
| `orchestrator`, `default_stages`, `runtime`, `run_config` | stage sequence Cup 1 → Cup 6 → final characterization, run setup and preflight |
| `cup1_optimizer` … `cup6_optimizer`, `rcds_optimizer`, `scan1d`, `qpt_scan2d` | per-stage optimizers, RCDS and scans |
| `*_model` (magnet, cooler, RFQ, QPT, ESA, guide field), `physics` | physics models that seed each stage |
| `hardware_safety`, `hardware_guard`, `rfq_safety`, `cologneams_safety`, `safe_transition` | limits, guarded transitions and machine-specific safety rules |
| `measurement`, `readback_quality`, `readback_freshness`, `settling` | measurement with settling, noise and stale-readback checks |
| `reference`, `reference_orchestrator`, `final_characterization` | Cup 1 reference checks and the frozen final transmission measurement |
| `mass_profile`, `range_learning`, `run_logging` | learned ranges per mass and full run logs |

## License

GPL-3.0 — see [LICENSE](LICENSE).
