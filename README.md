# SIRIUS

SIRIUS is the automated ion-beam tuning and transmission optimization system for the FLAVIA-controlled beamline.

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
