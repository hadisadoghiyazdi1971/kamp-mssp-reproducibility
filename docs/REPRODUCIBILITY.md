# Reproducibility protocol

1. Create a clean Python 3.11 environment.
2. Install `requirements.txt`.
3. Run `python experiments/reproduce_paper.py --only R01` as a smoke test.
4. Run `python experiments/reproduce_paper.py` for the complete experiment suite.
5. Compare generated outputs with `results/paper_data/` and manuscript figures.

The fixed random seeds are defined in the supplied experiment suite. Runtimes are machine-dependent and should not be expected to match exactly. Statistical conclusions should be regenerated from raw trial outputs when the experimental pipeline is changed.
