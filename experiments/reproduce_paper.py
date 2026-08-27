"""Paper-facing dispatcher for the KAMP reproducibility package.

The shared development suite contains two DKAMP-only studies numbered R15/R16.
Those studies are outside the manuscript.  This wrapper exposes the manuscript's
R01-R15 numbering and maps manuscript R15 to the legacy function R17.
"""
from pathlib import Path
import sys, shutil, argparse
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(HERE))
import kamp_regime_suite as suite

PAPER = {
 'R01': suite.experiment_r01_static_benchmark,
 'R02': suite.experiment_r02_phase_transition,
 'R03': suite.experiment_r03_statistical,
 'R04': suite.experiment_r04_correlated_measurements,
 'R05': suite.experiment_r05_heteroscedastic_noise,
 'R06': suite.experiment_r06_heavy_tailed_noise,
 'R07': suite.experiment_r07_colored_noise,
 'R08': suite.experiment_r08_dynamic_tracking,
 'R09': suite.experiment_r09_change_point,
 'R10': suite.experiment_r10_graph_prior,
 'R11': suite.experiment_r11_onebit,
 'R12': suite.experiment_r12_pcrlb,
 'R13': suite.experiment_r13_nees_nis,
 'R14': suite.experiment_r14_active_sensing,
 'R15': suite.experiment_r17_jakes_channel,
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=list(PAPER))
    ap.add_argument('--fast', action='store_true')
    args=ap.parse_args()
    if args.fast: suite.FAST=True
    for key in args.only:
        if key not in PAPER: raise SystemExit(f'Unknown paper experiment: {key}')
        PAPER[key]()
    print('Completed:', ', '.join(args.only))

if __name__=='__main__': main()
