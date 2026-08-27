# Reproducibility audit - issues to understand before publication

## 1. AMP normalization (high priority)
The generator in `kamp_complete.py` uses Gaussian entries scaled by `1/sqrt(n)`. Classical AMP presentations commonly normalize entries with variance `1/m`. The very large AMP errors in the supplied benchmark may therefore partly reflect an implementation/normalization mismatch. The manuscript has been revised so those errors are reported as behavior of the supplied implementation, not a general theoretical claim about AMP. A stronger revision would rerun all experiments after validating/tuning the AMP baseline under its conventional scaling.

## 2. R14 is not direct posterior-covariance active sensing
The supplied R14 code uses `P1 = inv(A_S^T A_S + eps I)` for row scoring; it does not use KAMP's propagated posterior `P_t`. The manuscript now describes R14 as a PFIM information-guided proxy and explicitly states that direct KAMP-posterior sensing remains untested.

## 3. R15 is Jakes-inspired, not a full Jakes simulator
The development code generates an AR(1) channel with coefficient `a = 1 - 2(pi f_d T_s)^2`; it does not synthesize a separate full Jakes Doppler spectrum. The manuscript now calls this an application-motivated Jakes-inspired AR(1) stress test.

## 4. NEES statistic is diagonal-only
R13 computes the mean of coordinate-wise squared errors divided by diagonal covariance entries. This is not the full quadratic-form NEES when off-diagonal covariance is nonzero. The manuscript now calls it a diagonal NEES proxy.

## 5. Repository URL
The public repository URL is set to https://github.com/hadisadoghiyazdi1971/kamp-mssp-reproducibility. Push the contents of this folder to that repository before manuscript submission and preserve a release/tag for the submitted version.
