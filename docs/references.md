# References

The external methods this project builds on, with persistent identifiers. Every DOI
below was verified against the publisher record before inclusion. Citations like [R1]
are used at the point of need throughout [architecture](architecture.md) and
[evaluation](evaluation.md).

## Reference list

**[R1] LDM / Autoencoder latent**. R. Rombach, A. Blattmann, D. Lorenz, P. Esser,
B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models", Proc.
IEEE/CVF CVPR 2022, pp. 10684–10695.
<https://doi.org/10.1109/CVPR52688.2022.01060>

*Where used:* the overall autoencoder design (`src/model.py`), ResBlocks with
zero-init final conv, KL-regularized latent with double-z encoder output, latent-space
ratio as the compression knob, attention in the mid block, nearest+conv decoder
upsampling.

**[R2] Taming Transformers**. P. Esser, R. Rombach, B. Ommer, "Taming Transformers
for High-Resolution Image Synthesis", Proc. IEEE/CVF CVPR 2021, pp. 12868–12878.
<https://doi.org/10.1109/CVPR46437.2021.01268>

*Where used:* the optional Phase 8 PatchGAN (`src/gan.py`, `src/train_gan.py`), the
NLayerDiscriminator, hinge loss, and the adaptive generator weighting scheme. Off by
default; kept only if LPIPS and descriptors both improve.

**[R3] MCRpy**. P. Seibert, J. Scholz, F. Welschof, K. Schladitz, S. O. Stephan,
"Microstructure Characterization and Reconstruction in Python: MCRpy", *Integrating
Materials and Manufacturing Innovation* 11 (2022) 450–466.
<https://doi.org/10.1007/s40192-022-00273-4>

*Where used:* the success-criterion measurement (`scripts/descriptors.py`):
VolumeFractions, Two-Point Correlations (S2) and LinealPath descriptors computed on
original/reconstruction pairs. GitHub: <https://github.com/NEFM-TUDresden/MCRpy>.
Runs in a separate process from torch (TensorFlow + torch cannot share 8 GB).

**[R4] MS-SSIM**. Z. Wang, E. P. Simoncelli, A. C. Bovik, "Multiscale Structural
Similarity for Image Quality Assessment", Proc. IEEE Asilomar Conf. on Signals,
Systems and Computers, 2003, pp. 1398–1402.
<https://doi.org/10.1109/ACSSC.2003.1292216>

*Where used:* the structural term of the loss (`src/losses.py`, weight 0.25) and as a
reported evaluation metric. Implemented via `pytorch_msssim`, with the window adapted
to input size ((win−1)·16 < size).

**[R5] LPIPS**. R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, "The
Unreasonable Effectiveness of Deep Features as a Perceptual Metric", Proc. IEEE/CVF
CVPR 2018, pp. 586–595. <https://doi.org/10.1109/CVPR.2018.00068>

*Where used:* the perceptual term of the loss (`src/losses.py`, weight 1.0, AlexNet
backbone) and as the checkpoint-selection metric (best val LPIPS). Grayscale is
replicated to 3 channels.

## Verification note

DOI links were resolved against publisher records (IEEE Xplore, Springer) during
documentation. If a link fails, search the DOI directly at <https://doi.org>. The
arXiv preprint of [R1] is <https://arxiv.org/abs/2112.10752>.

## In-repo measurement provenance

External papers define *methods*; every *number* in this documentation is measured
in-repo and traces to `reports/phase<N>_<name>.json`, see the
[reports index](reports.md) and the mapping table in
[results](results.md#success-criterion).
