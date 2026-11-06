# IrisSecure: Five-Year Literature Review and R&D Plan

**Review window:** 6 November 2021–6 November 2026
**Prepared for:** 5th Semester Development / R&D Project
**Project stack:** Python, FastAPI, Next.js/React with TypeScript, Apache ECharts, SQLite/SQLAlchemy, statistics and optimization

## 1. Executive summary

Recent iris research has improved segmentation and recognition, but reported performance depends strongly on the task, dataset, sensor, and evaluation protocol. Presentation-attack detection (PAD) remains especially sensitive to unseen attack types and cross-domain shifts. Template protection also involves requirements beyond deleting the source image: transformed or encrypted templates have to remain useful for matching while addressing renewability, unlinkability, and key management.

IrisSecure should therefore make a narrow, testable contribution: **select a verification threshold from held-out genuine/impostor scores, either by balancing FAR/FRR (EER) or by minimizing FRR under a target FAR, then display the evidence and its provenance.** This directly fits the existing embedding pipeline and the requested optimization/statistics stack.

The app now saves pair-score distributions during empirical evaluation. The API uses those measured scores when available; otherwise it marks the threshold as a Gaussian estimate from aggregate reference statistics. It supports both EER and target-FAR objectives. This distinction matters: a threshold estimated from a few means and standard deviations is not a measured biometric result.

The current project does **not** implement iris segmentation, PAD/liveness, confidence intervals, or a cancelable template scheme. Those are valid future directions, but this repository does not include their required masks, spoof labels, or protection evaluation protocol. They must not be presented as completed features.

## 2. Review method

The attached brief's six references were checked against publisher, conference, or author-hosted records. The review was expanded with closely related work published within the same five-year window, including the later 2025 LivDet competition and 2026 preprints. The included items are a focused reading set, not an exhaustive or systematic review. Peer-reviewed papers, competition reports, standards, and preprints are identified separately below.

## 3. Research papers

| Work | Method and evidence | What it contributes | Caveat for IrisSecure |
|---|---|---|---|
| **Sheela & Radhika, 2022 — Multi-objective Artificial Bee Colony + autoencoder** ([Springer DOI](https://doi.org/10.1007/s11063-022-10775-z)) | Combines Daugman/Hough segmentation, DTCWT/Gabor/LBP/GLCM features, multi-objective feature selection, and an autoencoder classifier. The paper reports 99.67% accuracy on CASIA-Iris and 98.73% on IITD. | Shows how optimization can balance feature reduction with classification error. | These classification accuracies are not directly comparable with verification EER, TAR@FAR, or a different train/test protocol. IrisSecure should report its own metrics on its own held-out scores. |
| **Lee et al., 2022 — Alignment-robust cancelable iris verification (R-HoG)** ([IEEE DOI](https://doi.org/10.1109/TIFS.2022.3208812)) | Applies random augmentation and gradient-orientation grouping to transform IrisCode into an alignment-robust cancelable template; evaluates CASIA-IrisV3-Internal and CASIA-IrisV4-Thousand, including unlinkability/security properties. | Establishes that template transformation, revocability, and unlinkability are explicit evaluation goals. | R-HoG protects binary IrisCode. It cannot be assumed to protect IrisSecure's 512-float neural embedding without a new design and evaluation. |
| **Boyd et al., 2022 — State of the Art in Open-Set Iris PAD** ([arXiv](https://arxiv.org/abs/2208.10564)) | Builds a large PAD benchmark from 26 existing sources plus new data, totaling 450,000 images and seven attack types. Uses leave-one-attack-type-out evaluation and reports severe generalization failures for some algorithms that perform well in closed-set tests. | Makes unseen-attack evaluation central to PAD claims. | A bona fide CASIA recognition dataset cannot train or validate this capability; PAD requires labelled bona fide and attack samples with an explicit unseen-attack protocol. |
| **Nguyen, Proença & Alonso-Fernandez, 2024 — Deep Learning for Iris Recognition: A Survey** ([ACM DOI](https://doi.org/10.1145/3651306)) | Reviews more than 200 works across segmentation, recognition, PAD, forensics, tools, and open challenges. | Useful map of the pipeline and research trends; emphasizes that the recognition system is more than a CNN classifier. | It is a survey, not a new benchmark result. The 99.58% Iris-SAM number should not be generalized to recognition accuracy. |
| **Farmanifard & Ross, 2024 — Iris-SAM** ([arXiv preprint](https://arxiv.org/abs/2402.06497)) | Fine-tunes Segment Anything for pixel-level iris segmentation with loss functions including focal loss. Reports average segmentation accuracy of 99.58% on ND-IRIS-0405 versus 89.75% for the cited best baseline. | Supports the value of explicit segmentation and attention to class imbalance. | Pixel-level segmentation accuracy is a different task and metric from identity verification. The foundation model is heavier and needs segmentation masks for training. |
| **Mitcheff et al., 2024 — Privacy-Safe Iris PAD** ([IEEE DOI](https://doi.org/10.1109/IJCB62174.2024.10744455), [arXiv](https://arxiv.org/abs/2408.02750)) | Trains PAD using synthetic, identity-leakage-screened bona fide iris images and synthetic textured-contact-lens attacks; reports a performance gap compared with models trained on real iris captures. | Offers a direction for reducing exposure of identity-bearing training images while studying PAD. | Synthetic training is not automatically equivalent to real capture conditions. The authors still evaluate on established PAD benchmarks. |
| **Tinsley et al., 2023 — LivDet-Iris 2023** ([IJCB DOI](https://doi.org/10.1109/IJCB57857.2023.10448637), [arXiv](https://arxiv.org/abs/2310.04541)) | Competition report includes seven PAI categories, including GAN-generated images, and eight submitted algorithms. Best weighted ACER was 37.31%; the unweighted winner reported 22.15%. | Demonstrates that conclusions depend on attack weighting and that digital/synthetic attacks belong in evaluation. | The brief cites only 37.31% and calls it the best error without the weighting context or the alternate winner. Avoid presenting one number as a universal PAD error rate. |
| **Mitcheff et al., 2025 — LivDet-Iris 2025** ([organizer preprint](https://livdet-iris.org/2025/livdet-iris-2025-ijcb-preprint.pdf), [official competition repository](https://github.com/CVRL/livdet-iris-2025)) | Adds separate tasks for an industry benchmark, morphed iris images, and advanced textured-contact-lens attacks, plus systematic testing of a commercial hardware/software system. The winning team's reported AUCs were 90.57%, 68.23%, and 99.99% across the three algorithm tasks. | Updates the brief's 2023-only view and shows that PAD outcomes vary substantially by attack/task. | The three AUC values describe different tasks and are not a single comparable liveness score. Some competition data require access agreements. |
| **Karakosta, Alhedaithy & Knottenbelt, 2026 — Privacy-Preserving Iris Recognition** ([arXiv preprint](https://arxiv.org/abs/2603.26890)) | Evaluates an FHE-based iris-code comparison framework aligned with biometric information protection requirements; reports accuracy close to cleartext matching and about 120,000× pairwise comparison overhead in its evaluated setup. | Makes the privacy/compute trade-off concrete and motivates scalable two-level matching. | This is a 2026 preprint, not a universal FHE cost estimate. Its overhead is specific to its implementation, data, and comparison setup. |
| **Dowling et al., 2026 — VISER for open-set iris PAD** ([arXiv preprint](https://arxiv.org/abs/2603.17859)) | Studies saliency signals and DINOv2 features in leave-one-attack-type-out PAD; reports gains from denoised eye-tracking heatmaps over a cross-entropy baseline. | A very recent direction for generalization to attack types absent during training. | A preprint and specialized PAD study; not a drop-in liveness detector for IrisSecure. |
| **Diff-IrisPAD, 2026 — Unsupervised PAD** ([arXiv preprint](https://arxiv.org/abs/2610.29611)) | Models bona fide score/image distributions for unsupervised PAD; its preprint reports an 8-attack, multi-sensor evaluation with 30.52% overall ACER. | Shows that open-set and limited-label PAD remains active research in late 2026. | Very recent preprint evidence; results require independent review and reproduction before comparison with competition results. |

### Standards context

[ISO/IEC 24745:2022](https://www.iso.org/standard/75302.html) describes biometric information protection, including confidentiality, integrity, and renewability/revocability considerations. Storing only an embedding is not, by itself, evidence that the embedding is irreversible, unlinkable, or safely revocable. IrisSecure currently stores the API embedding in its database and has not implemented a cancelable-template scheme.

## 4. Synthesis and research gap

1. **Evaluation protocol matters as much as architecture.** Closed-set classification accuracy cannot substitute for verification metrics. PAD performance on known attacks cannot establish performance on unseen attacks.
2. **Segmentation and quality filtering are real pipeline stages.** The current preprocessing uses grayscale conversion, blur, CLAHE, and resize; it does not produce or consume iris masks.
3. **Optimization must operate on observed validation scores.** A Gaussian approximation from aggregate summary values is useful as a fallback/demo, but it cannot support a claim of measured EER or a certified FAR operating point.
4. **Privacy claims need an explicit protection threat model.** Not retaining the source image reduces one exposure, but a biometric embedding is itself sensitive information. Encryption, cancelability, unlinkability, revocation, and key management are distinct questions.
5. **PAD requires a separate dataset and protocol.** CASIA-Iris-Thousand does not supply the labelled PAD coverage needed to substantiate liveness claims. The LivDet papers show why webcam capture plus a recognition model is not a liveness system.

## 5. Proposed R&D project

### Research question

On a held-out CASIA-Iris-Thousand evaluation split, how does selecting an operating threshold by (a) equal error rate or (b) a maximum target FAR compare with the fixed 0.75 threshold in terms of FAR, FRR, EER, and TAR at chosen FAR levels?

### Hypothesis

A threshold selected from held-out genuine/impostor scores will provide a more explicit and reproducible operating point than the hard-coded 0.75 default. The target-FAR mode will reduce false accepts to the selected empirical constraint when that constraint is supported by enough impostor comparisons, at the cost of increased false rejects.

### Independent variable

- Fixed threshold: 0.75
- Equal-error operating threshold
- Target-FAR operating thresholds: 1%, 0.1%, and 0.01%

### Outcomes

- FAR, FRR, EER, ROC AUC
- TAR at selected FAR values
- Number of genuine/impostor pairs supporting each metric
- Threshold chosen and whether the requested FAR was met
- Optional API/job latency as an engineering measure (not a biometric accuracy metric)

### Existing implementation in this repository

- `evaluate.py` computes held-out genuine/impostor scores and now saves them as `outputs/evaluation/score_distributions.npz`.
- `api/analytics.py` uses these empirical score distributions when present and marks the source as empirical. Before evaluation data exist, it uses aggregate Gaussian statistics and marks that output as an estimate.
- FastAPI background tasks run threshold optimization locally. The API accepts EER balancing or a target-FAR constraint; the Next.js dashboard exposes those operating modes and charts FAR/FRR.
- SQLAlchemy stores the gallery; the dashboard handles enrollment and identification/verification when a trained checkpoint is installed.

### Experimental protocol to complete

1. Record exact dataset release, file count, sensor/modality, excluded files, code commit, model checkpoint hash, and seed.
2. Keep a reproducible subject-eye stratified train/validation/test split for closed-set recognition, but group repeated acquisitions/sessions when metadata allows. Report clearly that train/test share identity classes; do not call it identity-disjoint generalization.
3. Fit/select operating thresholds using the validation split. Report final metrics once on the untouched test split. Do not tune on test scores and then report the same test as an unbiased result.
4. Compare the three threshold policies above, with pair counts and the exact target FAR. At very small FAR, the impostor sample count limits what can be claimed; for example, 0 observed false accepts among a finite sample does not prove a zero population FAR.
5. Add subject-cluster confidence intervals before making population-level uncertainty claims. Pair scores reuse images and are correlated; a naive bootstrap over pairs would overstate the effective sample size.
6. Run liveness/PAD as a separate extension only after obtaining an appropriate LivDet or comparable labelled PAD dataset and using cross-attack evaluation. Do not train a “liveness” classifier on ordinary CASIA identity images.
7. Evaluate a protected/cancelable template as a separate security workstream against irreversibility, unlinkability, revocability, and matching performance. Do not relabel ordinary JSON/SQL embedding storage as privacy-preserving.

## 6. What the current project can and cannot claim

**Can claim:** a working research prototype with a learned iris embedding model, API/web interface, gallery storage, empirical score-based threshold selection after running evaluation, and a clearly labelled aggregate-statistic fallback.

**Cannot yet claim:** state-of-the-art accuracy, deployment-grade biometric security, liveness/PAD, iris segmentation, statistically independent confidence intervals, cancelable templates, or compliance with ISO/IEC 24745. The repository currently lacks a checkpoint, the CASIA dataset, LivDet labels, segmentation masks, and generated empirical test scores.

## 7. References

1. K. Nguyen, H. Proença, and F. Alonso-Fernandez, “Deep Learning for Iris Recognition: A Survey,” *ACM Computing Surveys*, 56(9), 2024. [doi:10.1145/3651306](https://doi.org/10.1145/3651306)
2. S. V. Sheela and K. R. Radhika, “Iris Recognition using Multi Objective Artificial Bee Colony Optimization Algorithm with Autoencoder Classifier,” *Neural Processing Letters*, 54, 3489–3505, 2022. [doi:10.1007/s11063-022-10775-z](https://doi.org/10.1007/s11063-022-10775-z)
3. M. J. Lee, Z. Jin, S.-N. Liang, and M. Tistarelli, “Alignment-Robust Cancelable Biometric Scheme for Iris Verification,” *IEEE TIFS*, 17, 3449–3464, 2022. [doi:10.1109/TIFS.2022.3208812](https://doi.org/10.1109/TIFS.2022.3208812)
4. A. Boyd et al., “State of the Art in Open-Set Iris Presentation Attack Detection,” 2022. [arXiv:2208.10564](https://arxiv.org/abs/2208.10564)
5. P. Tinsley et al., “Iris Liveness Detection Competition (LivDet-Iris) — The 2023 Edition,” *IJCB*, 2023. [doi:10.1109/IJCB57857.2023.10448637](https://doi.org/10.1109/IJCB57857.2023.10448637)
6. P. Farmanifard and A. Ross, “Iris-SAM: Iris Segmentation Using a Foundation Model,” 2024. [arXiv:2402.06497](https://arxiv.org/abs/2402.06497)
7. M. Mitcheff, P. Tinsley, and A. Czajka, “Privacy-Safe Iris Presentation Attack Detection,” *IJCB*, 2024. [doi:10.1109/IJCB62174.2024.10744455](https://doi.org/10.1109/IJCB62174.2024.10744455)
8. M. Mitcheff et al., “Iris Liveness Detection Competition (LivDet-Iris) — The 2025 Edition,” *IJCB*, 2025. [organizer preprint](https://livdet-iris.org/2025/livdet-iris-2025-ijcb-preprint.pdf)
9. C. Karakosta, L. Alhedaithy, and W. J. Knottenbelt, “Privacy-Preserving Iris Recognition: Performance Challenges and Outlook,” 2026 preprint. [arXiv:2603.26890](https://arxiv.org/abs/2603.26890)
10. B. Dowling et al., “VISER: Visually-Informed System for Enhanced Robustness in Open-Set Iris Presentation Attack Detection,” 2026 preprint. [arXiv:2603.17859](https://arxiv.org/abs/2603.17859)
11. “Diff-IrisPAD: Modeling Bona Fide Iris Distributions for Unsupervised Presentation Attack Detection,” 2026 preprint. [arXiv:2610.29611](https://arxiv.org/abs/2610.29611)
12. ISO/IEC 24745:2022, *Biometric information protection*. [ISO catalogue entry](https://www.iso.org/standard/75302.html)
