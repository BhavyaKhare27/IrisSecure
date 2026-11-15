# IrisSecure: Five-Year Literature Review and R&D Plan

**Review window:** 6 November 2021–6 November 2026
**Prepared for:** 5th Semester Development / R&D Project
**Project stack:** Python, FastAPI, React/Next.js with JavaScript, Apache ECharts, SQLite/SQLAlchemy, statistics and optimization

## 1. Executive summary

Recent iris research has improved segmentation and recognition, but reported performance depends strongly on the task, dataset, sensor, and evaluation protocol. Presentation-attack detection (PAD) remains especially sensitive to unseen attack types and cross-domain shifts. Template protection also involves requirements beyond deleting the source image: transformed or encrypted templates have to remain useful for matching while addressing renewability, unlinkability, and key management.

IrisSecure should therefore make a narrow, testable contribution: **select a verification threshold from held-out genuine/impostor scores, either by balancing FAR/FRR (EER) or by minimizing FRR under a target FAR, then display the evidence and its provenance.** This directly fits the existing embedding pipeline and the requested optimization/statistics stack.

The app now saves pair-score distributions during empirical evaluation. The API uses those measured scores when available; otherwise it marks the threshold as a Gaussian estimate from aggregate reference statistics. It supports both EER and target-FAR objectives. This distinction matters: a threshold estimated from a few means and standard deviations is not a measured biometric result.

The current project does **not** implement iris segmentation, PAD/liveness, or a cancelable template scheme. Its evaluator now reports person-cluster bootstrap intervals for test FAR/FRR/TAR; zero observed false accepts still produce a degenerate bootstrap interval and must not be described as zero population risk. These unimplemented features require segmentation masks, spoof labels, or a template-protection evaluation protocol and must not be presented as completed features.

## 2. Review method

The attached brief's six references were checked against publisher, conference, or author-hosted records. The review was expanded with closely related work published within the same five-year window, including the later 2025 LivDet competition and 2026 preprints. The included items are a focused reading set, not an exhaustive or systematic review. Peer-reviewed papers, competition reports, standards, and preprints are identified separately below.

## 3. Research papers

| Work | Method and evidence | What it contributes | Caveat for IrisSecure |
|---|---|---|---|
| **Sheela & Radhika, 2022 — Multi-objective Artificial Bee Colony + autoencoder** ([Springer DOI](https://doi.org/10.1007/s11063-022-10775-z)) | Combines Daugman/Hough segmentation, DTCWT/Gabor/LBP/GLCM features, multi-objective feature selection, and an autoencoder classifier. The paper reports 99.67% accuracy on CASIA-Iris and 98.73% on IITD. | Shows how optimization can balance feature reduction with classification error. | These classification accuracies are not directly comparable with verification EER, TAR@FAR, or a different train/test protocol. IrisSecure should report its own metrics on its own held-out scores. |
| **Lee et al., 2022 — Alignment-robust cancelable iris verification (R-HoG)** ([IEEE DOI](https://doi.org/10.1109/TIFS.2022.3208812)) | Applies random augmentation and gradient-orientation grouping to transform IrisCode into an alignment-robust cancelable template; evaluates CASIA-IrisV3-Internal and CASIA-IrisV4-Thousand, including unlinkability/security properties. | Establishes that template transformation, revocability, and unlinkability are explicit evaluation goals. | R-HoG protects binary IrisCode. It cannot be assumed to protect IrisSecure's 512-float neural embedding without a new design and evaluation. |
| **Boyd et al., 2022 — State of the Art in Open-Set Iris PAD** ([arXiv](https://arxiv.org/abs/2208.10564)) | Builds a large PAD benchmark from 26 existing sources plus new data, totaling 450,000 images and seven attack types. Uses leave-one-attack-type-out evaluation and reports severe generalization failures for some algorithms that perform well in closed-set tests. | Makes unseen-attack evaluation central to PAD claims. | A bona fide CASIA recognition dataset cannot train or validate this capability; PAD requires labelled bona fide and attack samples with an explicit unseen-attack protocol. |
| **Nguyen, Proença & Alonso-Fernandez, 2024 — Deep Learning for Iris Recognition: A Survey** ([ACM DOI](https://doi.org/10.1145/3651306)) | Reviews more than 200 works across segmentation, recognition, PAD, forensics, tools, and open challenges. | Useful map of the pipeline and research trends; emphasizes that the recognition system is more than a CNN classifier. | It is a survey, not a new benchmark result. The 99.58% Iris-SAM number should not be generalized to recognition accuracy. |
| **Li et al., 2024 — Complex-Valued Gabor-Attention Residual Fusion Network** ([ECAI 2024 DOI](https://doi.org/10.3233/FAIA240487)) | Evaluates CASIA-Iris-Thousand and CASIA-Iris-Lamp with disjoint identities and reports EER, FRR at FAR=0.1%, and AUC. For CASIA-Iris-Thousand, the reported GRFN result is 0.68% EER, 2.59% FRR at 0.1% FAR, and 0.9997 AUC. | A same-dataset research reference stressing identity-disjoint evaluation and verification metrics. | IrisSecure now has a subject-disjoint result, but exact comparison still requires alignment of split construction, preprocessing, pair protocol, and threshold policy. The figures are not a direct leaderboard comparison. |
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

### Implemented research improvement and current results

- A reproducible **subject-disjoint split has been prepared** from the local CASIA-Iris-Thousand dataset: 700 people / 14,000 images / 1,400 person-eye classes for training; 150 / 3,000 / 300 for validation; and 150 / 3,000 / 300 for test. Both eyes and every image belonging to a person remain in the same subset. The split uses seed 42 and is stored locally under `data/splits/subject_disjoint/`.
- `train.py` now reads the subject-disjoint manifests, trains ArcFace classes only for the 1,400 training person-eye classes, evaluates validation verification scores on unseen identities, and selects the checkpoint by validation EER. It writes to a unique run directory so the existing deployed checkpoint is preserved.
- `evaluate.py` requires checkpoint metadata matching those manifests. It selected EER and target-FAR thresholds on validation, froze them, and reported held-out test results against fixed 0.75. It also computed subject-cluster bootstrap intervals and recorded dataset/manifest/checkpoint provenance.
- **Subject-disjoint evaluation completed:** best checkpoint is epoch 26 with validation EER 0.7673%. The EER threshold selected on validation was 0.373255. On test (150 people; 3,000 genuine and 50,000 sampled impostor pairs), FAR was 0.906%, FRR 0.867%, TAR 99.133%, and ROC-AUC 0.999346.
- At fixed threshold 0.75, test FAR was 0.006% (3/50,000), FRR 26.667%, and TAR 73.333%. The EER-selected operating point gains 25.80 percentage points of TAR while increasing observed FAR by 0.900 percentage points; this is an operating-point trade-off, not an unqualified improvement.
- Validation-selected target-FAR points also show test transfer: target 1% yielded test FAR 1.162%, FRR 0.767%, TAR 99.233%; target 0.1% yielded 0.138%, 3.800%, 96.200%; target 0.01% yielded 0.028%, 11.700%, 88.300%, respectively. Test rates can differ from validation targets.
- The 500-repetition subject-cluster bootstrap 95% intervals at the EER threshold were FAR 0.691%–1.174%, FRR 0.400%–1.433%, and TAR 98.567%–99.600%; at fixed 0.75 they were FAR 0%–0.0161%, FRR 22.467%–30.967%, and TAR 69.033%–77.533%. These intervals reflect 150 people and the sampled pairs, not all deployment populations.
- `api/analytics.py` trusts empirical calibration only when the report fingerprint matches the active checkpoint. Otherwise it labels the Gaussian aggregate-statistics fallback as an estimate. FastAPI background tasks and the React dashboard support EER balancing and target-FAR exploration locally.
- The dashboard and API provide enrollment and identification/verification workflows. A working local demo is a functional result, distinct from the measured subject-disjoint biometric evaluation.

### Protocol status and next experiment

1. Record dataset release, file count, sensor/modality, exclusions, code revision, checkpoint hash, and seed for the final experiment.
2. **Prepared:** use the reproducible person-disjoint manifests and verify no subject overlap. The split groups each person and both eyes together; repeated acquisition/session metadata are not available to define a stronger session-disjoint protocol.
3. **Completed:** train on the 700-person training set; best checkpoint was selected at epoch 26 using validation EER.
4. **Completed:** freeze validation-selected EER and target-FAR thresholds, then report the untouched 150-person test set against fixed 0.75. Pair scores reuse people/images, so 50,000 comparisons do not mean 50,000 independent identities.
5. **Completed:** compute 500 person-cluster bootstrap repetitions. Interpret intervals within this sample; do not treat observed FAR or an interval endpoint as a guarantee for other populations.
6. Compare with published results only after aligning dataset version, split protocol, task, and metric. The earlier image-level checkpoint and its generated artifacts are excluded from the cleaned project workspace.
7. Run liveness/PAD as a separate extension only after obtaining an appropriate LivDet or comparable labelled PAD dataset and using cross-attack evaluation. Do not train a “liveness” classifier on ordinary CASIA identity images.
8. Evaluate a protected/cancelable template as a separate security workstream against irreversibility, unlinkability, revocability, and matching performance. Do not relabel ordinary JSON/SQL embedding storage as privacy-preserving.

### Final subject-disjoint result summary

| Measure | Subject-disjoint result |
|---|---|
| Selected checkpoint / epoch / checkpoint SHA-256 | Epoch 26; `a90916232faa7a727871fb0467a5bfb2bed65376c5dcfcbf2a294d89d743e5ee` |
| Validation EER and selected threshold | 0.7673%; 0.373255 |
| Test ROC-AUC | 0.999346 |
| Test FAR / FRR / TAR at frozen validation-EER threshold | 0.906% / 0.867% / 99.133% |
| Test FAR / FRR / TAR at fixed threshold 0.75 | 0.006% / 26.667% / 73.333% |
| Test results at validation-selected target-FAR thresholds | Target 1%: FAR 1.162%, FRR 0.767%; target 0.1%: 0.138%, 3.800%; target 0.01%: 0.028%, 11.700% |
| Genuine/impostor pair counts and person-cluster 95% intervals | 3,000 / 50,000; 500 resamples over 150 people; see report above |

## 6. What the current project can and cannot claim

**Can claim now:** a working local research prototype; a completed subject-disjoint train/validation/test experiment with validation-selected thresholding, held-out test metrics, pair counts, and subject-cluster bootstrap intervals; and a local API/dashboard demo using the selected subject-disjoint checkpoint. Old checkpoint and identity-overlapping evaluation artifacts have been removed from the workspace.

**Cannot claim:** state-of-the-art performance or direct comparability with published results without aligned protocols, zero population FAR, deployment-grade biometric security, liveness/PAD, iris segmentation, cancelable templates, or compliance with ISO/IEC 24745. The app has no user authentication and does not encrypt stored embeddings at rest; it is intended for a local classroom/research demo and should not be exposed publicly. The Git repository excludes local datasets, checkpoints, and generated evaluation outputs.

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
13. Z. Li et al., “Complex-Valued Gabor-Attention Residual Fusion Network for Iris Recognition,” ECAI 2024. [doi:10.3233/FAIA240487](https://doi.org/10.3233/FAIA240487)
14. ISO/IEC 19795-1:2021, *Biometric performance testing and reporting — Part 1: Principles and framework*. [ISO catalogue entry](https://www.iso.org/standard/73515.html)
