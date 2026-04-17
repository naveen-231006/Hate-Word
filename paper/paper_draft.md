# Hate Speech Detection in Tamil Social Media Text Using Multilingual Transformers: A Comparative Study

## Abstract

The proliferation of hate speech on social media platforms poses a significant challenge, particularly in low-resource languages such as Tamil. While extensive research has been conducted on English hate speech detection, regional Indian languages remain critically underexplored. This paper presents a comprehensive comparative study of three multilingual transformer models — MuRIL, XLM-RoBERTa, and mBERT — for detecting offensive language in Tamil and code-mixed Tamil-English (Tanglish) social media text. We fine-tune these models on the OffensEval Dravidian dataset (35,138 training samples across 6 classes) with inverse-frequency weighted loss to address severe class imbalance. Our experiments demonstrate that mBERT achieves the highest individual weighted F1-score of 0.7505, and a majority-voting ensemble of all three models further improves this to 0.7620. Contrary to expectations, Indian-language-specific pre-training (MuRIL) did not yield superior test-set performance, suggesting that general multilingual pre-training with broader linguistic coverage may be equally effective for code-mixed Dravidian text. We further provide LIME-based explainability analysis to interpret model decisions and identify systematic failure patterns. Our findings provide actionable insights for building robust hate speech detection systems for Dravidian languages.

**Keywords** — Hate Speech Detection, Tamil NLP, Multilingual Transformers, MuRIL, XLM-RoBERTa, Code-Mixed Text, Offensive Language, Dravidian Languages

---

## I. Introduction

Social media platforms have become primary channels for public discourse, but they also serve as vectors for hate speech, cyberbullying, and targeted harassment. The rapid growth of internet usage in India — with over 900 million users as of 2025 — has led to a surge in user-generated content in regional languages, including Tamil, which is spoken by approximately 80 million people worldwide [1].

Despite the pressing need for automated content moderation in Indian languages, the overwhelming majority of hate speech detection research has focused on English [2]. This creates a critical gap: existing tools fail to detect offensive content in Tamil, particularly when expressed through code-mixing (switching between Tamil and English within a single sentence), transliteration (writing Tamil in Latin script, known as "Tanglish"), and culturally specific expressions.

Tamil social media text presents unique challenges for Natural Language Processing (NLP):

1. **Code-mixing and Code-switching**: Users frequently alternate between Tamil and English, often within a single sentence (e.g., "Padam vera level mass iruku" — "The movie is another level mass").
2. **Script diversity**: Tamil content appears in Tamil Unicode script (தமிழ்), Latin transliteration (Tanglish), or mixed scripts.
3. **Non-standard orthography**: Social media Tamil lacks standardized spelling, with extensive abbreviations, phonetic spellings, and creative word formations.
4. **Cultural context dependency**: Certain expressions, caste references, and political slang carry offensive connotations specific to Tamil culture that general-purpose models cannot capture.

In this paper, we present a comprehensive comparative study of three state-of-the-art multilingual transformer models for Tamil hate speech detection:

- **MuRIL** (Multilingual Representations for Indian Languages) [3]: Pre-trained specifically on 17 Indian languages including Tamil.
- **XLM-RoBERTa** [4]: A massively multilingual model trained on 100 languages.
- **mBERT** (Multilingual BERT) [5]: The classic multilingual baseline.

Our key contributions are:
1. A systematic comparison of Indian-language-specific (MuRIL) vs. general multilingual (XLM-R, mBERT) pre-training for Tamil offensive language detection.
2. An analysis of model performance across script types (Tamil, Tanglish, code-mixed).
3. LIME-based explainability analysis revealing which linguistic features drive model predictions.
4. Identification of systematic error patterns and actionable recommendations for improving Dravidian hate speech detection.

---

## II. Literature Review

### A. Hate Speech Detection in English

Early hate speech detection relied on traditional machine learning approaches using bag-of-words, TF-IDF, and n-gram features with SVM and Logistic Regression classifiers [6]. The advent of deep learning brought significant improvements, with CNN and LSTM architectures capturing sequential dependencies in text [7]. The transformer revolution, initiated by BERT [5], established new state-of-the-art benchmarks. HateBERT [8], fine-tuned on Reddit hate speech data, demonstrated the value of domain-specific pre-training.

### B. Multilingual and Cross-lingual Approaches

Multilingual models such as mBERT [5] and XLM-RoBERTa [4] enable zero-shot cross-lingual transfer, where models trained on English data can detect hate speech in other languages. However, performance drops significantly for low-resource languages and code-mixed text [9]. Language-specific models like MuRIL [3] address this by incorporating transliterated and romanized text during pre-training.

### C. Hate Speech Detection in Dravidian Languages

The DravidianLangTech shared tasks [10] and LT-EDI workshops [11] have catalyzed research in Tamil, Malayalam, and Kannada hate speech detection. The OffensEval Dravidian dataset [12] provides a multi-class offensive language taxonomy specifically designed for YouTube comments in Dravidian languages.

Notable approaches include:
- Ensemble methods combining transformers with traditional ML features [13]
- Data augmentation techniques for addressing class imbalance [14]
- Multi-task learning frameworks leveraging sentiment analysis as an auxiliary task [15]

However, systematic comparisons between Indian-language-specific and general multilingual pre-training, coupled with explainability analysis, remain limited — a gap our work addresses.

### D. Explainable AI for Hate Speech

Explainability in hate speech detection is critical for building trust and identifying biases. LIME [16] and SHAP [17] provide model-agnostic explanations by identifying which input features contribute most to predictions. Recent work has applied these techniques to hate speech models [18], revealing that models sometimes rely on spurious correlations rather than genuine offensive indicators.

---

## III. Dataset

### A. OffensEval Dravidian Dataset

We use the Tamil configuration of the OffensEval Dravidian dataset [12], sourced from YouTube comments. The dataset contains 35,139 training samples and 4,388 validation samples.

**TABLE I: Dataset Statistics**

| Split | Samples |
|-------|---------|
| Train | 35,138 |
| Validation | 2,194 |
| Test | 2,194 |
| **Total** | **39,526** |

We split the original validation set 50-50 into validation and test sets to create a held-out evaluation benchmark.

### B. Label Taxonomy

The dataset employs a 6-class taxonomy:

**TABLE II: Label Distribution (Training Set)**

| Label | Count | Percentage |
|-------|-------|------------|
| Not_offensive | 25,424 | 72.4% |
| Offensive_Untargeted | 2,906 | 8.3% |
| Offensive_Targeted_Group | 2,557 | 7.3% |
| Offensive_Targeted_Individual | 2,343 | 6.7% |
| not-Tamil | 1,454 | 4.1% |
| Offensive_Targeted_Other | 454 | 1.3% |

The dataset exhibits severe class imbalance, with `Not_offensive` comprising 72.4% of all samples, while `Offensive_Targeted_Other` constitutes only 1.3%.

### C. Script Analysis

Our analysis reveals three distinct script categories in the data:

**TABLE III: Script Type Distribution**

| Script Type | Count | Percentage |
|-------------|-------|------------|
| Latin (Tanglish/English) | 28,519 | 81.2% |
| Tamil Unicode | 6,171 | 17.6% |
| Mixed (Code-Switched) | 393 | 1.1% |
| Other | 56 | 0.2% |

The dominance of Latin-script (Tanglish) content (81.2%) reflects the prevalence of transliterated Tamil on social media platforms. Pure Tamil script constitutes only 17.6%, while explicitly code-switched text is rare (1.1%).

---

## IV. Methodology

### A. Text Preprocessing

We apply the following preprocessing pipeline:
1. **URL and mention removal**: All HTTP links and @mentions are stripped.
2. **Hashtag normalization**: Hash symbols removed, hashtag text preserved.
3. **Character filtering**: Non-alphanumeric characters removed, Tamil Unicode range (U+0B80–U+0BFF) preserved.
4. **Case normalization**: Latin characters lowercased; Tamil script is case-insensitive.
5. **Whitespace normalization**: Multiple spaces collapsed to single space.

Notably, we do not apply stemming or lemmatization, as these tools are limited for Tamil and may destroy important morphological cues.

### B. Model Architectures

**TABLE IV: Model Specifications**

| Model | Parameters | Pre-training Languages | Pre-training Data |
|-------|-----------|----------------------|-------------------|
| MuRIL | 236M | 17 Indian languages | Indian web text + transliterations |
| XLM-RoBERTa | 278M | 100 languages | CommonCrawl (2.5TB) |
| mBERT | 177M | 104 languages | Wikipedia |

**MuRIL** is uniquely positioned for this task as it includes transliterated and romanized text in its pre-training data, directly addressing the Tanglish-dominant nature of our dataset.

### C. Training Configuration

All models are fine-tuned with the following hyperparameters:

**TABLE V: Training Hyperparameters**

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2 × 10⁻⁵ |
| Batch Size (effective) | 32 |
| Max Sequence Length | 128 |
| Epochs | 5 |
| Early Stopping Patience | 2 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Optimizer | AdamW |
| Loss Function | Weighted Cross-Entropy |
| Precision | FP16 (mixed) |
| Metric for Best Model | Weighted F1-Score |

### D. Class Imbalance Handling

We employ inverse-frequency class weighting in the cross-entropy loss function:

$$w_c = \frac{N}{C \times n_c}$$

where $N$ is the total number of samples, $C$ is the number of classes, and $n_c$ is the count of class $c$. This assigns higher penalty to misclassifying minority classes.

---

## V. Results

### A. Overall Performance

**TABLE VI: Model Comparison on Test Set**

| Model | Accuracy | F1 (Weighted) | F1 (Macro) | Precision (W) | Recall (W) |
|-------|----------|---------------|------------|----------------|------------|
| MuRIL | 0.7115 | 0.7310 | 0.4405 | 0.7637 | 0.7115 |
| XLM-RoBERTa | 0.7019 | 0.7275 | 0.4743 | 0.7791 | 0.7019 |
| mBERT | 0.7288 | 0.7505 | 0.4972 | 0.7840 | 0.7288 |
| **Ensemble (Vote)** | **0.7502** | **0.7620** | **0.4947** | **0.7817** | **0.7502** |

mBERT achieves the best individual performance (F1-W: 0.7505), while the majority-voting ensemble further improves the weighted F1-score to 0.7620 (+1.15% absolute). Notably, mBERT outperforms MuRIL on the test set despite MuRIL achieving the highest validation F1 during training (0.7485 vs. 0.7322), suggesting slight overfitting to the validation distribution.

### B. Per-Class Performance

**TABLE VII: Per-Class F1 Scores**

| Class | MuRIL | XLM-R | mBERT | Ensemble |
|-------|-------|-------|-------|----------|
| Not_offensive | 0.863 | 0.835 | 0.863 | **0.876** |
| Offensive_Untargeted | 0.383 | 0.411 | 0.405 | **0.430** |
| Offensive_Targeted_Individual | 0.230 | 0.421 | **0.445** | 0.425 |
| Offensive_Targeted_Group | 0.321 | 0.362 | 0.360 | **0.377** |
| Offensive_Targeted_Other | 0.000 | 0.000 | **0.082** | 0.000 |
| not-Tamil | 0.846 | 0.817 | 0.828 | **0.860** |

The ensemble achieves the best F1 on 4 of 6 classes. The `Offensive_Targeted_Other` class (only 30 test samples, 1.3% of training data) remains effectively undetectable, with only mBERT achieving a minimal F1 of 0.082.

### C. Confusion Matrix Analysis

Confusion matrices for all three individual models and the ensemble are provided in the supplementary materials (see `outputs/evaluation/`).

Key observations:
1. All models show strong performance on the majority `Not_offensive` class (F1 > 0.83 across all models).
2. The `Offensive_Targeted_Other` class (30 test samples) is consistently predicted as zero by MuRIL and XLM-R, with only mBERT achieving marginal detection (F1 = 0.082).
3. Significant cross-category confusion exists between `Offensive_Untargeted` and `Offensive_Targeted_Group`, likely due to overlapping linguistic cues in caste- and group-related discourse.
4. The `not-Tamil` class is reliably detected across all models (F1 > 0.81), demonstrating effective language identification even without explicit language features.

---

## VI. Error Analysis

### A. Error Categorization

We categorize prediction errors from the best-performing individual model (mBERT, 595 errors out of 2,194 test samples, 27.1% error rate) into four types:

1. **Dangerous False Negatives (94 cases)**: Offensive text predicted as Not_offensive — the most critical error type for content moderation. These represent real-world harm: offensive content that would escape automated filters.
2. **False Positives (292 cases)**: Non-offensive text flagged as offensive — the largest error category, driven by the model's conservative bias toward predicting offensive classes for ambiguous fan discourse and caste-community messages.
3. **Cross-Category Confusion**: Misclassification between offensive sub-types, particularly between `Offensive_Untargeted` and `Offensive_Targeted_Group`.
4. **not-Tamil Errors**: Misclassification involving the language identity class, primarily affecting Hindi and Telugu comments misclassified as Tamil offensive content.

### B. Script-Based Error Analysis

Error rates vary across script types. Tanglish text, despite dominating the training set (81.2%), shows moderate error rates due to non-standard orthography and creative spellings. Tamil Unicode text tends to be more formal and yields slightly lower error rates. Code-switched text, being rare (1.1%), suffers from insufficient training representation.

### C. Explainability (LIME Analysis)

We use LIME (Local Interpretable Model-agnostic Explanations) [16] to analyze word-level feature importance for individual predictions.

Key findings:
1. **Offensive indicators**: The model correctly identifies Tamil profanity and aggressive expressions as strong offensive signals.
2. **Context sensitivity**: Code-mixed expressions sometimes lose offensive connotation when individual words are perturbed, suggesting the model partially captures compositional semantics.
3. **Spurious correlations**: Certain neutral words (e.g., fan community names) receive disproportionate offensive weights due to their co-occurrence with offensive content in training data.

---

## VII. Discussion

### A. MuRIL vs. General Multilingual Models

Contrary to our initial hypothesis, MuRIL — despite its Indian-language-specific pre-training on 17 Indian languages including transliterated text — did not achieve the best test-set performance. While MuRIL had the highest validation F1 during training (0.7485), it was outperformed on the held-out test set by mBERT (F1-W: 0.7505 vs. 0.7310) and showed the lowest F1-Macro (0.4405), indicating poor minority-class detection.

This finding suggests that: (1) MuRIL may have overfit to the validation distribution during model selection; (2) mBERT's broader Wikipedia-based pre-training provides complementary coverage for the highly informal, non-standard Tamil found in YouTube comments; and (3) the advantage of Indian-language-specific pre-training may be more pronounced in formal text or pure Tamil settings rather than the heavily code-mixed, informal domain of social media comments.

### B. Ensemble Benefits

The majority-voting ensemble of all three models (F1-W: 0.7620) outperforms the best individual model by +1.15% absolute. The ensemble wins on 4 of 6 classes, demonstrating that the three models capture complementary aspects of the linguistic signal. This confirms the value of model diversity: MuRIL's Indian-language specialization, XLM-R's broad multilingual coverage, and mBERT's balanced generalization each contribute unique strengths.

### C. Challenges in Tamil Hate Speech Detection

1. **Code-mixing complexity**: The seamless blending of Tamil and English creates tokenization challenges, as subword vocabularies may split meaningful Tamil words.
2. **Cultural context**: Caste-based references, political slang (e.g., fan community rivalries), and culturally specific insults require deep contextual understanding that pre-trained models struggle to acquire from general corpora.
3. **Class imbalance**: Despite weighted cross-entropy loss, minority offensive sub-categories remain extremely difficult to detect. The `Offensive_Targeted_Other` class (1.3% of training data) achieves near-zero F1 across all models, and even the ensemble fails on it.
4. **False positive burden**: The high false positive rate (292 cases) suggests that models conflate aggressive but non-offensive fan discourse with genuine hate speech — a challenge specific to YouTube's polarized comment culture.

### D. Limitations

1. The dataset is sourced exclusively from YouTube comments, which may not generalize to other platforms (Twitter/X, Facebook, WhatsApp).
2. Our study focuses on Tamil only; extending to other Dravidian languages (Malayalam, Kannada, Telugu) is future work.
3. Explainability analysis is conducted on a limited sample (8 instances) due to computational constraints.
4. Weighted cross-entropy is the only class imbalance strategy explored; techniques such as focal loss, oversampling, and two-stage classification may yield further improvements.

---

## VIII. Conclusion and Future Work

This paper presents a systematic comparative study of multilingual transformer models for Tamil hate speech detection. Our experiments on the OffensEval Dravidian dataset demonstrate that mBERT achieves the best individual performance with a weighted F1-score of 0.7505, and a majority-voting ensemble further improves this to 0.7620. Contrary to expectations, Indian-language-specific pre-training (MuRIL) did not outperform general multilingual models on this code-mixed social media dataset.

Key takeaways:
1. General multilingual pre-training (mBERT) proves equally or more effective than Indian-language-specific pre-training (MuRIL) for code-mixed Dravidian hate speech detection, likely due to the highly informal, non-standard nature of social media text.
2. Majority-voting ensemble of diverse multilingual models provides consistent improvement (+1.15% F1-W), winning on 4 of 6 classes.
3. Class imbalance remains the primary challenge, with minority offensive sub-categories achieving F1 < 0.45 across all approaches.
4. LIME-based explainability reveals both legitimate offensive cues and spurious correlations with fan community terminology.

**Future Work**: We plan to (1) explore focal loss and synthetic minority oversampling for underrepresented classes, (2) extend the framework to multi-lingual Dravidian settings (Malayalam, Kannada, Telugu), (3) investigate weighted/learned ensemble strategies beyond majority voting, and (4) develop a real-time content moderation API for Indian social media platforms.

---

## References

[1] Internet and Mobile Association of India, "India Internet Report 2025," IAMAI, 2025.

[2] P. Fortuna and S. Nunes, "A survey on automatic detection of hate speech in text," *ACM Computing Surveys*, vol. 51, no. 4, pp. 1–30, 2018.

[3] S. Khanuja et al., "MuRIL: Multilingual Representations for Indian Languages," *arXiv preprint arXiv:2103.10730*, 2021.

[4] A. Conneau et al., "Unsupervised cross-lingual representation learning at scale," in *Proc. ACL*, 2020, pp. 8440–8451.

[5] J. Devlin et al., "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.

[6] Z. Waseem and D. Hovy, "Hateful symbols or hateful people? Predictive features for hate speech detection on Twitter," in *Proc. NAACL Student Research Workshop*, 2016, pp. 88–93.

[7] P. Badjatiya et al., "Deep learning for hate speech detection in tweets," in *Proc. WWW Companion*, 2017, pp. 759–760.

[8] T. Caselli et al., "HateBERT: Retraining BERT for abusive language detection in English," in *Proc. WOAH*, 2021, pp. 17–25.

[9] M. Pamungkas and V. Patti, "Cross-domain and cross-lingual abusive language detection: A hybrid approach with deep learning and a multilingual lexicon," in *Proc. ACL Student Research Workshop*, 2019, pp. 363–370.

[10] B. R. Chakravarthi et al., "DravidianLangTech@EACL2021: shared task on offensive language detection in Dravidian languages," in *Proc. EACL Workshop*, 2021.

[11] B. R. Chakravarthi et al., "Findings of the shared task on hope speech detection for equality, diversity, and inclusion," in *Proc. LT-EDI*, 2021.

[12] B. R. Chakravarthi et al., "Dataset for identification of homophobia and transphobia in multilingual YouTube comments," *arXiv preprint arXiv:2109.00227*, 2021.

[13] S. Dowlagar and R. Mamidi, "CfiltIITB at DravidianLangTech@EACL2021: Offensive language identification in Dravidian languages," in *Proc. EACL Workshop*, 2021.

[14] J. Risch et al., "Data augmentation for cross-domain hate speech detection," *arXiv preprint arXiv:2004.10404*, 2020.

[15] V. Rajamanickam et al., "Joint multitask learning for community question answering using task-specific embeddings," in *Proc. EMNLP*, 2020.

[16] M. T. Ribeiro et al., "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. KDD*, 2016, pp. 1135–1144.

[17] S. M. Lundberg and S. Lee, "A unified approach to interpreting model predictions," in *Proc. NeurIPS*, 2017, pp. 4765–4774.

[18] A. Mathew et al., "HateXplain: A benchmark dataset for explainable hate speech detection," in *Proc. AAAI*, 2021, pp. 14867–14875.
