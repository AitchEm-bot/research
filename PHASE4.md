---

# **PHASE 4 — Data Analysis & Visualization (UPDATED FINAL VERSION)**

This document replaces any previous Phase 4 draft.
It matches the final architecture of the project and is written so Claude can implement analysis code step-by-step.

---

# **🎯 Phase 4 Goal**

Produce all analytical outputs required for the thesis, including:

* manifest retention analysis
* baseline vs post-transform structural integrity
* perceptual quality analysis (PSNR, SSIM, VMAF aligned/stretched)
* platform behavior comparison
* statistical tests
* publication-ready plots

This phase **does not** make claims about post-transform cryptographic robustness, because phase results showed the manifest is stripped before verification can take place.

---

# **📥 1. Input Files Required**

Claude should expect these files:

```
data/results/final_metrics.csv
```

If Phase 2.5 round-trip testing was performed:

```
data/results/platform_results.csv
```

And optionally:

```
data/results/baseline_validation.csv
```

---

# **📚 2. Expected Columns in final_metrics.csv**

Claude must design analysis code assuming these fields exist:

### **A. Structural & Cryptographic Fields**

```
manifest_present
verified
signature_valid
hash_match
trust_valid
failure_reason
validation_state
```

### **B. Perceptual Quality Fields**

```
psnr
ssim
vmaf_stretched
vmaf_aligned
vmaf_method
```

### **C. Metadata Fields**

```
filename
asset_type
transform_type
transform_level
seed
model_version
platform_source
lossless_match
lossless_transform
processing_time_ms
timestamp
```

---

# **🧱 3. Load & Prepare Data (Claude should implement)**

* Load CSV into pandas
* Create `df_images` = images only
* Create `df_videos` = videos only
* For platform testing, create `df_platform` if platform_results.csv exists
* Normalize missing values
* Ensure numeric columns are converted properly

---

# **📊 4. Analysis Tasks for Claude to Implement**

Claude must implement all analysis code exactly as follows:

---

## **A. Manifest Retention Analysis (Structural Integrity)**

Plot the % of files where the manifest is still present after each transformation type.

Expected: **0% manifest retention for all transforms**.

Claude should generate:

* bar plot of retention %
* CSV or printout with retention by transform_type

---

## **B. Cryptographic Integrity (Baseline Only)**

Because manifests are stripped, cryptographic metrics after transformation will be zero.
Claude should:

1. Load `baseline_validation.csv`
2. Compare baseline integrity vs post-transform integrity
3. Produce a text summary:

```
Baseline cryptographic integrity = 100%
Post-transform cryptographic integrity = 0% (manifest stripped)
```

Claude should **NOT make claims** about cryptographic robustness under transformation.

---

## **C. Perceptual Quality Analysis (PSNR, SSIM)**

Claude should:

* Produce **boxplots** of PSNR and SSIM grouped by transform_type
* Identify lossless transforms via lossless_match == 1
* Include interpretation text:

```
Lossless transforms → PSNR=∞ and SSIM=1.0  
Other transforms → high fidelity despite manifest loss
```

---

## **D. Dual-VMAF Analysis (Stretched & Aligned)**

Claude must produce:

### **1. Stretched VMAF Plot**

* Boxplot by transform_type
* Expect Instagram/TikTok near **0–6 VMAF**

### **2. Aligned VMAF Plot**

* Boxplot by transform_type
* Expect much higher values (**50–85**)

### **3. Method Distribution Plot**

* Countplot of vmaf_method
* Helps sanity check correct alignment logic

---

## **E. Correlation Analysis**

Claude should calculate:

```
corr = df[['manifest_present','verified','psnr','ssim','vmaf_aligned']].corr()
```

Claude must produce a heatmap and a text explanation:

* Expect “no correlation” between perceptual quality and metadata retention.

---

## **F. Platform-Specific Analysis (if Phase 2.5 was performed)**

Claude should:

* Compare vmaf_stretched vs vmaf_aligned for each platform
* Compare PSNR vs SSIM for each platform
* Produce platform-level statistics:

```
Manifest retention rate
Average aligned VMAF
Average stretched VMAF
```

Expected outcomes:

* Manifest retention = **0%** for all platforms
* Aligned VMAF ≫ Stretched VMAF

---

## **G. Statistical Significance Tests**

Claude should run:

### **1. Chi-square for manifest retention**

```
chi2, p = stats.chi2_contingency(crosstab)
```

Expected p-value ≈ 1 (uniform destruction across transforms).

### **2. t-test for VMAF (internal vs platform)**

```
t_stat, p_val = ttest_ind(original_vmaf, insta_vmaf)
```

---

# **🖼️ 5. Outputs Claude Must Save**

Inside `plots/`, Claude should save:

```
manifest_retention.png
psnr_boxplot.png
ssim_boxplot.png
vmaf_stretched_boxplot.png
vmaf_aligned_boxplot.png
vmaf_method_distribution.png
correlation_heatmap.png
platform_vmaf_comparison.png   (if platform testing done)
```

Inside `data/analysis/`, Claude should save:

```
retention_table.csv
statistical_summary.txt
platform_summary.csv
phase4_conclusions.txt
```

---

# **📄 6. Claude Must Produce a Written Summary**

Claude should generate:

* Summary of key findings
* Bullet points for Results section
* Bullet points for Discussion section
* Limitations summary

**Claude must include the limitation:**

> “Cryptographic robustness under transformation could not be evaluated because all editing tools and social media platforms stripped the manifest before cryptographic validation could be performed.”

This is essential.

---

# **🎉 7. Completion Criteria**

Phase 4 is complete when:

* All plots are saved
* All CSV summaries are saved
* All statistics computed
* Written summaries produced
* No claims made about cryptographic robustness under transformation

---