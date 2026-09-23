# AnyTrack: Unifying Visual Object Tracking with Any Modalities

[![ACM MM 2026 Oral](https://img.shields.io/badge/ACM%20MM%202026-Oral-red)](https://dl.acm.org/doi/10.1145/3767308.3835550)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://dl.acm.org/doi/10.1145/3767308.3835550)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![GitHub stars](https://img.shields.io/github/stars/IdolLab/AnyTrack?style=social)](https://github.com/IdolLab/AnyTrack)

> 
> **AnyTrack: Unifying Visual Object Tracking with Any Modalities**
> 
> 
> *Hao Li, Yunzhi Zhuge, Wenning Hao*, Pingping Zhang*, Xiaoxiong Zhang, Dong Wang, Huchuan Lu*
> 
> 
> **ACM Multimedia 2026 Oral Presentation 🎤**
 
---

## 📋 Abstract

This repository contains the official implementation of <a href="https://arxiv.org/abs/2608.06773"><strong>AnyTrack</strong></a>, a unified visual object tracking framework that can handle <strong>any combination of modalities</strong> through a single model with flexible prompts. We propose a unified tokenization scheme to convert visual inputs of any modality (RGB, grayscale, depth, thermal infrared, event streams) and auxiliary prompts (box trajectories, text descriptions, audio clips) into a unified token space. A Modality-aware Interaction Module (MIM) based on Mixture-of-Experts dynamically adapts to heterogeneous modalities while maintaining temporal coherence. Furthermore, a Context Understanding Module (CUM) constructs global-local prompts from multi-modal references to enable target-aware context modeling. Extensive experiments on four multi-modal tracking benchmarks (RGBDT500, LasHeR, VisEvent, DepthTrack) demonstrate state-of-the-art performance across various modality combinations.

---

## 🔥 Motivation

<p align="center">
  <img src="assets/motivation.jpg" width="90%" alt="AnyTrack Motivation">
  <br>
  <em>Figure 1. Comparison with different object tracking paradigms. (a) Single-modal tracker uses a separate model for each individual modality. (b) Multi-modal tracker employs specific models for fixed modality combinations. (c) Architecture-shared tracker uses one model with task-specific parameters. (d) Unified tracker supports a fixed set of modalities. (e) Our AnyTrack enables object tracking with any modalities through a unified model and flexible prompts.</em>
</p>


---

## 🏗️ Framework

<p align="center">
  <img src="assets/pipeline.jpg" width="95%" alt="AnyTrack Framework">
  <br>
  <em>Figure 2. Overall framework of AnyTrack. Firstly, template and search region images of any modalities are tokenized to form vision tokens, which are then concatenated with temporal tokens from previous frames. Then, these tokens are processed by the modality-shared vision encoder for feature extraction. Subsequently, MIM performs dynamic feature interaction while aggregating temporal information to ensure spatio-temporal consistency. Afterwards, CUM constructs global-local prompts from multi-modal references to enable target-aware context modeling. Finally, a prediction head is used for target localization.</em>
</p>

---

## ✨ Key Modules

### Context Understanding Module (CUM)

<p align="center">
  <img src="assets/AIUM.jpg" width="85%" alt="CUM Details">
  <br>
  <em>Figure 4. Details of CUM. CUM constructs global-local prompts from multi-modal references and employs asymmetric bidirectional attention to fully exploit cross-modal information between prompts and visual features.</em>
</p>

CUM maintains a mask memory storing encoded target representations from previous frames, enabling the global-local prompts to serve as a bridge between historical spatial context and current visual features for precise localization.

---
