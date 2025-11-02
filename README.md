## 🧭 Overview

**SpatialCL** is a *plug-and-play contrastive learning framework* designed for spatially structured modalities, including **RGB**, **thermal**, and **3D** data.  
It robustly handles *intra-* and *inter-class variability*, enabling consistent embeddings across challenging datasets.

🧪 As a demonstration of its capabilities, **SpatialCL** has been applied in **[DiSCO 🔗](https://github.com/Olemou/SpatialCL)** — *Detection of Spills in Indoor environments using weakly supervised contrastive learning* — showcasing its practical impact in real-world spill detection scenarios.

⚙️ While the framework is **modality-agnostic** and can be extended to other dense spatial tasks, extending **SpatialCL** to sparse, graph-structured data such as **skeletons** represents an exciting direction for future work.
**Framework Architecture**
  <p align="center">
    <img src="docs/architecture.png" alt="SpatialCL Architecture" width="600"/>
  </p>

## Key Features
- ✅ Handles **ambiguous and irregular objects** that standard vision models struggle with
- ✅ Supports: **RGB, thermal, depth, etc.**
- ✅ **Memory-optimized** contrastive learning for faster training
- ✅ Produces **highly discriminative embeddings** for downstream tasks
- ✅ Easy integration into existing PyTorch pipelines
