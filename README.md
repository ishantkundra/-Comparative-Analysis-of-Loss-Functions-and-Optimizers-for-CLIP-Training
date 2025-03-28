# 📊 Comparative Analysis of Loss Functions and Optimizers for CLIP Training

This research project investigates improvements to **Contrastive Language–Image Pretraining (CLIP)** by benchmarking advanced **loss functions** and **optimizers**. We propose a novel **Dynamic Temperature Scaling Loss** and evaluate its performance on multiple image-text retrieval and zero-shot classification tasks.

✅ **Honorable Mention Winner** – CLIP Training Competition (Fall 2024 Deep Learning Course)

Developed by **Ishant Kundra** and **Rahul Ravi Kadam**  
📍 Texas A&M University – CSCE Deep Learning Course Project

---

## 🏆 Recognition

> 🎖️ **Award**: Honorable Mention at CLIP Training Competition – Fall’24 DL Course  
> 🗓️ Date: Dec 10, 2024  
> 📄 View Certificate: [Hornoral-2.pdf](./Hornoral-2.pdf)

---

## 🚀 Project Overview

| Item              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| 🎯 **Goal**        | Improve CLIP performance through custom loss functions + optimizer tuning  |
| 🧠 **Losses**      | CLIP, CyCLIP, SogCLR, iSogCLR, **Dynamic Temp Loss (ours)**                 |
| ⚙️ **Optimizers**  | AdamW, RAdam, NovoGrad, Adafactor                                            |
| 🗂️ **Datasets**    | CC3M (100K train), MSCOCO (5K eval), ImageNet (ZS)                          |
| 🧪 **Evaluations** | Text-to-Image (T2I), Image-to-Text (I2T), Zero-shot classification accuracy |
| 📊 **Metrics**     | Recall@1/5/10, Top-k Accuracy, Efficiency (Time)                            |

---

## 📁 Repository Structure

<pre>
📦 Comparative-Analysis-of-Loss-Functions-and-Optimizers-for-CLIP-Training
│
├── Best_Model-iSogCLR_new_RAdam/           # Best-performing iSogCLR+RAdam model
│   ├── output/
│   │   ├── eval_isogclr_new_radam.../      # COCO & ImageNet logs
│   │   └── isogclr_new_radam.../           # Checkpoints & config (args.json)
│   ├── eval.slurm                          # HPC eval job
│   ├── run.slurm                           # HPC training job
│   ├── job_output_eval.slurm.12169911
│   └── job_output_run.slurm.12166717
│
├── Novel_Model-Dynamic_temp_scaling_loss_RAdam.zip   # 🔥 Our proposed loss implementation
├── bimodal_exps.zip                                  # Additional bimodal training experiments
├── Report/
│   └── Project_Report.pdf                            # 📄 Final detailed PDF report
├── Honorral-2.pdf                                    # 🏆 Award certificate
├── Screenshots/
│   └── Screenshot_*.png                              # Repo preview images
├── .gitattributes
├── .gitignore
└── README.md                                         # ← You're here!
</pre>

---

## 🧪 Methodology

- ✅ **Model**: CLIP-style dual encoders (ViT-B/32 + DistilBERT)
- ✅ **Training Setup**: 30 epochs, batch size 128, cosine LR scheduler, warm restarts
- ✅ **Loss Exploration**:
  - `CLIP`: Standard symmetric loss
  - `CyCLIP`: Adds cyclic consistency
  - `SogCLR`: Global contrastive
  - `iSogCLR`: Adaptive temp per sample
  - `DynamicTemp`: Adapts based on variance in pos/neg pairs (new)
- ✅ **Evaluation**:
  - **MSCOCO**: T2I & I2T retrieval (Recall@1/5/10)
  - **ImageNet**: Zero-shot Top-1, Top-5
- ✅ **Metrics**: Accuracy, training time, log scaling, visualizations

---

## 📊 Key Results (from Report)

| Loss Function | Optimizer | T2I R@1 | I2T R@1 | ZS Top-1 Acc | Training Time |
|---------------|-----------|---------|---------|--------------|----------------|
| iSogCLR       | RAdam     | 14.86%  | 11.23%  | **27.03%**   | 3h 40m         |
| DynamicTemp   | RAdam     | 14.58%  | 9.32%   | 22.40%       | 3h 58m         |
| iSogCLR       | AdamW     | 14.48%  | 11.15%  | 26.74%       | 3h 23m         |
| iSogCLR       | NovoGrad  | 11.72%  | 9.98%   | 15.96%       | **3h 20m**     |

📌 *Detailed tables and evaluation graphs can be found in the* [Project_Report.pdf](./Report/Project_Report.pdf)

---

## 🔥 Novel Loss Function: Dynamic Temperature Scaling Loss

We propose a new contrastive loss that adaptively adjusts temperature per batch based on variance in the similarity scores between positive and negative pairs.

```python
class DynamicTemperatureScalingLoss(nn.Module):
    def __init__(self, initial_tau=0.05, alpha=0.1, min_tau=1e-3, max_tau=1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(initial_tau))
        self.alpha = alpha
        self.min_tau = min_tau
        self.max_tau = max_tau

    def forward(self, image_features, text_features):
        sim_matrix = torch.matmul(image_features, text_features.T)
        pos_pairs = torch.diag(sim_matrix)
        neg_pairs = sim_matrix[~torch.eye(sim_matrix.size(0), dtype=bool)].view(sim_matrix.size(0), -1)
        pos_var = torch.var(pos_pairs, unbiased=False)
        neg_var = torch.var(neg_pairs, unbiased=False)

        with torch.no_grad():
            tau_update = self.tau * (1 + self.alpha * torch.tanh(neg_var - pos_var))
            self.tau.data = torch.clamp(tau_update, self.min_tau, self.max_tau)

        scaled_sim_matrix = sim_matrix / self.tau
        labels = torch.arange(sim_matrix.size(0)).to(image_features.device)
        return F.cross_entropy(scaled_sim_matrix, labels)

## 👨‍💻 Author

**Ishant Kundra**  
M.S. Computer Science, Texas A&M University  
📬 ishantkundra9@gmail.com
