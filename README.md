# 🚁 Causal Decision Transformer

Language: English | [Deutsch](README.de.md)

A causal reasoning–based aerial combat decision agent framework that combines Decision Transformer and Causal Graph Networks for robust, explainable decision-making in complex multi-task scenarios.

## ✨ Key Features
- Causal reasoning with multi-level and temporal causal matrices
- Counterfactual decision-making via soft interventions
- Multi-task coordination for air-combat skills and strategies
- Adaptive training phases (exploration → refinement → exploitation)
- Uncertainty estimation with adaptive exploration
- Integrated simulation with `gym_dogfight` (Harfang 3D)

## 📁 Project Structure
```
odt/
├── README.md                # English documentation (language switch on top)
├── README.de.md             # German documentation (mirror)
├── main.py                  # Entry point and experiment runner
├── causal_dt_trainer.py     # Causal sequence trainer
├── trainer.py               # Base trainer
├── evaluation.py            # Evaluation utilities
├── data.py                  # Dataset loading and processing
├── decision_transformer/
│   └── models/
│       ├── model.py         # Core model components
│       ├── causal_graph.py  # Causal Graph and Counterfactual modules
│       └── lora.py          # Optional LoRA adapter
├── gym_dogfight/            # Simulation environment
├── data/                    # Training datasets
└── collected_data/          # Collected trajectories
```

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 1.8+
- CUDA 11+ (optional)

### Setup
```bash
# Clone the project
git clone <repository-url>
cd odt

# Core dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install networkx scikit-learn

# Install local gym environment
pip install -e .

# Optional: transformers (local copy)
cd transformers-4.5.1
pip install -e .
```

## 🚀 Quick Start

### 1) Data Preparation
```bash
python data.py --data_path ./data/episodes_*.pkl
# or
python load_data_test.py
```

### 2) Baseline Training (Decision Transformer)
```bash
python main.py \
  --env dogfight \
  --dataset ./data/episodes_20250731-000040.pkl \
  --model_type dt \
  --embed_dim 128 \
  --n_layer 3 \
  --n_head 1 \
  --activation_function relu \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --warmup_steps 10000 \
  --num_eval_episodes 100 \
  --max_iters 20 \
  --num_steps_per_iter 10000 \
  --device cuda \
  --log_to_wandb True
```

### 3) Causal‑Enhanced Training
```bash
python main.py \
  --env dogfight \
  --dataset ./data/episodes_20250731-000040.pkl \
  --model_type causal_dt \
  --embed_dim 128 \
  --n_layer 3 \
  --n_head 1 \
  --causal_discovery_method pc \
  --sparsity_weight 0.01 \
  --consistency_weight 0.1 \
  --learning_rate 1e-4 \
  --max_iters 50 \
  --device cuda
```

### 4) Evaluation
```bash
python evaluation.py \
  --model_path ./models/causal_dt_model.pt \
  --env dogfight \
  --num_eval_episodes 100 \
  --render True
```

## 🏗️ Technical Architecture
- Decision Transformer: sequence model (states/actions/rewards/returns) with causal integration
- CausalGraph: parameterized multi-level and temporal causal matrices; soft interventions
- CounterfactualDecisionMaker: decision fusion with uncertainty-guided adaptive exploration
- CausalTrainer: statistics update, structure learning (PC/Granger/score), counterfactual training, visualization

## 📈 Training Workflow
1. Preprocess trajectories and extract features
2. Learn causal structure from transitions and outcomes
3. Train with sequence loss, consistency and diversity constraints, sparsity regularization
4. Adapt via LR scheduling, dynamic weights, and phase switching

## ⚙️ Configuration Examples
```python
MODEL_CONFIG = {
  'embed_dim': 128,
  'n_layer': 3,
  'n_head': 1,
  'activation_function': 'relu',
  'dropout': 0.1,
  'max_length': 20,
}

TRAINING_CONFIG = {
  'learning_rate': 1e-4,
  'weight_decay': 1e-4,
  'warmup_steps': 10000,
  'max_iters': 50,
  'batch_size': 64,
  'num_steps_per_iter': 10000,
}

CAUSAL_CONFIG = {
  'causal_discovery_method': 'pc',  # pc, granger, score_based
  'sparsity_weight': 0.01,
  'consistency_weight': 0.1,
  'intervention_strength': 0.5,
  'uncertainty_threshold': 0.3,
}
```

## 📊 Monitoring
- Task prediction accuracy
- Causal structure change
- Counterfactual diversity
- Decision consistency and uncertainty calibration

Visualization:
```python
trainer.visualize_causal_graph(save_path='./figures/')
```

## 🔧 Advanced
- Custom causal discovery via callback (`trainer.causal_discovery_method = your_fn`)
- Multi‑environment training loops
- Distributed training (`torch.distributed.launch`)

## 🐛 Troubleshooting
- CUDA OOM: reduce `--batch_size`, use gradient accumulation
- Causal matrices not converging: tune `--sparsity_weight` / `--consistency_weight`
- Unstable runs: lower LR, increase `--warmup_steps`

## 📚 References
- Decision Transformer: Reinforcement Learning via Sequence Modeling
- Causal Discovery in Machine Learning: Theory and Applications
- Counterfactual Reasoning for Decision Making under Uncertainty

## 🤝 Contributing
1. Fork
2. Feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add some AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open PR

## 📄 License
MIT License — see `LICENSE`.

## 👥 Authors
- **bafs** — initial work — [GitHub](https://github.com/bafs)

## 🙏 Acknowledgments
- Inspiration from Decision Transformer
- Thanks to gym_dogfight
- Thanks to the open‑source community