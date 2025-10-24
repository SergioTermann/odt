# 🚁 Causal Decision Transformer

Sprache: [English](README.md) | Deutsch

Ein auf kausaler Schlussfolgerung basierendes Framework für Luftkampf‑Entscheidungsagenten. Es kombiniert Decision Transformer und Kausalgraph‑Netzwerke, um in komplexen Mehrfachaufgaben‑Szenarien robuste und erklärbare Entscheidungen zu treffen.

## ✨ Hauptmerkmale

- 🧠 **Kausales Schließen**: Lernbare Kausalmatrizen auf mehreren Ebenen und über die Zeit
- 🔄 **Kontrafaktische Entscheidungen**: Weiche Interventionen zur Bewertung alternativer Handlungen
- 🎯 **Mehrfachaufgaben‑Lernen**: Koordination von Fähigkeiten/Tasks im Luftkampf
- 📊 **Adaptives Training**: Phasen „Exploration → Verfeinerung → Nutzung“
- ⚖️ **Unsicherheitsabschätzung**: Entscheidungskonfidenz und adaptive Exploration
- 🎮 **Simulation**: Integration mit `gym_dogfight` (Harfang 3D)

## 📁 Projektstruktur

```
odt/
├── README.md
├── README.de.md
├── main.py
├── setup.py
├── causal_dt_trainer.py
├── trainer.py
├── evaluation.py
├── data.py
├── utils.py
├── decision_transformer/
│   └── models/
│       ├── model.py
│       ├── causal_graph.py
│       └── lora.py
├── gym_dogfight/
│   ├── envs/
│   ├── spaces/
│   └── utils/
├── data/
└── collected_data/
```

## 🛠️ Installation

### Anforderungen
- Python 3.8+
- PyTorch 1.8+
- CUDA 11+ (optional)

### Abhängigkeiten
```bash
# Projekt klonen
git clone <repository-url>
cd odt

# Kernabhängigkeiten
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install networkx scikit-learn

# Gym-Umgebung
pip install -e .

# Transformers (optional)
cd transformers-4.5.1
pip install -e .
```

### Umgebung
```bash
# Python‑Pfad setzen (Linux/macOS)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Datenordner
mkdir -p data collected_data
```

## 🚀 Schnellstart

### 1) Daten vorbereiten
```bash
python data.py --data_path ./data/episodes_*.pkl
# oder
python load_data_test.py
```

### 2) Basis‑Training (Decision Transformer)
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

### 3) Kausal‑Erweitertes Training
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

### 4) Modellbewertung
```bash
python evaluation.py \
  --model_path ./models/causal_dt_model.pt \
  --env dogfight \
  --num_eval_episodes 100 \
  --render True
```

## 🏗️ Technische Architektur

- **Decision Transformer**: Sequenzmodell für Zustände/Handlungen/Belohnungen; integriert Kausalgraph und kontrafaktische Module.
- **CausalGraph**: Parametrisierte, lernbare Kausalmatrizen (mehrstufig, zeitlich); weiche Interventionen für differenzierbares Kontrafaktisches.
- **CounterfactualDecisionMaker**: Entscheidungsfusion mit Unsicherheitsabschätzung und adaptiver Exploration.
- **CausalTrainer**: Statistiken aktualisieren, Struktur lernen (PC/Granger/Score), Kontrafaktisches trainieren, Visualisierung.

## 📈 Trainingsablauf

1. Datenvorverarbeitung: Trajektorien, Normalisierung, Feature‑Extraktion
2. Kausalstruktur‑Lernen: Übergangszählungen, Mutual Information, Discovery‑Algorithmen
3. Modelltraining: Sequenzloss, Konsistenz‑Loss, Diversitäts‑Loss, Sparsitätsregularisierung
4. Adaptiv: LR‑Scheduler, Gewichtsanpassung, Phasenwechsel

## ⚙️ Experiment‑Konfiguration

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
  'causal_discovery_method': 'pc',
  'sparsity_weight': 0.01,
  'consistency_weight': 0.1,
  'intervention_strength': 0.5,
  'uncertainty_threshold': 0.3,
}
```

## 📊 Monitoring

- **Vorhersagegenauigkeit** (Tasks)
- **Kausalstruktur‑Änderung**
- **Kontrafaktische Diversität**
- **Entscheidungskonsistenz**

Visualisierung:
```python
trainer.visualize_causal_graph(save_path='./figures/')
```

## 🔧 Erweiterte Einstellungen

- Eigene Kausal‑Discovery:
```python
def custom_causal_discovery(statistics):
    # Algorithmus implementieren
    return causal_matrix

trainer.causal_discovery_method = custom_causal_discovery
```

- Mehrere Umgebungen:
```python
envs = ['dogfight_1v1', 'dogfight_2v2', 'dogfight_formation']
for env in envs:
    trainer = CausalSequenceTrainer(env=env, ...)
    trainer.train()
```

- Verteiltes Training:
```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py --distributed True
```

## 🐛 Fehlerbehebung

- **CUDA OOM**: `--batch_size` reduzieren, `--gradient_accumulation_steps` erhöhen
- **Kausalmatrizen konvergieren nicht**: `--sparsity_weight` / `--consistency_weight` anpassen
- **Instabile Trainingsläufe**: LR senken, `--warmup_steps` erhöhen

## 📚 Referenzen

- Decision Transformer: Reinforcement Learning via Sequence Modeling
- Causal Discovery in Machine Learning: Theory and Applications
- Counterfactual Reasoning for Decision Making under Uncertainty

## 🤝 Mitwirken

1. Fork erstellen
2. Feature‑Branch (`git checkout -b feature/AmazingFeature`)
3. Änderungen committen (`git commit -m 'Add some AmazingFeature'`)
4. Pushen (`git push origin feature/AmazingFeature`)
5. Pull Request öffnen

## 📄 Lizenz
MIT‑Lizenz — siehe `LICENSE`.

## 👥 Autoren
- **bafs** — Initiale Arbeit — [bafs](https://github.com/bafs)

## 🙏 Danksagung
- Inspiration durch Decision Transformer
- Dank an gym_dogfight
- Dank an die Open‑Source‑Community