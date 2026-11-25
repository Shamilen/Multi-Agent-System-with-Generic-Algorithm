# Multi-Agent RL Research Benchmark on PettingZoo WaterWorld: CUDA Graph Simulation, Equivariance & Cooperative Transformers

Исследовательский репозиторий по глубокому многоагентному обучению с подкреплением (**Deep MARL**) на непрерывной кооперативной среде **PettingZoo WaterWorld** ([`waterworld_marl/`](waterworld_marl/README.md)).

Проект воспроизводит и существенно дорабатывает эксперименты статьи МГТУ им. Н.Э. Баумана (Ш.Р. Гатиятуллин, науч. рук. д.т.н. А.Н. Алфимцев, *«Применение методов глубокого мультиагентного обучения в кооперативной среде»*):
- полный перенос физики `pymunk` и лучевых датчиков в батчевый **CUDA Graph** симулятор (**`748×`** ускорение относительно CPU `pymunk`, `4.7×` относительно GPU Eager);
- сравнение трёх парадигм MARL на 5 сценариях (`base`, `single`, `no_speed`, `few_sensors`, `more_poison`): **SO(2)-эквивариантной нейроэволюции (`GA`)**, **Double DQN / Hybrid QMIX** и **CTDE MAPPO / PPO-Coop**;
- переход с плотных **`MLP`** на компактный геометрический **`RayTransformerEncoder`** (`3.5×–6.8×` меньше параметров) и кооперативные доработки (`--coop`: **Hybrid QMIX**, поканальный **salience ray pooling** без размытия узких 1-лучевых сигналов яда, межагентный Self-Attention в акторе **`actor_comm`**, Cross-Agent Attention Critic, `GRU`-память, curriculum `n_coop: 1 → 2` и shaping на уклонение от яда по скорости сближения + анти-простой / пары / рандеву).

> Подробное методологическое исследование, разбор расхождений со статьёй, таблицы по всем 5 сценариям и поведенческая аналитика: **[`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md)**.

---

## Живая демонстрация всех 6 методов вместе

Синхронизированный запуск на **одинаковом отложенном (held-out) сиде** оригинальной `pymunk`-среды PettingZoo WaterWorld (`base`: 5 агентов, `n_coop=2`, `30` лучей, `seed=950001`):
- **Верхний ряд:** `GA (SO(2), 17 пар.)` | `DQN (Ray-Tx, до --coop)` | `PPO (Ray-Tx, до --coop)`
- **Нижний ряд:** `PPO (MLP, 128K реф.)` | **`Hybrid QMIX (после --coop)`** | **`PPO-Coop (Salience+Comm, после --coop)`**

![Синхронная демонстрация всех 6 методов WaterWorld](docs/methods_showcase.gif)

---

## Архитектурная карта исследования

| Семейство методов | Сравниваемые архитектуры | Параметров (`base`) | Ключевые механизмы | Симулятор обучения (RTX 4080) | Проверка качества |
| --- | --- | ---: | --- | --- | --- |
| **Нейроэволюция (`GA`)** | [`WaterworldGAPolicy`](waterworld_marl/algos/ga.py) | `17` (`11` в `no_speed`) | Точная **SO(2)-эквивариантность** + антитетические CRN-пары + momentum-ранжирование | [`VectorizedWaterworldGPU`](waterworld_marl/env/gpu_waterworld.py) CUDA Graph (`1.39M env-steps/s`) | 20 held-out сидов `950000..950019` на чистом `pymunk` |
| **Value-based (`DQN` / `QMIX`)** | `DuelingQNet (MLP)` vs `Ray-Tx` vs **Hybrid QMIX** ([`dqn.py`](waterworld_marl/algos/dqn.py)) | `49.7K` → **`10.4K`** | 9 направлений тяги, 5-step return, геометрический `[cos θ, sin θ]` readout, монотонный `QMixerNet` + curriculum `n_coop: 1→2` | [`VectorizedWaterworldGPU`](waterworld_marl/env/gpu_waterworld.py) + `TeamReplayBufferGPU` в VRAM | 20 held-out сидов `950000..950019` (100% децентрализованный тест) |
| **Policy Gradient (`MAPPO` / `PPO-Coop`)** | `ActorCriticPPO (MLP)` vs `Ray-Tx` vs **PPO-Coop** ([`ppo.py`](waterworld_marl/algos/ppo.py)) | `128.1K` → **`18.9K`** (`Ray-Tx`) | **Actor Comm (`actor_comm`)**, Cross-Agent Attention Critic, `GRU(128→64)` BPTT, активно-взвешенное кредитование, анти-простой/пары/рандеву shaping + curriculum | [`VectorizedWaterworldGPU`](waterworld_marl/env/gpu_waterworld.py) + `RunningMeanStdGPU` в VRAM | 20 held-out сидов `950000..950019` на чистом `pymunk` |

---

## Ключевые результаты и графики

### 1. Инженерное ускорение симуляции (CPU `pymunk` → GPU Eager → CUDA Graph)

![Сравнение скорости CPU vs GPU CUDA Graph](docs/acceleration_summary.png)

- **Пропускная способность среды (`base`, 500 шагов):** `CPU pymunk (NumPy-сенсоры)` `1,858 step/s` (`269.1 ms/ep`) → `GPU Eager (B=1024)` `297,793 step/s` (`160×`) → **`GPU CUDA Graph (B=1024)` `1,389,224 step/s` (`0.360 ms/ep`, `748×` против CPU и `4.7×` против GPU Eager)**.
- **Полный прогон всей матрицы (`5 сценариев × 3 алгоритма × 600 эпизодов`):** снизился с `3,250.8s` (`54.2 мин` на CPU) до **`491.9s` (`8.2 мин`, `6.6×`)** с `MLP`, **`717.1s` (`12.0 мин`, `4.5×`)** с `Ray-Transformer` и **`1,169.5s` (`19.5 мин`, `2.8×`)** в кооперативном режиме `--coop` (с `24`-средовой GPU-валидацией чекпоинтов).
- Расхождение наблюдений GPU-симулятора с эталонным `pymunk` не превышает `1e-5` на всех 5 сценариях, а на CPU расчёт сенсоров векторизован на NumPy (`atol < 1e-12`).

### 2. Сравнение архитектур и кооперативных доработок (`20` held-out сидов `pymunk`)

![Сравнение WaterWorld MLP vs Ray-Transformer vs Coop](docs/waterworld_mlp_vs_transformer.png)

Командная награда **`mean_team_return`** на 20 отложенных сидах `950000..950019` (`food / poison` и кривые обучения — в [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md)):

| Сценарий | GA (`SO(2)`) | DQN (`MLP`) | DQN (`Ray-Tx`) | **Hybrid QMIX (`Coop-Tx`)** | PPO (`MLP`) | PPO (`Ray-Tx`) | **PPO-Coop (`Salience+Comm`)** | Победитель |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **`base`** (`5 ag, 30s`) | `+28.12` | `+11.08` | `+4.87` | **`+8.24`** | `+68.88` | `+51.07` | **`+108.32`** | **PPO-Coop (рекорд!)** |
| **`single`** (`1 ag, 30s`) | **`+231.12`** | `+162.11` | `+193.86` | **`+195.07`** | `+173.95` | `+179.12` | **`+203.80`** | **GA**, PPO-Coop #1 нейросеть |
| **`no_speed`** (`5 ag, 152d`) | `+70.18` | `+3.75` | `-1.02` | **`+8.58`** | `+57.51` | `+69.42` | **`+78.81`** | **PPO-Coop (рекорд!)**, QMIX > DQN |
| **`few_sensors`** (`5 ag, 10s`) | `+60.32` | `+0.98` | `+4.02` | **`+11.48`** | `+60.54` | **`+63.59`** | `+50.32` | **PPO (Ray-Tx)**, QMIX > DQN |
| **`more_poison`** (`3f / 25p`) | **`+33.76`** | `-9.33` | `-3.67` | **`-1.04`** | `+32.53` | `+21.26` | **`+30.79`** | **GA / PPO (MLP / Coop — мин. яда `1.6`)** |

- **Hybrid Local + QMIX** превзошёл базовый `Ray-Tx DQN` на **всех 5 сценариях из 5 (`5/5`)**, подняв захваты еды в **`10×–12×`** на `few_sensors` (`1.0 → 10.0`) и `no_speed` (`0.9 → 11.2`).
- **PPO-Coop** (`salience_readout` + `actor_comm` + Cross-Agent Attention Critic + `GRU` + уклонение от яда по `v_close` + curriculum) установил абсолютные рекорды среди всех 7 моделей на **`base` (`+108.32`, `food=18.8`, `poison=2.3`, `+112%` к `Ray-Tx` и `+57%` к `MLP`)** и **`no_speed` (`+78.81`, обогнав `GA +70.18`)**, победил `PPO Ray-Tx` на **`4/5`** сценариях, показал лучший средний скор по 5 сценариям (**`+94.41`** vs `+78.68` MLP / `+76.89` Ray-Tx) и сократил попадания в яд в **`3.8×`** в среднем (**`11.3×`** на `more_poison`: `18.1 → 1.6`, ниже всех алгоритмов).

---

## Быстрый старт (`venv`)

> Используется виртуальное окружение `.\venv\Scripts\python.exe` (зависимости зафиксированы в [`requirements.txt`](requirements.txt)).

```powershell
$env:PYTHONIOENCODING="utf-8"

# 1. Прогон всех 12 автотестов (эквивалентность GPU-симулятора с pymunk, CUDA Graph, монотонность QMIX, smoke-тесты)
.\venv\Scripts\python.exe -m pytest -v

# 2. Полная матрица WaterWorld (5 сценариев x 3 алгоритма; --coop включает Hybrid QMIX + PPO-Coop Comm+Attn-GRU)
.\venv\Scripts\python.exe -m waterworld_marl.run_experiments `
  --episodes 600 --eval-interval 100 --eval-episodes 8 --heldout-episodes 20 `
  --device cuda --arch transformer --coop --out-dir runs/waterworld_coop

# 3. Базовые конфигурации для A/B-сравнения (Ray-Transformer и MLP)
.\venv\Scripts\python.exe -m waterworld_marl.run_experiments --episodes 600 --device cuda --arch transformer --out-dir runs/waterworld_tx
.\venv\Scripts\python.exe -m waterworld_marl.run_experiments --episodes 600 --device cuda --arch mlp --out-dir runs/waterworld

# 4. Пересборка сравнительных графиков и общей 2x3 GIF-демонстрации в docs/ из runs/
.\venv\Scripts\python.exe scripts/build_research_plots.py
.\venv\Scripts\python.exe scripts/build_showcase_gif.py
```

---

## Структура репозитория

```text
├── README.md                          # Обзор исследования, единая GIF-демонстрация, таблицы и быстрый старт
├── requirements.txt                   # Зафиксированные зависимости (PyTorch 2.14.1+cu126, PettingZoo 1.27.0)
├── docs/
│   ├── RESEARCH_REPORT.md             # Полный исследовательский отчёт и разбор статьи МГТУ
│   ├── methods_showcase.gif           # Единая синхронная 2x3 GIF-демонстрация всех 6 методов WaterWorld
│   ├── waterworld_mlp_vs_transformer.png  # Сравнение MLP, Ray-Transformer, Hybrid QMIX и PPO-Coop по 5 сценариям
│   ├── waterworld_learning_curves.png # Кривые обучения WaterWorld (600 эпизодов)
│   └── acceleration_summary.png       # Ускорение CPU pymunk -> GPU Eager -> CUDA Graph и время матрицы 5x3
├── scripts/
│   ├── build_research_plots.py        # Генератор всех графиков docs/*.png из runs/**/results.json
│   └── build_showcase_gif.py          # Генератор общей 2x3 гифки docs/methods_showcase.gif с live-счётчиками
├── waterworld_marl/                   # Основной пакет WaterWorld MARL
│   ├── README.md                      # Документация модуля и подробные таблицы по 3 сериям запусков
│   ├── scenarios.py                   # 5 конфигураций среды и параллельная held-out оценка на pymunk
│   ├── run_experiments.py             # Единый CLI для матрицы 5x3, графиков и GIF
│   ├── env/                           # PettingZoo pymunk (NumPy-сенсоры) и CUDA Graph симулятор VectorizedWaterworldGPU
│   └── algos/                         # RayTransformerEncoder, DuelingQNet/QMIX, ActorCriticPPO/PPO-Coop, SO(2) GA
└── tests/                             # 12 автотестов (эквивалентность GPU/pymunk, NumPy-сенсоры, QMIX, smoke)
```
