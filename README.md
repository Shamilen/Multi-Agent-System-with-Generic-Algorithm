# Multi-Agent RL Research Benchmark: GPU Simulation, Equivariance & Transformers

Исследовательский репозиторий по глубокому многоагентному обучению с подкреплением (**Deep MARL**) на средах библиотеки **PettingZoo**. Проект объединяет два полноценных трека:

1. **[`waterworld_marl/`](waterworld_marl/README.md) — PettingZoo WaterWorld:** воспроизведение и существенная доработка экспериментов статьи МГТУ им. Н.Э. Баумана (Ш.Р. Гатиятуллин, науч. рук. д.т.н. А.Н. Алфимцев), перенос физики `pymunk` в **CUDA Graph** симулятор (**696×** ускорение), сравнение **SO(2)-эквивариантной нейроэволюции**, **Double DQN** и **CTDE MAPPO** в вариантах **`MLP`** и геометрического энкодера **`RayTransformerEncoder`**.
2. **[`pursuit_ga/`](pursuit_ga/README.md) — PettingZoo `sisl/pursuit-v5`:** нейроэволюция команды из 8 агентов с точной **C4-эквивариантностью**, рекуррентной памятью (`GRU`) и сравнением **`eq_cnn_gru`** vs **`eq_transformer`** (Vision-Transformer) на векторизованном GPU-симуляторе (**154×** ускорение).

> Подробное методологическое исследование, разбор статьи, таблицы по всем 5 сценариям и поведенческая аналитика: **[`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md)**.

---

## Архитектурная карта исследования

| Проект / Среда | Алгоритмы | Сравниваемые архитектуры | Симметрия / Инвариантность | Симулятор обучения (RTX 4080) | Проверка качества |
| --- | --- | --- | --- | --- | --- |
| **Pursuit** (`sisl/pursuit-v5`, 8 преследователей, 30 целей) | Генетический алгоритм (`GeneticAlgorithm`, `pop=96`, `70` поколений) | `eq_transformer` (`7,414` пар.) vs `eq_cnn_gru` (`8,954` пар.) + неэквивариантные `cnn`, `cnn_gru`, `transformer` | Точная **C4-эквивариантность** (повороты `90°`, `rotate_obs_c4` / `rotate_prev_action_c4` / `rotate_agent_phase_c4`) | [`VectorizedPursuitGPU`](pursuit_ga/gpu_sim.py) (`49,469 env-steps/s`, `154×` vs CPU) | 20 held-out сидов `470000..470019` на официальной CPU-среде PettingZoo |
| **WaterWorld** (`waterworld`, 5 сценариев × 3 метода, `600` эпизодов) | `GA (SO(2) + CRN)`, `9-action 5-step Dueling Double DQN`, `CTDE MAPPO` | [`RayTransformerEncoder`](waterworld_marl/algos/common.py) (`10.3K–18.9K`) vs `MLP` (`29.2K–128.1K`) vs [`WaterworldGAPolicy`](waterworld_marl/algos/ga.py) (`11–17` пар.) | Геометрический `[cos θ, sin θ]` readout в Transformer; точная **SO(2)-эквивариантность** в GA | [`VectorizedWaterworldGPU`](waterworld_marl/env/gpu_waterworld.py) CUDA Graph (`1,293,996 env-steps/s`, `696×` vs `pymunk`) | 20 held-out сидов `950000..950019` на `pymunk`-среде PettingZoo |

---

## Ключевые результаты и визуальное сравнение

### 1. Инженерное ускорение симуляции (CPU vs GPU / CUDA Graph)

![Сравнение скорости CPU vs GPU](docs/acceleration_summary.png)

- **WaterWorld:** время полного прогона всей матрицы (`5 сценариев × 3 алгоритма × 600 эпизодов`) снизилось с `3,250.8s` (`~54.2 мин` на CPU) до **`491.9s` (`8.2 мин`, `6.6×`)** с `MLP` и **`717.1s` (`12.0 мин`, `4.5×`)** с `RayTransformerEncoder`. Расхождение наблюдений GPU-симулятора с `pymunk` не превышает `1e-5`, а на CPU расчёт сенсоров ускорен векторизацией на NumPy (`atol < 1e-12`).
- **Pursuit:** 70 поколений (`pop=96 × 3 эп. = 288` параллельных сред + валидация элит) выполняются за **`511.7s`** (`7.31s/ген`, `eq_cnn_gru`) и **`528.3s`** (`7.55s/ген`, `eq_transformer`).

### 2. Pursuit (`sisl/pursuit-v5`): C4 Vision-Transformer vs C4 CNN-GRU

![Сравнение Pursuit](docs/pursuit_comparison.png)

| Политика (`20` сидов `470000..470019`) | Параметров | Награда агента | Поймано (`/ 30`) | Поимок (%) | Full clear | Перекрытий / шаг |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Baseline | `—` | `-46.00 ± 1.59` | `0.85 / 30` | `2.8%` | `0.0%` | `—` |
| Greedy Heuristic | `—` | `-44.01 ± 0.57` | `0.00 / 30` | `0.0%` | `0.0%` | `—` |
| `eq_transformer` (`Greedy Argmax`, default) | **`7,414`** | `-19.42 ± 3.98` | **`23.90 / 30`** | **`79.7%`** | **`5.0%`** | `2.246` |
| `eq_transformer` (`Stochastic T=0.15`) | `7,414` | `-20.17 ± 2.44` | `23.30 / 30` | `77.7%` | `0.0%` | `2.909` |
| **`eq_cnn_gru` (`Stochastic T=0.15`)** | `8,954` | **`-17.40 ± 2.63`** | **`24.65 / 30`** | **`82.2%`** | `0.0%` | **`0.223`** |
| `eq_cnn_gru` (`Greedy Argmax`) | `8,954` | `-18.31 ± 4.56` | `22.45 / 30` | `74.8%` | `0.0%` | `0.641` |

- `eq_transformer` компактнее на `17.2%` и выигрывает в детерминированном `Argmax`-режиме (`79.7%` поимок, единственная модель с `5.0%` полного зачищения карты и `0` пропусков `STAY`), тогда как `eq_cnn_gru` лучше держит дистанцию между преследователями (`0.223` перекрытий за шаг) и достигает максимума `82.2%` при `T=0.15`.

### 3. WaterWorld: `MLP` vs `RayTransformerEncoder` vs `SO(2) GA`

![Сравнение WaterWorld MLP vs Ray-Transformer](docs/waterworld_mlp_vs_transformer.png)

Командная награда **`mean_team_return`** на 20 held-out сидах `pymunk` (`mean_agent_return` и `food / poison` — в [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md)):

| Сценарий | Zero / Random | GA (`SO(2)`) | DQN (`MLP`) | DQN (`Ray-Tx`) | PPO (`MLP`) | PPO (`Ray-Tx`) | Победитель |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `base` (`5 ag, 30s`) | `-2.25` / `-563.18` | `+28.12 ± 49.44` | **`+11.08`** | `+4.87` | **`+68.88 ± 33.79`** | `+51.07 ± 32.28` | **PPO (MLP)** |
| `single` (`1 ag, 30s`) | `+23.78` / `-143.79` | **`+231.12 ± 53.32`** | `+162.11` | **`+193.86`** | `+173.95` | **`+179.12`** | **GA (SO(2))**, Tx > MLP |
| `no_speed` (`5 ag, 152d`) | `-2.25` / `-563.18` | **`+70.18 ± 39.86`** | **`+3.75`** | `-1.02` | `+57.51` | **`+69.42 ± 30.70`** | **GA / PPO (Ray-Tx)** |
| `few_sensors` (`5 ag, 10s`) | `-2.25` / `-563.18` | `+60.32 ± 32.39` | `+0.98` | **`+4.02`** | `+60.54` | **`+63.59 ± 46.14`** | **PPO (Ray-Tx)** |
| `more_poison` (`3f / 25p`) | `-12.57` / `-584.55` | **`+33.76 ± 30.50`** | `-9.33` | **`-3.67`** | **`+32.53 ± 27.89`** | `+21.26 ± 32.10` | **GA / PPO (MLP)** |

- **Геометрический `RayTransformerEncoder`** сократил число параметров в **`4.8×` (DQN)** и **`6.8×` (PPO)**, обеспечил инвариантность размера сети к числу сенсорных лучей (`n_sensors=10` и `30`) и обошёл MLP на **`single`**, **`no_speed PPO`**, **`few_sensors DQN/PPO`** и **`more_poison DQN`**.
- **Классический `MLP`** остаётся быстрее в градиентном шаге (`491.9s` против `717.1s`) и точнее избегает яда при плотной многоагентной координации (`base PPO`, `more_poison PPO`).

---

## Быстрый старт (`venv`)

> Используется виртуальное окружение `.\venv\Scripts\python.exe` (зависимости зафиксированы в [`requirements.txt`](requirements.txt)).

```powershell
$env:PYTHONIOENCODING="utf-8"

# 1. Прогон всех 32 автотестов (C4-эквивариантность, паритет GPU-симуляторов с PettingZoo, smoke-тесты)
.\venv\Scripts\python.exe -m pytest -v

# 2. Обучение Pursuit GA (70 поколений на RTX 4080, --arch eq_transformer или eq_cnn_gru)
.\venv\Scripts\python.exe -u -m pursuit_ga.train --device cuda --arch eq_transformer --output-dir runs/pursuit_ga_transformer
.\venv\Scripts\python.exe -u -m pursuit_ga.evaluate --checkpoint runs/pursuit_ga_transformer/best_model.pt --episodes 20 --seed-start 470000 --deterministic --save-gif runs/pursuit_ga_transformer/eval_pursuit.gif

# 3. Полная матрица WaterWorld (5 сценариев x 3 алгоритма, --arch transformer или mlp)
.\venv\Scripts\python.exe -m waterworld_marl.run_experiments --episodes 600 --eval-interval 100 --eval-episodes 8 --heldout-episodes 20 --device cuda --arch transformer --out-dir runs/waterworld_tx

# 4. Пересборка сравнительных графиков в docs/ из сохранённых запусков в runs/
.\venv\Scripts\python.exe scripts/build_research_plots.py
```

---

## Структура репозитория

```text
├── README.md                          # Обзор исследования, сводные таблицы и быстрый старт
├── requirements.txt                   # Зафиксированные зависимости (PyTorch 2.14.1+cu126, PettingZoo 1.27.0)
├── docs/
│   ├── RESEARCH_REPORT.md             # Полный исследовательский отчёт и разбор статьи МГТУ
│   ├── pursuit_comparison.png         # Сравнение eq_cnn_gru vs eq_transformer (сходимость, точность, баланс действий)
│   ├── waterworld_mlp_vs_transformer.png  # Сравнение MLP, Ray-Transformer и SO(2) GA по 5 сценариям
│   ├── waterworld_learning_curves.png # Кривые обучения WaterWorld (600 эпизодов)
│   └── acceleration_summary.png       # Профили ускорения CPU -> CUDA Graph / Vectorized GPU
├── scripts/
│   └── build_research_plots.py        # Генератор всех графиков docs/*.png из runs/**/results.json и history.json
├── pursuit_ga/                        # Трек PettingZoo SISL Pursuit (GA + C4 Transformer/CNN + GPU-симулятор)
│   ├── README.md                      # Документация модуля Pursuit GA
│   ├── config.py                      # EnvConfig, ModelConfig, GAConfig, TrainConfig
│   ├── env_utils.py                   # Препроцессинг (7,7,3) -> (3,7,7) и обёртка sisl/pursuit-v5
│   ├── networks.py                    # cnn, cnn_gru, eq_cnn_gru, transformer, eq_transformer (PursuitPolicyNet)
│   ├── genetic_algorithm.py           # ГА: элитизм, турнир, структурный кроссовер, антистагнация
│   ├── gpu_sim.py                     # VectorizedPursuitGPU и батчевый инференс BatchedPopulationPolicyGPU
│   ├── evaluator.py / evaluate.py     # Оценка на PettingZoo и совместимая загрузка чекпоинтов
│   └── train.py                       # CLI обучения и оценки
├── waterworld_marl/                   # Трек PettingZoo WaterWorld (воспроизведение + модификации статьи)
│   ├── README.md                      # Документация и таблицы бенчмарка WaterWorld
│   ├── scenarios.py                   # 5 конфигураций среды и параллельная оценка
│   ├── run_experiments.py             # Единый CLI для матрицы 5x3, графиков и GIF
│   ├── env/                           # PettingZoo pymunk (NumPy-сенсоры) и CUDA Graph симулятор VectorizedWaterworldGPU
│   └── algos/                         # RayTransformerEncoder, DuelingQNet (DQN), ActorCriticPPO (MAPPO), WaterworldGAPolicy (SO(2) GA)
└── tests/                             # 32 автотеста (эквивалентность CPU/GPU, C4-симметрия, smoke-проверки)
```
