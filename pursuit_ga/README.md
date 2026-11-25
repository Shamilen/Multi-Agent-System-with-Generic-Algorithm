# Pursuit GA (`sisl/pursuit-v5`) — C4-эквивариантная нейроэволюция на CUDA

Модуль обучает команду из **8 децентрализованных преследователей** в дискретной многоагентной среде PettingZoo **`sisl/pursuit-v5`** методом **генетического алгоритма** (`GeneticAlgorithm`) с шаговым CUDA-симулятором [`VectorizedPursuitGPU`](gpu_sim.py) и батчевым инференсом всей популяции [`BatchedPopulationPolicyGPU`](gpu_sim.py).

## Среда и проблема «наивных» политик

- **Параметры (`EnvConfig` в [config.py](config.py)):** сетка `16×16`, `n_pursuers=8`, `n_evaders=30`, окно наблюдения `obs_range=7` (`(7, 7, 3)` → `(3, 7, 7)` в [`preprocess_observations`](env_utils.py)), `max_cycles=500`, `surround=True`, `n_catch=2`, `catch_reward=5.0`, `tag_reward=0.01`, `urgency_reward=-0.1`, `shared_reward=True`.
- **Препроцессинг ([env_utils.py](env_utils.py)):** каналы преобразуются в `[walls, allied_pursuers, evaders]`, из центра союзного канала вычитается сам агент (`max(c - 1, 0)`), счётчики сжимаются через `tanh(0.5 * x)`.
- **Проблема окружения:** жадная эвристика сближения сталкивает преследователей в одну клетку (`0.00/30` поимок), а неэквивариантные политики при отборе склонны смещаться в 1–2 оси и «фармить» `tag_reward` без коллективного окружения цели.

## Поддерживаемые архитектуры (`PursuitPolicyNet` в [networks.py](networks.py))

Все агенты используют общие веса (`parameter sharing`), историю предыдущего действия (`use_prev_action=True`) и синус-косинусную фазу агента [`build_agent_phase_features`](networks.py) для расхождения в пустых областях сетки.

| Архитектура (`--arch`) | Энкодер признаков | Эквивариантность C4 | Память (GRU) | Параметров |
| --- | --- | :---: | :---: | ---: |
| `cnn` | 3-слойный `Conv2d(3→8→12→8)` | Нет | Нет | `4,017` |
| `cnn_gru` | 3-слойный `Conv2d(3→8→12→8)` | Нет | `GRUCell(24→20)` | `6,757` |
| `eq_cnn_gru` | 4-поворотный shared `Conv2d(3→8→12→8)` | Точная | `GRUCell(24→20)` | `8,954` |
| `transformer` | [`_GridTransformerBlock`](networks.py) (`49` токенов, `d=16, 2h, 1L, d_ff=32`) + 2D-коорд. `(dy, dx)` | Нет | `GRUCell(24→20)` | `6,177` |
| `eq_transformer` (default) | 4-поворотный shared [`_GridTransformerBlock`](networks.py) (`d=16, 2h, 1L, d_ff=32`) | **Точная** | `GRUCell(24→20)` | `7,414` |
| **`eq_comm_transformer`** | Гибридный `3×3` Conv-стем (`8` кан.) + 4-поворотный [`_GridTransformerBlock`](networks.py) + межагентный Self-Attention [`_AgentCommBlock`](networks.py) (`comm_dim=12, 2h`) | **Точная** | `GRUCell(24→20)` | **`8,994`** |

### Механизм точной C4-эквивариантности (`eq_cnn_gru`, `eq_transformer` и `eq_comm_transformer`)
1. Вход `(3, 7, 7)` проецируется в 4 ортогональные ориентации (`rotate_obs_c4`), соответствующие `[LEFT, RIGHT, DOWN, UP]`.
2. Общий пространственный энкодер извлекает признаки каждой ориентации; инвариантное среднее по 4 направлениям (+ флаг предыдущего `STAY`) подаётся в `GRUCell(24 → 20)`.
3. В `eq_comm_transformer` поверх C4-инвариантных состояний `GRU` всей команды из `8` агентов выполняется раунд межагентного Self-Attention ([`_AgentCommBlock`](networks.py)), сохраняющий **точную C4-эквивариантность** (подтверждено тестами) и передающий контекст команды в головы действий.
4. В [`GPUPopulationEvaluator`](gpu_sim.py) поддерживается тренировочный кооперативный shaping: анти-наложение (`--overlap-penalty`) и поощрение разностороннего окружения цели (`--flank-bonus`); итоговые метрики всегда считаются на чистой среде PettingZoo.

## Генетический алгоритм и CUDA-симуляция

- **ГА ([genetic_algorithm.py](genetic_algorithm.py)):** `pop_size=96`, `generations=70`, `episodes_per_genome=3`, `elite_count=10` (дооценка элит на `8` фиксированных сидах), турнирный отбор (`tournament_size=4`), структурный кроссовер нейронов (`crossover_type="neuron"`, `crossover_rate=0.30`), послойно-масштабированная мутация (`sigma=0.055 → 0.007`, затухание `0.978`), антистагнационный сброс (`stagnation_patience=12`, `stagnation_boost=1.25`), `immigrant_count=3`, вспомогательный бонус `catch_bonus_weight=0.8` только при эволюционном отборе.
- **GPU-симулятор ([gpu_sim.py](gpu_sim.py)):** [`VectorizedPursuitGPU`](gpu_sim.py) пошагово эквивалентен `sisl/pursuit-v5` (на 288 параллельных эпизодах `49,469 env-steps/s`, `10.11 ms/ep` против `1,552.4 ms/ep` и `322 env-steps/s` на CPU — ускорение **~154×**). Среднее время поколения: **`7.31s`** (`eq_cnn_gru`, `511.7s` на 70 поколений) и **`7.55s`** (`eq_transformer`, `528.3s` на 70 поколений) на RTX 4080.

## Сравнение `eq_cnn_gru` vs `eq_transformer` (20 отложенных сидов `470000..470019`)

Проверка выполнена на оригинальной среде PettingZoo `sisl/pursuit-v5` после завершения обучения:

| Модель / Режим | Пар-ры | Награда агента | Поймано (`/ 30`) | Поимок | Full clear | Перекрытий / шаг | Баланс действий `[L, R, D, U, STAY]` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Random Baseline | `—` | `-46.00 ± 1.59` | `0.85 / 30` | `2.8%` | `0.0%` | `—` | Равномерный случайный выбор |
| Greedy Heuristic | `—` | `-44.01 ± 0.57` | `0.00 / 30` | `0.0%` | `0.0%` | `—` | Жадное сближение |
| `eq_cnn_gru` (`Argmax`) | `8,954` | `-18.31 ± 4.56` | `22.45 / 30` | `74.8%` | `0.0%` | `0.641` | `[19952, 20000, 20118, 19804, 126]` |
| **`eq_cnn_gru` (`Stochastic T=0.15`)** | `8,954` | **`-17.40 ± 2.63`** | **`24.65 / 30`** | **`82.2%`** | `0.0%` | **`0.223`** | `[19905, 19637, 19929, 19943, 586]` |
| `eq_transformer` (`Argmax`) | `7,414` | `-19.42 ± 3.98` | `23.90 / 30` | `79.7%` | **`5.0%`** | `2.246` | `[18845, 19799, 20608, 20556, 0]` |
| `eq_transformer` (`Stochastic T=0.15`) | `7,414` | `-20.17 ± 2.44` | `23.30 / 30` | `77.7%` | `0.0%` | `2.909` | `[19887, 20083, 20179, 19806, 45]` |
| **`eq_comm_transformer` (`Argmax`)** | `8,994` | `-19.47 ± 3.12` | `23.40 / 30` | `78.0%` | `0.0%` | **`0.118`** | `[21355, 20717, 19021, 18907, 0]` |
| **`eq_comm_transformer` (`Stoch T=0.15`)** | `8,994` | `-19.74 ± 2.82` | `23.55 / 30` | `78.5%` | `0.0%` | **`0.142`** | `[20062, 20045, 19987, 19905, 1]` |

## CLI запуска

```powershell
# Обучение гибридного C4 Comm-Transformer с кооперативным shaping
.\venv\Scripts\python.exe -u -m pursuit_ga.train --device cuda --arch eq_comm_transformer --overlap-penalty 1.0 --flank-bonus 0.5 --output-dir runs/pursuit_ga_coop

# Обучение базового C4 Vision-Transformer (--arch eq_transformer) или CNN-GRU (--arch eq_cnn_gru)
.\venv\Scripts\python.exe -u -m pursuit_ga.train --device cuda --arch eq_transformer --output-dir runs/pursuit_ga_transformer
.\venv\Scripts\python.exe -u -m pursuit_ga.train --device cuda --arch eq_cnn_gru --output-dir runs/pursuit_ga_eq

# Оценка любого сохранённого чекпоинта на PettingZoo (обратно совместимо со старыми чекпоинтами)
.\venv\Scripts\python.exe -u -m pursuit_ga.evaluate --checkpoint runs/pursuit_ga_coop/best_model.pt --episodes 20 --seed-start 470000 --save-gif runs/pursuit_ga_coop/eval_pursuit.gif
```
