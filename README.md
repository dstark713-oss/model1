# Tactical v0.1 — README

## Что это

Daily ML Trading System для российского рынка. Система ежедневно:
- выбирает торговый универс (stocks),
- считает фичи по ~200 акциям TQBR,
- оценивает инструменты двумя независимыми CatBoost-моделями (`P(TP)` и `P(SL)`),
- ранжирует кандидатов, фильтрует по корреляции,
- собирает равновесный портфель 4–8 активов с загрузкой по ширине рынка,
- формирует сигнал после Close, исполнение — на следующем Open.

## ⚠️ Статус: пилотный запуск, 50% объёма

**Система торгуется вживую с 2026-11-17, но НЕ прошла живую валидацию.**

- **Объём: 0.5× от полного.** Причина: модель зафиксирована (`v_2026-11-17`), но её поведение вне train (после даты retrain) ещё не подтверждено реальными результатами. Бэктест — это история, а не живая торговля.
- **Вердикт: май 2027.** К этому моменту накопится ~6 месяцев живых данных (≈125 торговых дней). Если live-результаты не расходятся с бэктестом (Sharpe, MDD, hit rate в пределах доверительных интервалов) — объём увеличивается до 1.0×. Если расходятся — модель переобучается или система останавливается.
- **Retrain: раз в 6 месяцев** (следующий — ~2027-05). Между retrain модель фиксирована, гиперпараметры и `N*` не меняются.

## Структура проекта

```
tactical/
├── configs/                  # все параметры (не хардкод)
│   ├── universe.yaml         # тикеры, режим IMOEX, TOP100
│   ├── features.yaml         # окна/периоды фич
│   ├── targets.yaml          # H=5, K_TP=1.5, K_SL=1.0, vol_window=20
│   ├── models.yaml           # CatBoost, калибровка, persistence
│   ├── portfolio.yaml        # N-grid, корреляция, breadth
│   ├── wf.yaml               # pretrain 3Y, retrain 6M, purge 10D
│   ├── backtest.yaml         # costs 3+3 bps
│   ├── signal.yaml           # daily signal
│   └── agent.yaml            # LLM-фильтр новостей
│
├── data/                     # Phase 1: данные
│   ├── loader.py             # MOEX ISS + Yahoo, все TQBR + споты
│   ├── storage.py            # SQLite (prices, securities, data_version)
│   ├── validation.py         # проверка OHLCV (без подмены значений)
│   └── main.py               # CLI: update / validate
│
├── portfolio/                # Phase 2 + 6: универс, режим, портфель
│   ├── regime.py             # IMOEX: Close vs EMA100 -> stocks/futures
│   ├── universe.py           # point-in-time TOP100 по value за 60 дней
│   ├── breadth.py            # Market Breadth (EMA20)
│   ├── ranking.py            # score = p_tp * (1 - p_sl)
│   ├── correlation.py        # окно 20, порог 0.8, greedy filter
│   ├── allocation.py         # равные веса, exposure = breadth
│   └── selection.py          # выбор N* на train
│
├── features/                 # Phase 2: фичи
│   ├── trend.py              # EMA, slope, momentum, ADX
│   ├── volatility.py         # ATR, RV, vol expansion
│   ├── volume.py             # rel volume, volume trend, shock
│   ├── oscillators.py        # RSI, Stochastic, MACD
│   ├── price_action.py       # breakout, wicks, body
│   └── pipeline.py           # сборка всех фич
│
├── targets/                  # Phase 3: разметка
│   └── triple_barrier.py     # TP/SL/timeout/ambiguous, H=5
│
├── models/                   # Phase 4: ML
│   ├── tp_model.py           # CatBoost для P(TP)
│   ├── sl_model.py           # CatBoost для P(SL)
│   ├── calibration.py        # isotonic
│   ├── metrics.py            # logloss, ROC-AUC, PR-AUC, Brier
│   └── persistence.py        # версионирование моделей
│
├── validation/               # Phase 5: WF
│   ├── walk_forward.py       # генерация фолдов (pretrain 3Y, retrain 6M)
│   └── purging.py            # purge 10D между train и test
│
├── backtest/                 # Phase 7: бэктест
│   ├── engine.py             # ежедневный ребаланс, P&L Open(t+1)->Open(t+2)
│   ├── costs.py              # turnover, costs 3+3 bps
│   └── metrics.py            # CAGR, MDD, Sharpe, Sortino, Calmar
│
├── signals/                  # Phase 8: ежедневный сигнал
│   ├── updater.py            # догрузка до сегодня
│   └── daily.py              # сигнал по зафиксированной модели
│
├── agent/                    # LLM-фильтр новостей
│   ├── llm.py                # OpenAI Responses API + web_search
│   ├── filter.py             # блокировка сигнала по нарративу
│   └── run.py                # CLI: python -m agent.run
│
├── diags/                    # диагностика
│   └── backtest_diagnose.py  # score, turnover, costs, drawdown
│
├── artifacts/                # результаты (не в git)
│   ├── phase2/               # regime, top100
│   ├── phase3/               # targets
│   ├── phase5/               # wf_results, oos_predictions
│   ├── phase7/               # backtest_daily, summary
│   ├── signals/              # signal_YYYY-MM-DD.json
│   ├── agent/                # agent_YYYY-MM-DD.json
│   ├── models/               # v_YYYY-MM-DD/ (tp.cbm, sl.cbm, metadata)
│   └── live/                 # live_backtest
│
├── retrain.py                # переобучение + фиксация версии
├── run_phase5.py             # WF (stocks)
├── run_phase7.py             # бэктест
├── run_signal.py             # ежедневный сигнал
├── live_validate.py          # живая валидация после retrain
└── README.md
```

## Результаты (stocks-only, OOS 2017-10 .. 2026-11)

| Метрика | Значение |
|---|---|
| CAGR | **16.0%** |
| MDD | **−21.4%** |
| Sharpe | **1.18** |
| Sortino | **1.77** |
| Calmar | **0.75** |
| Volatility (ann) | 13.3% |
| Total return | 282% |
| Avg turnover | 0.50 / день |
| Total costs | 0.34 (34% капитала за 9 лет) |
| Avg exposure | 0.47 (47% в рынке, остальное cash) |
| Avg cash | 0.53 |

**По годам:** 2019 +25%, 2020 +20%, 2021 +20%, 2022 +11%, 2023 **+73%**, 2024 ~0%, 2025 +17%, 2026 −5% (неполный).

**ML-метрики (ROC-AUC по фолдам):**
- TP model: 0.61–0.69, стабильно > 0.6.
- SL model: 0.58–0.73, стабильно > 0.6.

## Ключевые решения (зафиксированы)

| Параметр | Значение | Где |
|---|---|---|
| Режим IMOEX | Close > EMA100 → stocks | `portfolio/regime.py` |
| TOP100 | сумма value за 60 торговых дней, point-in-time | `portfolio/universe.py` |
| Горизонт таргета H | 5 сессий | `configs/targets.yaml` |
| K_TP / K_SL | 1.5 / 1.0 | `configs/targets.yaml` |
| Vol window | 20 | `configs/targets.yaml` |
| Score | `p_tp × (1 − p_sl)` | `portfolio/ranking.py` |
| Корреляция | окно 20, порог 0.8, greedy | `configs/portfolio.yaml` |
| N assets | 4–8, выбор на train | `configs/portfolio.yaml` |
| Веса | равные, `exposure = breadth` | `portfolio/allocation.py` |
| Costs | 3 + 3 bps (round trip 6) | `configs/backtest.yaml` |
| Retrain | раз в 6 месяцев | `configs/wf.yaml` |
| Purge | 10 дней | `configs/wf.yaml` |

## Что проверено и отвергнуто

- **Futures-универс** — добавлен, протестирован, **откатан**. Причина: мало инструментов (13), высокая волатильность, turnover 0.56 при costs 3+3 bps съедает 56% капитала, CAGR падает до 4.75%, MDD −48.6%, Sharpe 0.33. Не подходит в текущей архитектуре.
- **`score_threshold` grid** `{0.0..0.20}` на train — протестирован, **откатан**. Не улучшает turnover/costs, CAGR падает до 14%.

## Запуск

### Первичная загрузка данных
```bash
python -m data.main update
python -m data.main validate
```

### Phase 2: режим, TOP100, фичи
```bash
python run_phase2.py regime
python run_phase2.py top100
python run_phase2.py features SBER    # smoke
```

### Phase 5: Walk-Forward
```bash
python run_phase5.py run
```
Сохраняет `artifacts/phase5/wf_results.json` и `oos_predictions.parquet`.

### Phase 7: бэктест
```bash
python run_phase7.py run
python diags/backtest_diagnose.py
```

### Retrain (фиксация версии модели)
```bash
python retrain.py run                    # на сегодня
python retrain.py run --date 2025-01-01  # на прошлую дату (для теста)
```
Сохраняет `artifacts/models/v_YYYY-MM-DD/` + `latest.json`.

### Ежедневный сигнал
```bash
python run_signal.py run                # апдейт + сигнал
python run_signal.py run --skip-update  # без догрузки
```

### Живая валидация
```bash
python live_validate.py run
```
Сравнивает поведение зафиксированной модели на данных **после** `retrain_date` с бэктестом.

### LLM-фильтр новостей (agent)
```bash
export OPENAI_API_KEY=sk-...   # или в configs/agent.yaml
python -m agent.run
```
Читает последний `signal_*.json`, по каждому тикеру через OpenAI `web_search` ищет новости, блокирует при нарративных рисках (суды, банкротство, отзыв аудитора, санкции и т.п.). Вес заблокированного уходит в cash. Веса не перераспределяет.

## Воспроизводимость (§18)

- `artifacts/data_version.json` — sha256, rows, start/end по каждому тикеру.
- `artifacts/phase5/wf_results.json` — границы WF-фолдов, `n_star`, `threshold_star`, метрики TP/SL.
- `artifacts/models/v_YYYY-MM-DD/metadata.json` — гиперпараметры, feature names, `n_star`, `threshold`, `train_start/end`, calibration method.
- `artifacts/phase7/backtest_daily.parquet` — ежедневная кривая, `target_w`, turnover, costs.
- `artifacts/signals/signal_YYYY-MM-DD.json` — ежедневный сигнал, версия модели, `p_tp`/`p_sl`/`score`/weights.

## Ограничения v0.1

1. **Только stocks.** Futures-универс откатан (см. выше).
2. **Не пройдена живая валидация.** Отсюда 0.5× объём.
3. **Модель переобучается вручную** (`retrain.py`), не автоматически по расписанию.
4. **Agent не подключён к провайдеру новостей** — использует OpenAI `web_search`, без отдельного news API.
5. **LLM-фильтр блокирует, но не перераспределяет вес** — freed weight уходит в cash (по решению: агент не лезет в веса).
6. **`live_validate.py`** — работает, но пока окно `retrain_date..today` пустое (retrain только что). Первый осмысленный прогон — через 1–2 месяца.
7. **`signal/daily.py`** пересчитывает фичи по всем тикерам каждый запуск — медленно, но корректно.

## Дальнейшие шаги

- **2026-11 → 2027-05:** мониторинг live-результатов, `live_validate.py` раз в месяц. Сравнение с бэктестом.
- **2027-05:** retrain на новых данных, **вердикт** по живой валидации:
  - **ОК** → объём 1.0×, продолжение.
  - **Не ОК** → анализ расхождений, возможный retrain с другими гиперпараметрами, или остановка.
- **Опционально:** подключить реальный news API в `agent/` вместо web_search (если web_search даст много шума/галлюцинаций).
- **Опционально:** futures-универс с другой архитектурой (другие таргеты/веса), не в v0.1.

## Контакты / ответственность

Проект — личный. Решения по торговле принимаются владельцем. Исполнение — вручную или через брокерский API (не входит в v0.1).

https://vk.ru/d.starkov713