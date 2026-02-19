## onlinerf – Online Random Forest / Hoeffding Trees (RU)

Лёгкая библиотека на Go 1.21 для онлайн‑бинарной классификации на основе ансамбля деревьев (Online Random Forest / Hoeffding Trees) с поддержкой:

- **онлайн‑обучения** (`Update`) без пересборки модели
- **числовых и категориальных признаков** через модуль `features`
- **ограничения памяти** (число деревьев, глубина, макс. узлов)
- **быстрого `Predict`** для использования в продакшене

### Установка

```bash
go get github.com/kudmo/onlinerf
```

### Базовый пример

Полный пример: `examples/basic/main.go`.

```go
cfg := onlinerf.PredictorConfig{
    NumTrees:            10,
    MaxDepth:            20,
    HoeffdingSplitDelta: 0.05,
    MinSamplesPerLeaf:   5,
    UseDriftDetection:   true,
    DriftAlpha:          0.01,
}

pred := onlinerf.NewPredictor(cfg)

numeric := map[string]float64{
    "cpu_util": 0.75,
    "mem_util": 0.60,
}
categorical := map[string]string{
    "reaction_type": "HPA",
}

fv := features.EmbedFeatures(numeric, categorical)

p := pred.Predict(fv)
pred.Update(fv, true)
```

Запуск примера:

```bash
go run ./examples/basic
```

### Потоковое, батч‑ и синтетическое обучение

- Online‑пример: `examples/online_training/main.go` — обучение по стриму событий с выводом before/after.
- Batch‑пример: `examples/batch_training/main.go` — обучение на батче и расчёт метрик (accuracy, precision, recall, F1, log loss).
- Простой синтетический пример: `examples/synthetic_simple/main.go` — большая выборка с простой зависимостью (2 числовые + 2 категориальные фичи) и оценкой accuracy/средних вероятностей.

---

## onlinerf – Online Random Forest / Hoeffding Trees (EN)

Lightweight Go 1.21 library for online binary classification using an ensemble of Hoeffding trees (Online Random Forest) with:

- **online learning** via `Update`
- **numeric and categorical features** via the `features` module
- **memory constraints** (number of trees, max depth, max nodes)
- **fast `Predict`** for production usage

### Install

```bash
go get github.com/kudmo/onlinerf
```

### Basic usage

See `examples/basic/main.go` for the full example.

```go
cfg := onlinerf.PredictorConfig{
    NumTrees:            10,
    MaxDepth:            20,
    HoeffdingSplitDelta: 0.05,
    MinSamplesPerLeaf:   5,
    UseDriftDetection:   true,
    DriftAlpha:          0.01,
}

pred := onlinerf.NewPredictor(cfg)

numeric := map[string]float64{"cpu_util": 0.75, "mem_util": 0.60}
categorical := map[string]string{"reaction_type": "HPA"}

fv := features.EmbedFeatures(numeric, categorical)

p := pred.Predict(fv)
pred.Update(fv, true)
```

Run example:

```bash
go run ./examples/basic
```


