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

See `examples/synthetic_simple/main.go` for the full example.


