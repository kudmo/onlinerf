## onlinerf – Online Random Forest / Hoeffding Trees

Lightweight Go 1.21 library for **online binary classification** using an ensemble
of Hoeffding trees (online random forest). The library is designed for streaming
workloads where data arrives continuously and the model must be updated in-place.

- **Online learning** via `Update` without rebuilding the model
- **Numeric and categorical features** via the `features` package
- **Explicit memory control** (number of trees, max depth, max nodes)
- **Fast `Predict`** for production usage

### Install

```bash
go get github.com/kudmo/onlinerf
```

### Core concepts

- **Predictor** (`onlinerf.Predictor`): main entry point that owns the forest,
  feature pipeline, and aggregation logic.
- **Hoeffding trees** (`internal/forest`): online decision trees that decide
  to split using Hoeffding bounds.
- **Concept drift detection** (`DriftDetector`): optional ADWIN-like detector
  that can reset leaves when the distribution changes.
- **Features pipeline** (`api/features`):
  - `Embedder` / `EmbedderFactory` to map raw inputs to dense vectors
  - `Normalizer` to optionally normalize numeric features online

### Quick start

The simplest way to use the library is to construct fixed-size feature vectors
and feed them directly into the predictor:

```go
package main

import (
	onlinerf "github.com/kudmo/onlinerf/api"
	"github.com/kudmo/onlinerf/api/features"
)

func main() {
	const numFeatures = 7

	cfg := onlinerf.PredictorConfig{
		NumTrees:            10,
		NumFeatures:         numFeatures,
		MaxDepth:            20,
		MaxNodesPerTree:     300,
		HoeffdingSplitDelta: 0.1,
		MinSamplesPerLeaf:   5,
		UseDriftDetection:   false,
	}

	model := onlinerf.NewPredictor(cfg)

	// Example: one sample encoded into a FeatureVector.
	fv := features.FeatureVector{0.5, 0.7 /* ... up to numFeatures ... */}
	label := true

	// Online update.
	model.Update(fv, label)

	// Prediction: probability of the positive class.
	score := model.Predict(fv)
	_ = score
}
```

See `examples/synthetic_simple/main.go` for a more complete synthetic example
with evaluation.

### Configuration

The main configuration struct is `onlinerf.PredictorConfig`:

- **NumTrees**: number of Hoeffding trees in the ensemble (capacity vs. cost).
- **NumFeatures**: dimensionality of the embedded feature vectors.
- **MaxDepth**: maximum depth of each tree (controls overfitting and memory).
- **MaxNodesPerTree**: hard cap on the number of nodes per tree.
- **HoeffdingSplitDelta**: confidence parameter for the Hoeffding bound;
  smaller values make splits more conservative.
- **MinSamplesPerLeaf**: minimum number of samples at a leaf before considering
  a split.
- **UseDriftDetection**: enable per-leaf concept drift detection.
- **DriftAlpha**: significance level for the drift detector.
- **FeatureConfig**: configuration for the feature embedding logic.
- **EmbedderFactory**: optional custom embedder factory; falls back to
  `IdentityEmbedder` if nil.
- **NormalizerConfig**: configuration for online normalization.

### Features and embedding

The `features` package defines the basic building blocks for representing and
transforming input data:

- **FeatureVector**: `[]float64` embedded representation consumed by the forest.
- **RawFeatureVector**: raw numeric + categorical features with metadata.
- **Embedder**: interface that maps `RawFeatureVector` to `FeatureVector`.
- **EmbedderFactory**: creates embedders from a `FeatureConfig`.
- **Normalizer** / `NormalizerConfig`: online normalization of numeric features.

For simple use cases you can rely on the helper `EmbedFeatures`:

```go
numeric := map[string]float64{"cpu": 0.7, "mem": 0.4}
categorical := map[string]string{"env": "prod", "role": "api"}

fv := features.EmbedFeatures(numeric, categorical)
// fv is a stable, dense FeatureVector that can be passed to Update/Predict.
```

Advanced users can implement their own `Embedder` and `EmbedderFactory` to
support richer schemas or more complex encodings.

### Streaming / online learning pattern

Typical integration into a streaming system looks like this:

```go
for sample := range stream {
	fv := myEmbed(sample)      // -> features.FeatureVector
	model.Update(fv, sample.Y) // online training
}
```

You can simultaneously call `Predict` in other goroutines; the predictor uses
an internal `sync.RWMutex` and is safe for concurrent use.

### Examples

- `examples/synthetic_simple/main.go` – synthetic binary classification example
  with mixed numeric/categorical features and basic evaluation metrics.

### Status

The public API (`onlinerf.Predictor`, `PredictorConfig`, and the `features`
package) is intended to be stable, while internal packages (`internal/forest`,
`internal/aggregator`) may evolve. Contributions and suggestions are welcome.

