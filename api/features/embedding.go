package features

import (
	"hash/fnv"
	"sort"
)

// RawFeatureVector is a generic representation of input features before
// embedding. It can hold heterogeneous data types and keeps track of the
// original feature ordering.
type RawFeatureVector struct {
	Numeric       []float64
	Categorical   []string
	FeatureTypes  []FeatureType
	OriginalIndex []int
}

// Embedder maps raw features to a dense numeric FeatureVector that is used
// by the forest. Implementations may apply hashing, one-hot encoding, or
// other feature engineering techniques.
type Embedder interface {
	Embed(raw RawFeatureVector) FeatureVector
}

// EmbedderFactory allows users to plug in custom embedders based on FeatureConfig.
// This is how the onlinerf package wires user-defined embedding logic into the
// Predictor.
type EmbedderFactory interface {
	NewEmbedder(cfg FeatureConfig) Embedder
}

// IdentityEmbedder is a simple implementation that assumes raw numeric
// features are already in the desired embedded space and simply copies them.
type IdentityEmbedder struct{}

func (IdentityEmbedder) Embed(raw RawFeatureVector) FeatureVector {
	return append([]float64(nil), raw.Numeric...)
}

// EmbedFeatures is a helper used by examples to turn simple maps of numeric
// and categorical features into a dense FeatureVector.
//
// The current strategy is:
//   - sort numeric feature names and append their values in that order
//   - sort categorical feature names and encode each categorical value into
//     a single numeric feature using a deterministic hash in [0,1).
//
// Example:
//
//	numeric := map[string]float64{"cpu": 0.7, "mem": 0.4}
//	categorical := map[string]string{"env": "prod", "role": "api"}
//	fv := features.EmbedFeatures(numeric, categorical)
//	_ = fv
func EmbedFeatures(numeric map[string]float64, categorical map[string]string) FeatureVector {
	values := make([]float64, 0, len(numeric)+len(categorical))

	// Numeric features in sorted key order.
	numKeys := make([]string, 0, len(numeric))
	for k := range numeric {
		numKeys = append(numKeys, k)
	}
	sort.Strings(numKeys)
	for _, k := range numKeys {
		values = append(values, numeric[k])
	}

	// Categorical features: hash value into [0,1).
	catKeys := make([]string, 0, len(categorical))
	for k := range categorical {
		catKeys = append(catKeys, k)
	}
	sort.Strings(catKeys)
	for _, k := range catKeys {
		v := categorical[k]
		h := fnv.New64a()
		_, _ = h.Write([]byte(v))
		hashVal := h.Sum64()
		const denom = float64(^uint64(0))
		values = append(values, float64(hashVal)/denom)
	}

	return values
}
