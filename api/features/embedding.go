package features

import (
	"hash/fnv"
	"sort"
)

// RawFeatureVector is a generic representation of input features before
// embedding. This can be extended to hold typed, heterogeneous data.
type RawFeatureVector struct {
	Numeric       []float64
	Categorical   []string
	FeatureTypes  []FeatureType
	OriginalIndex []int
}

// Embedder maps raw features to a dense numeric FeatureVector that is used
// by the forest. Implementations may apply hashing, one-hot encoding, etc.
type Embedder interface {
	Embed(raw RawFeatureVector) FeatureVector
}

// EmbedderFactory allows users to plug in custom embedders based on config.
type EmbedderFactory interface {
	NewEmbedder(cfg FeatureConfig) Embedder
}

// IdentityEmbedder is a simple implementation assuming that raw numeric
// features are already in the desired embedded space.
type IdentityEmbedder struct{}

func (IdentityEmbedder) Embed(raw RawFeatureVector) FeatureVector {
	return append([]float64(nil), raw.Numeric...)
}

// EmbedFeatures is a helper used by the example to turn simple maps of
// numeric and categorical features into a dense FeatureVector.
// For now it:
//   - sorts feature names to obtain a stable ordering
//   - appends all numeric values
//   - encodes each categorical value into a single numeric feature
//     using a deterministic hash.
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
