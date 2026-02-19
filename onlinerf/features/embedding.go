package features

import "github.com/kudmo/onlinerf/onlinerf"

// RawFeatureVector is a generic representation of input features before
// embedding. This can be extended to hold typed, heterogeneous data.
type RawFeatureVector struct {
	Numeric       []float64
	Categorical   []string
	FeatureTypes  []predictor.FeatureType
	OriginalIndex []int
}

// Embedder maps raw features to a dense numeric FeatureVector that is used
// by the forest. Implementations may apply hashing, one-hot encoding, etc.
type Embedder interface {
	Embed(raw RawFeatureVector) predictor.FeatureVector
}

// EmbedderFactory allows users to plug in custom embedders based on config.
type EmbedderFactory interface {
	NewEmbedder(cfg predictor.FeatureConfig) Embedder
}

// IdentityEmbedder is a simple implementation assuming that raw numeric
// features are already in the desired embedded space.
type IdentityEmbedder struct{}

func (IdentityEmbedder) Embed(raw RawFeatureVector) predictor.FeatureVector {
	return predictor.FeatureVector{Values: append([]float64(nil), raw.Numeric...)}
}

