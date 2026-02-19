package features

// NormalizerConfig contains configuration for online normalization of numeric
// features prior to feeding them into the forest.
type NormalizerConfig struct {
	Enable bool
}

// Normalizer performs online normalization of feature vectors (for example,
// mean-variance scaling). Implementations are expected to be safe for use in
// streaming settings.
type Normalizer interface {
	// Update updates internal statistics with a new sample.
	Update(fv FeatureVector)
	// Transform applies the current normalization to the given feature vector.
	Transform(fv FeatureVector) FeatureVector
}

// NoOpNormalizer leaves features unchanged. It is useful when normalization
// is disabled but a Normalizer implementation is still required by the API.
type NoOpNormalizer struct{}

func (n NoOpNormalizer) Update(_ FeatureVector) {}

func (n NoOpNormalizer) Transform(fv FeatureVector) FeatureVector {
	return fv
}

