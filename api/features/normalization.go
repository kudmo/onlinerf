package features


// NormalizerConfig contains configuration for online normalization of numeric features.
type NormalizerConfig struct {
	Enable bool
}

// Normalizer performs online normalization (e.g., mean-variance scaling).
type Normalizer interface {
	// Update updates internal statistics with a new sample.
	Update(fv FeatureVector)
	// Transform applies the current normalization to the given feature vector.
	Transform(fv FeatureVector) FeatureVector
}

// NoOpNormalizer leaves features unchanged.
type NoOpNormalizer struct{}

func (n NoOpNormalizer) Update(_ FeatureVector) {}

func (n NoOpNormalizer) Transform(fv FeatureVector) FeatureVector {
	return fv
}

