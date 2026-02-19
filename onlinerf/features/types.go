package features

// FeatureConfig contains settings for encoding numeric and categorical features.
// This is intentionally generic and can be extended in the future.
type FeatureConfig struct {
	// Example options (extend as needed):
	// - whether to standardize numeric features
	// - hashing dimensions for categorical features
	// - per-feature overrides

	NormalizeNumeric bool
}

// FeatureType describes the type of a single feature.
type FeatureType int

const (
	FeatureNumeric FeatureType = iota
	FeatureCategorical
)

// FeatureVector is a dense embedded numeric representation of features
// after passing through the feature pipeline.
type FeatureVector struct {
	Values []float64
}

// Sample represents a single labeled observation used for training.
type Sample struct {
	Features FeatureVector
	Label    bool
}

