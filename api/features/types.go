package features

// FeatureConfig contains settings for encoding numeric and categorical features.
// This struct is intentionally small and can be extended over time to describe
// how raw inputs should be embedded into a FeatureVector.
type FeatureConfig struct {
	// NormalizeNumeric controls whether numeric features should be normalized
	// (for example, via an online mean-variance normalizer) before being
	// passed into the forest.
	NormalizeNumeric bool
}

// FeatureType describes the semantic type of a single feature in the raw
// input space.
type FeatureType int

const (
	FeatureNumeric FeatureType = iota
	FeatureCategorical
)

// FeatureVector is a dense numeric representation of features after passing
// through the embedding / normalization pipeline. It is what the forest
// consumes in Predictor.Update and Predictor.Predict.
type FeatureVector []float64

// Sample represents a single labeled observation used for training in
// higher-level APIs. It is included here as a utility type.
type Sample struct {
	Features FeatureVector
	Label    bool
}
