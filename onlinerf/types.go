package onlinerf

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

