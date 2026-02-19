package onlinerf

import (
	"github.com/kudmo/onlinerf/onlinerf/features"
)

// FeatureConfig contains settings for encoding numeric and categorical features.
// This is intentionally generic and can be extended in the future.
type FeatureConfig struct {
	// Example options (extend as needed):
	// - whether to standardize numeric features
	// - hashing dimensions for categorical features
	// - per-feature overrides

	NormalizeNumeric bool
}

// PredictorConfig controls the forest model and feature pipeline.
type PredictorConfig struct {
	NumTrees            int
	MaxDepth            int
	MaxNodesPerTree     int
	HoeffdingSplitDelta float64
	MinSamplesPerLeaf   int
	UseDriftDetection   bool
	DriftAlpha          float64

	FeatureConfig FeatureConfig

	// Optional: custom embedder / normalizer factories can be plugged in by users.
	EmbedderFactory  features.EmbedderFactory
	NormalizerConfig features.NormalizerConfig
}

