// Package onlinerf provides an online random forest (ensemble of Hoeffding
// trees) for binary classification with support for streaming updates and a
// simple feature pipeline.
package onlinerf

import "github.com/kudmo/onlinerf/api/features"

// PredictorConfig controls the online random forest model and the feature
// processing pipeline used by a Predictor.
//
// Most users will:
//   - choose the number of trees and maximum depth / node budget
//   - decide whether to enable concept-drift detection
//   - configure the feature pipeline via FeatureConfig / NormalizerConfig
//
// All fields are exported so that configurations can be serialized if needed.
type PredictorConfig struct {
	// NumTrees is the size of the forest (number of Hoeffding trees).
	// Larger values increase accuracy at the cost of memory and CPU.
	NumTrees            int

	// NumFeatures is the dimensionality of the embedded feature vectors
	// that will be passed into the forest. This must match the length of
	// the FeatureVector values you pass to Predictor.Update / Predictor.Predict.
	NumFeatures         int

	// MaxDepth is the maximum depth allowed for each tree. Limiting depth
	// bounds memory usage and acts as a regularizer.
	MaxDepth            int

	// MaxNodesPerTree is a hard cap on the number of nodes in each tree.
	// When used together with MaxDepth it controls memory consumption.
	MaxNodesPerTree     int

	// HoeffdingSplitDelta is the confidence parameter used in the Hoeffding
	// bound when deciding whether the best splitting feature is statistically
	// better than the runner-up. Smaller values make splits more conservative.
	HoeffdingSplitDelta float64

	// MinSamplesPerLeaf is the minimum number of samples that must be seen
	// by a leaf before it becomes eligible for splitting.
	MinSamplesPerLeaf   int

	// UseDriftDetection enables concept-drift detection at the leaf level
	// using an ADWIN-like detector. When drift is detected, the leaf is
	// reset to forget stale statistics.
	UseDriftDetection   bool

	// DriftAlpha is the significance level controlling the sensitivity of
	// the drift detector. Smaller values make it harder to trigger drift.
	DriftAlpha          float64

	// FeatureConfig configures how raw numeric / categorical inputs are
	// mapped into an embedded FeatureVector by an Embedder.
	FeatureConfig features.FeatureConfig

	// EmbedderFactory optionally allows plugging in a custom feature
	// embedding implementation. If nil, IdentityEmbedder is used.
	EmbedderFactory  features.EmbedderFactory

	// NormalizerConfig controls whether and how online feature normalization
	// is applied before passing feature vectors into the forest.
	NormalizerConfig features.NormalizerConfig
}
