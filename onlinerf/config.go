package onlinerf

import (
	"github.com/kudmo/onlinerf/onlinerf/features"
)


// PredictorConfig controls the forest model and feature pipeline.
type PredictorConfig struct {
	NumTrees            int
	MaxDepth            int
	MaxNodesPerTree     int
	HoeffdingSplitDelta float64
	MinSamplesPerLeaf   int
	UseDriftDetection   bool
	DriftAlpha          float64

	FeatureConfig features.FeatureConfig

	// Optional: custom embedder / normalizer factories can be plugged in by users.
	EmbedderFactory  features.EmbedderFactory
	NormalizerConfig features.NormalizerConfig
}

