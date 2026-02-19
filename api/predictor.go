package onlinerf

import (
	"sync"

	"github.com/kudmo/onlinerf/internal/aggregator"
	"github.com/kudmo/onlinerf/internal/forest"
	"github.com/kudmo/onlinerf/api/features"
)

// Predictor is the main entry point to the online random forest API.
// It owns the ensemble of Hoeffding trees, the feature pipeline
// (embedder + normalizer) and the aggregation logic.
//
// A Predictor is safe for concurrent use from multiple goroutines.
type Predictor struct {
	cfg PredictorConfig

	trees []*forest.Tree

	embedder   features.Embedder
	normalizer features.Normalizer
	agg        aggregator.Aggregator

	numFeatures int
	initialized bool

	mu sync.RWMutex
}

// NewPredictor creates a new online random forest with the given configuration.
//
// The forest expects pre-embedded feature vectors of length cfg.NumFeatures.
// If cfg.EmbedderFactory is nil, IdentityEmbedder is used and callers are
// responsible for constructing FeatureVector values directly.
//
// Example:
//
//	cfg := onlinerf.PredictorConfig{
//		NumTrees:        20,
//		NumFeatures:     16,
//		MaxDepth:        10,
//		MaxNodesPerTree: 500,
//	}
//	model := onlinerf.NewPredictor(cfg)
//	prob := model.Predict(features.FeatureVector{0.1, 0.5 /* ... */})
//	_ = prob
func NewPredictor(cfg PredictorConfig) *Predictor {
	p := &Predictor{
		cfg: cfg,
	}

	// Feature pipeline
	if cfg.EmbedderFactory != nil {
		p.embedder = cfg.EmbedderFactory.NewEmbedder(cfg.FeatureConfig)
	} else {
		p.embedder = features.IdentityEmbedder{}
	}

	if cfg.NormalizerConfig.Enable {
		p.normalizer = &features.NoOpNormalizer{}
	} else {
		p.normalizer = &features.NoOpNormalizer{}
	}

	p.agg = aggregator.MeanAggregator{}

	p.initForest(cfg.NumFeatures)
	return p
}

func (p *Predictor) initForest(numFeatures int) {

	treeCfg := forest.TreeConfig{
		MaxDepth:            p.cfg.MaxDepth,
		MaxNodes:            p.cfg.MaxNodesPerTree,
		HoeffdingSplitDelta: p.cfg.HoeffdingSplitDelta,
		MinSamplesPerLeaf:   p.cfg.MinSamplesPerLeaf,
	}

	p.trees = make([]*forest.Tree, p.cfg.NumTrees)

	for i := 0; i < p.cfg.NumTrees; i++ {
		p.trees[i] = forest.NewTree(treeCfg, numFeatures)
	}

	p.numFeatures = numFeatures
	p.initialized = true
}

// Predict returns the estimated probability of the positive class (label == true)
// for the given embedded feature vector.
//
// The input fv must have length equal to cfg.NumFeatures. It is assumed to be
// already embedded using the same scheme that was used during training.
//
// Example:
//
//	fv := features.FeatureVector{0.3, 0.8 /* ... */ }
//	p := onlinerf.NewPredictor(cfg)
//	score := p.Predict(fv)
func (p *Predictor) Predict(fv features.FeatureVector) float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()

	// Apply feature pipeline.
	embedded := fv
	if p.normalizer != nil {
		embedded = p.normalizer.Transform(embedded)
	}

	probs := make([]float64, 0, len(p.trees))
	for _, t := range p.trees {
		if t == nil {
			continue
		}
		probs = append(probs, t.Predict(embedded))
	}

	return p.agg.Aggregate(probs)
}

// Update performs an online training update with a single labeled sample.
//
// The feature vector fv must have the same dimensionality as in Predict.
// label should be true for the positive class and false otherwise.
// Internally the sample is passed through the optional normalizer (if enabled)
// and then applied to every tree in the forest.
//
// Example:
//
//	for _, s := range stream {
//		fv := embed(s)         // user-defined embedding into FeatureVector
//		model.Update(fv, s.Y)  // online update
//	}
func (p *Predictor) Update(fv features.FeatureVector, label bool) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.normalizer != nil {
		p.normalizer.Update(fv)
	}
	embedded := fv
	if p.normalizer != nil {
		embedded = p.normalizer.Transform(embedded)
	}

	for _, t := range p.trees {
		if t == nil {
			continue
		}
		t.Update(embedded, label)
	}
}
