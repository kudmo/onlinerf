package onlinerf

import (
	"sync"

	"github.com/kudmo/onlinerf/onlinerf/aggregator"
	"github.com/kudmo/onlinerf/onlinerf/features"
	"github.com/kudmo/onlinerf/onlinerf/forest"
)

// Predictor is the main entry point: it owns the forest, feature pipeline
// and aggregation logic. It is safe for concurrent use if the caller
// serializes Update calls or uses external synchronization.
type Predictor struct {
	cfg PredictorConfig

	trees []*forest.Tree

	embedder   features.Embedder
	normalizer features.Normalizer
	agg        aggregator.Aggregator

	mu sync.RWMutex
}

// NewPredictor creates a new online random forest with the given configuration.
func NewPredictor(cfg PredictorConfig) *Predictor {
	p := &Predictor{
		cfg: cfg,
	}

	// Initialize feature pipeline.
	if cfg.EmbedderFactory != nil {
		p.embedder = cfg.EmbedderFactory.NewEmbedder(cfg.FeatureConfig)
	} else {
		p.embedder = features.IdentityEmbedder{}
	}

	if cfg.NormalizerConfig.Enable {
		// Placeholder: plug in real online normalizer implementation.
		p.normalizer = &features.NoOpNormalizer{}
	} else {
		p.normalizer = &features.NoOpNormalizer{}
	}

	// Default aggregator: mean over trees.
	p.agg = aggregator.MeanAggregator{}

	// Initialize trees.
	treeCfg := forest.TreeConfig{
		MaxDepth:            cfg.MaxDepth,
		MaxNodes:            cfg.MaxNodesPerTree,
		HoeffdingSplitDelta: cfg.HoeffdingSplitDelta,
		MinSamplesPerLeaf:   cfg.MinSamplesPerLeaf,
		UseDriftDetection:   cfg.UseDriftDetection,
		DriftAlpha:          cfg.DriftAlpha,
	}

	p.trees = make([]*forest.Tree, cfg.NumTrees)
	for i := 0; i < cfg.NumTrees; i++ {
		p.trees[i] = forest.NewTree(treeCfg)
	}

	return p
}

// Predict returns the probability of class 1 for the given already-embedded
// feature vector.
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

// Update performs an online training update with a single sample.
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

