package onlinerf

import (
	"sync"

	"github.com/kudmo/onlinerf/onlinerf/aggregator"
	"github.com/kudmo/onlinerf/onlinerf/features"
	"github.com/kudmo/onlinerf/onlinerf/forest"
)

// Predictor is the main entry point: it owns the forest,
// feature pipeline and aggregation logic.
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
