package forest

import "github.com/kudmo/onlinerf/onlinerf/features"

// TreeConfig controls a single Hoeffding tree within the forest.
type TreeConfig struct {
	MaxDepth            int
	MaxNodes            int
	HoeffdingSplitDelta float64
	MinSamplesPerLeaf   int
	UseDriftDetection   bool
	DriftAlpha          float64
}

// Tree is an online Hoeffding decision tree used as a base learner
// in the online random forest.
type Tree struct {
	Root        *Node
	Config      TreeConfig
	NumFeatures int
	NodeCount   int
}

func NewTree(cfg TreeConfig, numFeatures int) *Tree {
	return &Tree{
		Config:      cfg,
		NumFeatures: numFeatures,
	}
}

func (t *Tree) initRoot(fv features.FeatureVector) {
	bootstrap := make(features.FeatureVector, len(fv))
	copy(bootstrap, fv)

	t.Root = NewLeaf(0, t.NumFeatures, bootstrap)
	t.NodeCount = 1
}

// Predict returns the probability estimate of the positive class.
func (t *Tree) Predict(fv features.FeatureVector) float64 {
	if t.Root == nil {
		return 0.5
	}

	node := t.Root
	for !node.IsLeaf {
		node = node.ChooseChild(fv)
	}
	return node.Predict(fv)
}

// Update performs an online update of the tree with a single sample.
func (t *Tree) Update(fv features.FeatureVector, label bool) {

	if t.Root == nil {
		t.initRoot(fv)
	}

	t.Root.Update(
		features.FeatureVector(fv),
		label,
		t.Config,
	)
}
