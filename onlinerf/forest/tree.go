package forest

import "github.com/kudmo/onlinerf/onlinerf"

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
	Root   *Node
	Config TreeConfig
}

// NewTree creates a new tree with a single root leaf node.
func NewTree(cfg TreeConfig) *Tree {
	return &Tree{
		Root:   NewLeaf(0),
		Config: cfg,
	}
}

// Predict returns the probability estimate of the positive class.
func (t *Tree) Predict(fv predictor.FeatureVector) float64 {
	if t.Root == nil {
		return 0.5
	}

	node := t.Root
	for !node.IsLeaf {
		if fv.Values[node.FeatureIndex] <= node.Threshold {
			if node.Left == nil {
				break
			}
			node = node.Left
		} else {
			if node.Right == nil {
				break
			}
			node = node.Right
		}
	}
	return node.Predict(fv)
}

// Update performs an online update of the tree with a single sample.
func (t *Tree) Update(fv predictor.FeatureVector, label bool) {
	if t.Root == nil {
		t.Root = NewLeaf(0)
	}
	t.Root.Update(fv, label, t.Config)
}

