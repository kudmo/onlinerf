package forest

import "github.com/kudmo/onlinerf/onlinerf"

// Node represents a single Hoeffding tree node.
// This is a minimal skeleton; split criteria and feature statistics
// can be extended without changing the public predictor API.
type Node struct {
	Depth int

	// Split definition (for numeric features we use threshold; for
	// categorical it can be equality-based, etc.).
	FeatureIndex int
	Threshold    float64
	IsLeaf       bool

	Left  *Node
	Right *Node

	Stats ClassStats
}

// NewLeaf creates a new leaf node at the given depth.
func NewLeaf(depth int) *Node {
	return &Node{
		Depth:  depth,
		IsLeaf: true,
	}
}

// Predict returns the probability estimate at this node.
func (n *Node) Predict(_ predictor.FeatureVector) float64 {
	// For now, we simply use the empirical positive probability.
	return n.Stats.Prob()
}

// Update updates statistics along the path and potentially triggers a split.
// The exact Hoeffding split logic is left for future extension.
func (n *Node) Update(fv predictor.FeatureVector, label bool, cfg TreeConfig) {
	n.Stats.Update(label)

	if n.IsLeaf {
		// TODO: implement Hoeffding bound-based split decision.
		return
	}

	// Route down the tree if already split.
	if fv.Values[n.FeatureIndex] <= n.Threshold {
		if n.Left != nil {
			n.Left.Update(fv, label, cfg)
		}
	} else {
		if n.Right != nil {
			n.Right.Update(fv, label, cfg)
		}
	}
}

