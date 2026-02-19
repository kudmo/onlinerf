package forest

import (
	"math"

	"github.com/kudmo/onlinerf/onlinerf/features"
)

// Node represents a single Hoeffding tree node.
// This is a minimal skeleton; split criteria and feature statistics
// can be extended without changing the public predictor API.
type Node struct {
	IsLeaf bool
	Depth  int

	Stats Stats

	FeatureStats map[int]*FeatureStat

	SplitFeature int
	Threshold    float64

	Left  *Node
	Right *Node

	// DRIFT DETECTION
	DriftDetector *DriftDetector
}

func NewLeaf(depth int, numFeatures int, bootstrap features.FeatureVector) *Node {
	fs := make(map[int]*FeatureStat)

	for i := 0; i < numFeatures; i++ {
		fs[i] = &FeatureStat{
			Threshold: bootstrap[i], // bootstrap median-like
		}
	}

	return &Node{
		IsLeaf:       true,
		Depth:        depth,
		FeatureStats: fs,
	}
}

// Predict returns the probability estimate at this node.
func (n *Node) Predict(fv features.FeatureVector) float64 {
	if !n.IsLeaf {
		panic("Predict should only be called on leaf nodes")
	}
	return n.Stats.Prob()
}

func (n *Node) ChooseChild(fv features.FeatureVector) *Node {
	if fv[n.SplitFeature] <= n.Threshold {
		return n.Left
	}
	return n.Right
}

func (n *Node) Update(fv features.FeatureVector, label bool, cfg TreeConfig) {
	if n.IsLeaf {
		// 1. Обновляем статистику
		n.Stats.Update(label)

		// 2. Проверяем дрейф
		if n.DriftDetector != nil {
			drift := n.DriftDetector.Add(label)
			if drift {
				// Дрейф обнаружен — сбросить лист
				n.Left = nil
				n.Right = nil
				n.IsLeaf = true
				n.Stats = Stats{}
				// Перезапускаем детектор
				if cfg.UseDriftDetection {
					n.DriftDetector.reset()
				}
				return
			}
		}
		for i, v := range fv {
			n.FeatureStats[i].Update(v, label)
		}

		// 2. Проверка условий split
		if n.Stats.Total() < cfg.MinSamplesPerLeaf {
			return
		}

		if n.Depth >= cfg.MaxDepth {
			return
		}

		n.trySplit(cfg)
		return
	}

	// не лист — спускаемся
	child := n.ChooseChild(fv)
	child.Update(fv, label, cfg)
}

func (n *Node) trySplit(cfg TreeConfig) {
	// TODO: decline split if MAX NODES reached (need to pass tree-level node count and increment on split)

	total := n.Stats.Total()
	parentPos := n.Stats.Pos
	parentNeg := n.Stats.Neg

	var bestFeature int = -1
	var bestGain float64 = -1
	var secondBest float64 = -1

	for i, fs := range n.FeatureStats {
		gain := fs.GiniGain(parentPos, parentNeg)

		if gain > bestGain {
			secondBest = bestGain
			bestGain = gain
			bestFeature = i
		} else if gain > secondBest {
			secondBest = gain
		}
	}

	if bestFeature == -1 {
		return
	}

	// Hoeffding bound
	R := 1.0
	epsilon := math.Sqrt(
		(R * R * math.Log(1.0/cfg.HoeffdingSplitDelta)) /
			(2.0 * float64(total)),
	)

	if bestGain-secondBest > epsilon {

		bestFS := n.FeatureStats[bestFeature]

		n.IsLeaf = false
		n.SplitFeature = bestFeature
		n.Threshold = bestFS.Threshold

		numFeatures := len(n.FeatureStats)

		n.Left = NewLeaf(n.Depth+1, numFeatures, make(features.FeatureVector, numFeatures))
		n.Right = NewLeaf(n.Depth+1, numFeatures, make(features.FeatureVector, numFeatures))

		// очищаем статистику текущего листа
		n.FeatureStats = nil
	}
}
