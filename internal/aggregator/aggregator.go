package aggregator

// Aggregator combines per-tree probabilities into a single ensemble prediction.
type Aggregator interface {
	Aggregate(probs []float64) float64
}

// MeanAggregator averages probabilities from all trees.
type MeanAggregator struct{}

func (MeanAggregator) Aggregate(probs []float64) float64 {
	if len(probs) == 0 {
		return 0.5
	}
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	return sum / float64(len(probs))
}

// MaxAggregator selects the maximum probability from trees.
type MaxAggregator struct{}

func (MaxAggregator) Aggregate(probs []float64) float64 {
	if len(probs) == 0 {
		return 0.5
	}
	max := probs[0]
	for _, p := range probs[1:] {
		if p > max {
			max = p
		}
	}
	return max
}

