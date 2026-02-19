package forest

// ClassStats keeps online statistics for binary labels at a node.
type ClassStats struct {
	CountTotal int
	CountPos   int
}

func (s *ClassStats) Update(label bool) {
	s.CountTotal++
	if label {
		s.CountPos++
	}
}

// Prob returns the empirical probability of the positive class.
func (s *ClassStats) Prob() float64 {
	if s.CountTotal == 0 {
		return 0.5
	}
	return float64(s.CountPos) / float64(s.CountTotal)
}

