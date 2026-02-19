package forest

type Stats struct {
	Pos int
	Neg int
}

func (s *Stats) Update(label bool) {
	if label {
		s.Pos++
	} else {
		s.Neg++
	}
}

func (s *Stats) Total() int {
	return s.Pos + s.Neg
}

func (s *Stats) Prob() float64 {
	total := s.Total()
	if total == 0 {
		return 0.5
	}
	return float64(s.Pos) / float64(total)
}

func gini(pos, neg int) float64 {
	total := pos + neg
	if total == 0 {
		return 0
	}
	p := float64(pos) / float64(total)
	return 1 - p*p - (1-p)*(1-p)
}

type FeatureStat struct {
	Threshold float64

	LeftPos  int
	LeftNeg  int
	RightPos int
	RightNeg int
}

func (f *FeatureStat) Update(value float64, label bool) {
	if value <= f.Threshold {
		if label {
			f.LeftPos++
		} else {
			f.LeftNeg++
		}
	} else {
		if label {
			f.RightPos++
		} else {
			f.RightNeg++
		}
	}
}

func (f *FeatureStat) GiniGain(parentPos, parentNeg int) float64 {
	parentGini := gini(parentPos, parentNeg)

	leftTotal := f.LeftPos + f.LeftNeg
	rightTotal := f.RightPos + f.RightNeg
	total := leftTotal + rightTotal

	if total == 0 {
		return 0
	}

	leftWeight := float64(leftTotal) / float64(total)
	rightWeight := float64(rightTotal) / float64(total)

	weighted :=
		leftWeight*gini(f.LeftPos, f.LeftNeg) +
			rightWeight*gini(f.RightPos, f.RightNeg)

	return parentGini - weighted
}
