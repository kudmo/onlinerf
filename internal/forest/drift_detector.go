package forest

import "math"

// DriftDetector tracks a stream of binary labels and signals when concept
// drift is detected according to an ADWIN-like statistical test.
type DriftDetector struct {
	window     []float64
	width      int
	sum        float64
	sumSquares float64
	alpha      float64
	minWindow  int
}

// NewADWIN creates a new drift detector with the given significance level alpha.
// Smaller alpha makes the detector more conservative.
func NewADWIN(alpha float64) *DriftDetector {
	return &DriftDetector{
		window:     make([]float64, 0, 100),
		width:      0,
		sum:        0,
		sumSquares: 0,
		alpha:      alpha,
		minWindow:  10,
	}
}

// Add feeds a new binary observation into the detector.
// It returns true if drift is detected and the internal window was reset.
func (d *DriftDetector) Add(value bool) bool {
	var x float64
	if value {
		x = 1.0
	} else {
		x = 0.0
	}

	// Append to the current sliding window.
	d.window = append(d.window, x)
	d.width++
	d.sum += x
	d.sumSquares += x * x

	// If there is not enough data yet, never signal drift.
	if d.width < d.minWindow {
		return false
	}

	// Check for drift.
	return d.detect()
}

func (d *DriftDetector) detect() bool {
	n := float64(d.width)
	mean := d.sum / n
	variance := d.sumSquares/n - mean*mean
	// Guard against tiny or negative variance due to numerical noise.
	if variance < 1e-10 {
		variance = 1e-10
	}

	// Compute epsilon using a Hoeffding-style bound for the sample mean.
	eps := math.Sqrt(2 * variance * math.Log(2.0/d.alpha) / n)

	// Split the window at different cut points and test for a significant
	// difference between the left and right means.
	for i := d.minWindow; i <= d.width-d.minWindow; i++ {
		leftSum := 0.0
		for j := 0; j < i; j++ {
			leftSum += d.window[j]
		}
		rightSum := d.sum - leftSum

		nLeft := float64(i)
		nRight := float64(d.width - i)

		meanLeft := leftSum / nLeft
		meanRight := rightSum / nRight

		if math.Abs(meanLeft-meanRight) > eps {
			// Drift detected — reset the window and report true.
			d.reset()
			return true
		}
	}

	return false
}

func (d *DriftDetector) reset() {
	d.window = d.window[:0]
	d.width = 0
	d.sum = 0
	d.sumSquares = 0
}
