package forest

import "math"

// DriftDetector отслеживает поток бинарных меток и сигнализирует о дрейфе
type DriftDetector struct {
	window     []float64
	width      int
	sum        float64
	sumSquares float64
	alpha      float64
	minWindow  int
}

// NewADWIN создаёт новый детектор дрейфа
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

// Add добавляет новый бинарный пример (0.0 или 1.0) в детектор.
// Возвращает true, если обнаружен дрейф.
func (d *DriftDetector) Add(value bool) bool {
	var x float64
	if value {
		x = 1.0
	} else {
		x = 0.0
	}

	// добавляем в окно
	d.window = append(d.window, x)
	d.width++
	d.sum += x
	d.sumSquares += x * x

	// если слишком мало данных — не проверяем
	if d.width < d.minWindow {
		return false
	}

	// проверка дрейфа
	return d.detect()
}

func (d *DriftDetector) detect() bool {
	n := float64(d.width)
	mean := d.sum / n
	variance := d.sumSquares/n - mean*mean
	// защита от отрицательной дисперсии
	if variance < 1e-10 {
		variance = 1e-10
	}

	// вычисляем ε по формуле Hoeffding для биномиального среднего
	eps := math.Sqrt(2 * variance * math.Log(2.0/d.alpha) / n)

	// разбиваем окно пополам и проверяем статистическое различие
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
			// дрейф обнаружен → обнуляем окно
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
