package onlinerf

import (
	"testing"

	"github.com/kudmo/onlinerf/api/features"
)

// TestNewPredictorInitialization проверяет, что Predictor корректно
// инициализируется и создаёт ожидаемое количество деревьев.
func TestNewPredictorInitialization(t *testing.T) {
	cfg := PredictorConfig{
		NumTrees:            5,
		NumFeatures:         2,
		MaxDepth:            10,
		MaxNodesPerTree:     100,
		HoeffdingSplitDelta: 1e-3,
		MinSamplesPerLeaf:   2,
		UseDriftDetection:   true,
		DriftAlpha:          0.05,
	}

	pred := NewPredictor(cfg)

	if pred == nil {
		t.Fatalf("expected non-nil predictor")
	}
	if got := len(pred.trees); got != cfg.NumTrees {
		t.Fatalf("expected %d trees, got %d", cfg.NumTrees, got)
	}
	for i, tr := range pred.trees {
		if tr == nil {
			t.Fatalf("tree %d is nil", i)
		}
		if tr.Root != nil {
			t.Fatalf("tree %d has non-nil root", i)
		}
	}
}

// TestPredictInitialRange проверяет, что начальное Predict возвращает
// вероятность в диапазоне [0,1] и стабильно вызывается несколько раз.
func TestPredictInitialRange(t *testing.T) {
	cfg := PredictorConfig{
		NumTrees:          3,
		NumFeatures:       3,
		MaxDepth:          5,
		MaxNodesPerTree:   50,
		MinSamplesPerLeaf: 1,
	}
	pred := NewPredictor(cfg)

	fv := features.FeatureVector{0.1, 0.5, 1.0}

	const iters = 5
	var prev float64
	for i := 0; i < iters; i++ {
		p := pred.Predict(fv)
		if p < 0.0 || p > 1.0 {
			t.Fatalf("predict out of range [0,1]: %v", p)
		}
		if i > 0 && p != prev {
			// Не требуем точной стабильности, но убеждаемся, что вызов не паникует
			// и возвращает числовое значение.
		}
		prev = p
	}
}

// TestUpdateChangesPrediction проверяет, что после нескольких Update
// предсказания могут измениться (онлайн-обучение влияет на модель).
func TestUpdateChangesPrediction(t *testing.T) {
	cfg := PredictorConfig{
		NumTrees:          1,
		NumFeatures:       1,
		MaxDepth:          3,
		MaxNodesPerTree:   10,
		MinSamplesPerLeaf: 1,
	}
	pred := NewPredictor(cfg)

	fv := features.FeatureVector{0.3}

	before := pred.Predict(fv)
	if before < 0.0 || before > 1.0 {
		t.Fatalf("initial predict out of range [0,1]: %v", before)
	}

	// Несколько раз обучаем на положительном классе.
	for i := 0; i < 20; i++ {
		pred.Update(fv, true)
	}

	after := pred.Predict(fv)
	if after < 0.0 || after > 1.0 {
		t.Fatalf("updated predict out of range [0,1]: %v", after)
	}
	if after <= before {
		t.Fatalf("expected prediction after updates to increase, before=%v after=%v", before, after)
	}
}

// TestUpdateSequentialSamples проверяет, что онлайн-обучение работает
// для последовательности примеров и не приводит к панике.
func TestUpdateSequentialSamples(t *testing.T) {
	cfg := PredictorConfig{
		NumTrees:          2,
		NumFeatures:       1,
		MaxDepth:          4,
		MaxNodesPerTree:   20,
		MinSamplesPerLeaf: 1,
	}
	pred := NewPredictor(cfg)

	samples := []struct {
		fv    features.FeatureVector
		label bool
	}{
		{features.FeatureVector{0}, false},
		{features.FeatureVector{1}, true},
		{features.FeatureVector{0.5}, true},
		{features.FeatureVector{0.2}, false},
	}

	for _, s := range samples {
		// Проверяем, что Update не паникует и после него можно вызвать Predict.
		pred.Update(s.fv, s.label)
		p := pred.Predict(s.fv)
		if p < 0.0 || p > 1.0 {
			t.Fatalf("predict out of range [0,1] after update: %v", p)
		}
	}
}
