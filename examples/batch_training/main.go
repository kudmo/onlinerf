package main

import (
	"fmt"
	"math"

	"github.com/kudmo/onlinerf/onlinerf"
	"github.com/kudmo/onlinerf/onlinerf/features"
)

// Пример обучения на батче данных и расчёта базовых метрик качества
// бинарного классификатора (accuracy, precision, recall, F1, log loss).
func main() {
	// 1. Конфигурация модели.
	cfg := onlinerf.PredictorConfig{
		NumTrees:            20,
		MaxDepth:            8,
		MaxNodesPerTree:     500,
		HoeffdingSplitDelta: 0.01,
		MinSamplesPerLeaf:   2,
		UseDriftDetection:   false,
	}

	model := onlinerf.NewPredictor(cfg)

	// 2. Подготовка синтетического датасета.
	type sample struct {
		numeric     map[string]float64
		categorical map[string]string
		label       bool
	}

	train := []sample{
		{numeric: map[string]float64{"cpu": 0.2, "mem": 0.3}, categorical: map[string]string{"env": "dev"}, label: false},
		{numeric: map[string]float64{"cpu": 0.8, "mem": 0.9}, categorical: map[string]string{"env": "prod"}, label: true},
		{numeric: map[string]float64{"cpu": 0.7, "mem": 0.85}, categorical: map[string]string{"env": "prod"}, label: true},
		{numeric: map[string]float64{"cpu": 0.1, "mem": 0.2}, categorical: map[string]string{"env": "dev"}, label: false},
		{numeric: map[string]float64{"cpu": 0.6, "mem": 0.7}, categorical: map[string]string{"env": "staging"}, label: true},
		{numeric: map[string]float64{"cpu": 0.3, "mem": 0.4}, categorical: map[string]string{"env": "staging"}, label: false},
	}

	test := []sample{
		{numeric: map[string]float64{"cpu": 0.25, "mem": 0.35}, categorical: map[string]string{"env": "dev"}, label: false},
		{numeric: map[string]float64{"cpu": 0.9, "mem": 0.95}, categorical: map[string]string{"env": "prod"}, label: true},
		{numeric: map[string]float64{"cpu": 0.5, "mem": 0.6}, categorical: map[string]string{"env": "staging"}, label: true},
		{numeric: map[string]float64{"cpu": 0.15, "mem": 0.25}, categorical: map[string]string{"env": "dev"}, label: false},
	}

	// 3. Обучение на батче: проходим по train и вызываем Update.
	for _, s := range train {
		fv := features.EmbedFeatures(s.numeric, s.categorical)
		model.Update(fv, s.label)
	}

	// 4. Оценка на тестовом наборе: собираем предсказания и считаем метрики.
	type metrics struct {
		tp, fp, tn, fn int
		logLoss        float64
	}
	var m metrics

	for _, s := range test {
		fv := features.EmbedFeatures(s.numeric, s.categorical)
		p := model.Predict(fv)

		// бинаризация по порогу 0.5
		predLabel := p >= 0.5

		switch {
		case predLabel && s.label:
			m.tp++
		case predLabel && !s.label:
			m.fp++
		case !predLabel && !s.label:
			m.tn++
		case !predLabel && s.label:
			m.fn++
		}

		// log loss: -[y*log(p) + (1-y)*log(1-p)]
		y := 0.0
		if s.label {
			y = 1.0
		}
		// защита от log(0)
		eps := 1e-15
		pp := math.Max(eps, math.Min(1.0-eps, p))
		m.logLoss += -(y*math.Log(pp) + (1-y)*math.Log(1-pp))
	}

	n := float64(len(test))
	acc := float64(m.tp+m.tn) / n

	precDen := m.tp + m.fp
	recDen := m.tp + m.fn

	var precision, recall, f1 float64
	if precDen > 0 {
		precision = float64(m.tp) / float64(precDen)
	}
	if recDen > 0 {
		recall = float64(m.tp) / float64(recDen)
	}
	if precision+recall > 0 {
		f1 = 2 * precision * recall / (precision + recall)
	}

	avgLogLoss := m.logLoss / n

	fmt.Printf("Test metrics on batch:\n")
	fmt.Printf("  Accuracy : %.3f\n", acc)
	fmt.Printf("  Precision: %.3f\n", precision)
	fmt.Printf("  Recall   : %.3f\n", recall)
	fmt.Printf("  F1-score : %.3f\n", f1)
	fmt.Printf("  Log loss : %.3f\n", avgLogLoss)
}

