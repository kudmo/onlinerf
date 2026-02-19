package main

import (
	"fmt"

	"github.com/kudmo/onlinerf/onlinerf"
	"github.com/kudmo/onlinerf/onlinerf/features"
)

// Пример базового использования: инициализация леса, embedding признаков
// и вызовы Predict/Update.
func main() {
	// 1. Настройка модели.
	cfg := onlinerf.PredictorConfig{
		NumTrees:            10,
		MaxDepth:            20,
		HoeffdingSplitDelta: 0.05,
		MinSamplesPerLeaf:   5,
		UseDriftDetection:   true,
		DriftAlpha:          0.01,
	}

	pred := onlinerf.NewPredictor(cfg)

	// 2. Подготовка признаков.
	numericFeatures := map[string]float64{
		"cpu_util": 0.75,
		"mem_util": 0.60,
		"rps":      1200,
		"latency":  250,
		"replicas": 3,
	}

	categoricalFeatures := map[string]string{
		"reaction_type": "HPA",
	}

	fv := features.EmbedFeatures(numericFeatures, categoricalFeatures)

	// 3. Предсказание вероятности положительного класса.
	p := pred.Predict(fv)
	fmt.Printf("Predicted probability of class 1: %.3f\n", p)

	// 4. Онлайн-обучение по фактическому лейблу.
	label := true
	pred.Update(fv, label)
}

