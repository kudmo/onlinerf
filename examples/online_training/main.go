package main

import (
	"fmt"

	"github.com/kudmo/onlinerf/onlinerf"
	"github.com/kudmo/onlinerf/onlinerf/features"
)

// Пример потокового (online) обучения: последовательные вызовы Update
// на стриме событий и мониторинг изменения предсказаний.
func main() {
	cfg := onlinerf.PredictorConfig{
		NumTrees:            5,
		MaxDepth:            10,
		HoeffdingSplitDelta: 0.01,
		MinSamplesPerLeaf:   1,
		UseDriftDetection:   false,
	}

	pred := onlinerf.NewPredictor(cfg)

	type event struct {
		numeric      map[string]float64
		categorical  map[string]string
		label        bool
		description  string
	}

	stream := []event{
		{
			numeric: map[string]float64{
				"cpu_util": 0.3,
				"mem_util": 0.4,
			},
			categorical: map[string]string{"reaction_type": "NONE"},
			label:       false,
			description: "нормальное состояние",
		},
		{
			numeric: map[string]float64{
				"cpu_util": 0.8,
				"mem_util": 0.9,
			},
			categorical: map[string]string{"reaction_type": "HPA"},
			label:       true,
			description: "перегрузка и масштабирование",
		},
		{
			numeric: map[string]float64{
				"cpu_util": 0.7,
				"mem_util": 0.85,
			},
			categorical: map[string]string{"reaction_type": "HPA"},
			label:       true,
			description: "повторная перегрузка",
		},
	}

	for i, ev := range stream {
		fv := features.EmbedFeatures(ev.numeric, ev.categorical)

		before := pred.Predict(fv)
		pred.Update(fv, ev.label)
		after := pred.Predict(fv)

		fmt.Printf("event %d (%s): before=%.3f after=%.3f label=%v\n",
			i, ev.description, before, after, ev.label)
	}
}

