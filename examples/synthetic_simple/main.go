package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/kudmo/onlinerf/onlinerf"
	"github.com/kudmo/onlinerf/onlinerf/features"
)

// Простой пример с большим количеством синтетических данных.
// Есть две числовые метрики (cpu, mem) и две категориальные (env, role).
// Зависимость: объект "плохой" (label=true), если cpu+mem > 1.2 и env == "prod".
func main() {
	rand.Seed(42)

	cfg := onlinerf.PredictorConfig{
		NumTrees:            10,
		MaxDepth:            5,
		MaxNodesPerTree:     200,
		HoeffdingSplitDelta: 0.05,
		MinSamplesPerLeaf:   2,
		UseDriftDetection:   false,
	}
	model := onlinerf.NewPredictor(cfg)

	type sample struct {
		numeric     map[string]float64
		categorical map[string]string
		label       bool
	}

	// Генерируем датасет: 1000 объектов, 70% train / 30% test.
	var all []sample
	envs := []string{"dev", "staging", "prod"}
	roles := []string{"api", "batch"}

	for i := 0; i < 1000; i++ {
		cpu := rand.Float64() // [0,1)
		mem := rand.Float64()
		env := envs[rand.Intn(len(envs))]
		role := roles[rand.Intn(len(roles))]

		// Простое правило для генерации лейбла.
		label := cpu+mem > 1.2 && env == "prod"

		all = append(all, sample{
			numeric: map[string]float64{
				"cpu": cpu,
				"mem": mem,
			},
			categorical: map[string]string{
				"env":  env,
				"role": role,
			},
			label: label,
		})
	}

	// Перемешиваем и делим на train/test.
	rand.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })
	trainSize := int(0.7 * float64(len(all)))
	train := all[:trainSize]
	test := all[trainSize:]

	// Обучение на train (батч).
	for _, s := range train {
		fv := features.EmbedFeatures(s.numeric, s.categorical)
		model.Update(fv, s.label)
	}

	// Оценка качества на test: считаем accuracy и среднюю вероятность
	// для положительного и отрицательного класса.
	var tp, fp, tn, fn int
	var sumPosProb, sumNegProb float64
	var nPos, nNeg int

	for _, s := range test {
		fv := features.EmbedFeatures(s.numeric, s.categorical)
		p := model.Predict(fv)

		if s.label {
			sumPosProb += p
			nPos++
		} else {
			sumNegProb += p
			nNeg++
		}

		pred := p >= 0.5
		switch {
		case pred && s.label:
			tp++
		case pred && !s.label:
			fp++
		case !pred && !s.label:
			tn++
		case !pred && s.label:
			fn++
		}
	}

	total := float64(len(test))
	accuracy := float64(tp+tn) / total

	avgPosProb := 0.0
	if nPos > 0 {
		avgPosProb = sumPosProb / float64(nPos)
	}
	avgNegProb := 0.0
	if nNeg > 0 {
		avgNegProb = sumNegProb / float64(nNeg)
	}

	fmt.Printf("Synthetic simple example (%d train, %d test)\n", len(train), len(test))
	fmt.Printf("Time: %s\n", time.Now().Format(time.RFC3339))
	fmt.Printf("Accuracy: %.3f (tp=%d fp=%d tn=%d fn=%d)\n", accuracy, tp, fp, tn, fn)
	fmt.Printf("Avg prob for positive class (y=1): %.3f over %d samples\n", avgPosProb, nPos)
	fmt.Printf("Avg prob for negative class (y=0): %.3f over %d samples\n", avgNegProb, nNeg)
}

