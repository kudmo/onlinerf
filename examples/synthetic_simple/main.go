package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/kudmo/onlinerf/onlinerf"
	"github.com/kudmo/onlinerf/onlinerf/features"
)

// Синтетический пример:
// label = (cpu + mem > 1.2) AND (env == "prod")
func main() {
	rand.Seed(42)

	// Фиксированная схема признаков:
	// 0: cpu
	// 1: mem
	// 2: env_dev
	// 3: env_staging
	// 4: env_prod
	// 5: role_api
	// 6: role_batch
	const numFeatures = 7

	cfg := onlinerf.PredictorConfig{
		NumTrees:            10,
		NumFeatures:         numFeatures,
		MaxDepth:            20,
		MaxNodesPerTree:     300,
		HoeffdingSplitDelta: 0.1,
		MinSamplesPerLeaf:   5,
		UseDriftDetection:   false,
	}

	model := onlinerf.NewPredictor(cfg)

	type sample struct {
		cpu   float64
		mem   float64
		env   string
		role  string
		label bool
	}

	envs := []string{"dev", "staging", "prod"}
	roles := []string{"api", "batch"}

	var all []sample

	for i := 0; i < 1000; i++ {
		cpu := rand.Float64() // [0,1)
		mem := rand.Float64()
		env := envs[rand.Intn(len(envs))]
		role := roles[rand.Intn(len(roles))]

		label := cpu+mem > 0.7 && env == "prod" || role == "api"

		all = append(all, sample{
			cpu:   cpu,
			mem:   mem,
			env:   env,
			role:  role,
			label: label,
		})
	}

	// Shuffle + split
	rand.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })

	trainSize := int(0.7 * float64(len(all)))
	train := all[:trainSize]
	test := all[trainSize:]

	// ---- Helper: fixed embedding ----
	embed := func(s sample) features.FeatureVector {

		vec := make([]float64, numFeatures)

		vec[0] = s.cpu
		vec[1] = s.mem

		// env one-hot
		switch s.env {
		case "dev":
			vec[2] = 1
		case "staging":
			vec[3] = 1
		case "prod":
			vec[4] = 1
		}

		// role one-hot
		switch s.role {
		case "api":
			vec[5] = 1
		case "batch":
			vec[6] = 1
		}

		return vec
	}

	// ---- Training ----
	for _, s := range train {
		model.Update(embed(s), s.label)
	}

	// ---- Evaluation ----
	var tp, fp, tn, fn int
	var sumPosProb, sumNegProb float64
	var nPos, nNeg int

	for _, s := range test {
		p := model.Predict(embed(s))
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
