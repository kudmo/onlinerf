package main

import (
    "fmt"

    "github.com/kudmo/onlinerf/onlinerf"
    "github.com/kudmo/onlinerf/onlinerf/features"
)

func main() {
    config := onlinerf.PredictorConfig{
        NumTrees:            10,
        MaxDepth:            20,
        HoeffdingSplitDelta: 0.05,
        MinSamplesPerLeaf:   5,
        UseDriftDetection:   true,
        DriftAlpha:          0.01,
    }

    pred := onlinerf.NewPredictor(config)

    numericFeatures := map[string]float64{
        "cpu_util":  0.75,
        "mem_util":  0.60,
        "rps":       1200,
        "latency":   250,
        "replicas":  3,
    }

    categoricalFeatures := map[string]string{
        "reaction_type": "HPA",
    }

    fv := features.EmbedFeatures(numericFeatures, categoricalFeatures)

    p := pred.Predict(fv)
    fmt.Printf("Predicted probability of class 1: %.3f\n", p)

    label := true
    pred.Update(fv, label)

}
