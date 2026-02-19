package features

import (
	"testing"
)

// TestEmbedFeaturesNumericAndCategorical проверяет, что EmbedFeatures
// корректно кодирует числовые и категориальные признаки и выдаёт ожидаемую длину.
func TestEmbedFeaturesNumericAndCategorical(t *testing.T) {
	numeric := map[string]float64{
		"cpu_util": 0.75,
		"mem_util": 0.60,
	}
	categorical := map[string]string{
		"reaction_type": "HPA",
		"env":           "prod",
	}

	fv := EmbedFeatures(numeric, categorical)

	expectedLen := len(numeric) + len(categorical)
	if got := len(fv.Values); got != expectedLen {
		t.Fatalf("expected %d features after embedding, got %d", expectedLen, got)
	}
}

// TestEmbedFeaturesDeterministic проверяет, что EmbedFeatures детерминирован:
// повторный вызов с теми же входами даёт тот же вектор.
func TestEmbedFeaturesDeterministic(t *testing.T) {
	numeric := map[string]float64{
		"a": 1.0,
		"b": 2.0,
	}
	categorical := map[string]string{
		"cat1": "foo",
		"cat2": "bar",
	}

	fv1 := EmbedFeatures(numeric, categorical)
	fv2 := EmbedFeatures(numeric, categorical)

	if len(fv1.Values) != len(fv2.Values) {
		t.Fatalf("embedded vectors have different lengths: %d vs %d", len(fv1.Values), len(fv2.Values))
	}
	for i := range fv1.Values {
		if fv1.Values[i] != fv2.Values[i] {
			t.Fatalf("embedded vectors differ at index %d: %v vs %v", i, fv1.Values[i], fv2.Values[i])
		}
	}
}

// TestEmbedFeaturesEmpty проверяет корректную работу EmbedFeatures
// при пустых мапах числовых и категориальных признаков.
func TestEmbedFeaturesEmpty(t *testing.T) {
	fv := EmbedFeatures(map[string]float64{}, map[string]string{})
	if len(fv.Values) != 0 {
		t.Fatalf("expected 0 features for empty inputs, got %d", len(fv.Values))
	}
}

// TestEmbedFeaturesUnknownCategorical проверяет обработку "неизвестных"
// категориальных значений: функция должна стабильно хэшировать любые строки.
func TestEmbedFeaturesUnknownCategorical(t *testing.T) {
	numeric := map[string]float64{"x": 1.0}
	categorical := map[string]string{"unknown": "some-new-category"}

	fv := EmbedFeatures(numeric, categorical)

	if len(fv.Values) != 2 {
		t.Fatalf("expected 2 features (1 numeric + 1 categorical), got %d", len(fv.Values))
	}
	// Проверяем лишь то, что категориальный признак закодирован в число.
	if fv.Values[1] < 0.0 || fv.Values[1] >= 1.0 {
		t.Fatalf("expected hashed categorical feature in [0,1), got %v", fv.Values[1])
	}
}

