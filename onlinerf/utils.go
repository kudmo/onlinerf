package onlinerf

// boolToFloat converts a boolean label to numeric representation.
func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

