def demo_risk_score(tirads_score: int, features: dict) -> dict:
    """
    Non-clinical heuristic for MVP demonstration only.
    This is not a trained benign/malignant classifier.
    """
    score = 8.0 + tirads_score * 8.0
    if features.get("taller_than_wide"):
        score += 10.0
    if features.get("circularity", 1.0) < 0.45:
        score += 6.0
    if features.get("solidity", 1.0) < 0.75:
        score += 5.0
    if features.get("edge_density", 0.0) > 0.18:
        score += 4.0
    malignant = max(1.0, min(score, 95.0))
    return {
        "malignant_probability_demo": malignant,
        "benign_probability_demo": 100.0 - malignant,
        "model_type": "MVP rule-based demo score",
    }
