import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from model_utils import run_regression_for_region

from constants import LABEL_MAP

class TestModelUtils(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset similar to your climate data structure
        dates = pd.date_range(start="2000-01-01", periods=60, freq="M")
        regions = ["Dodoma"] * 60
        np.random.seed(42)

        self.df = pd.DataFrame({
            "Region": regions,
            "YearMonth": dates,
            "T2M": np.random.normal(25, 2, 60),
            "PRECTOTCORR": np.random.gamma(2, 2, 60),
            "WS2M": np.random.uniform(1, 5, 60),
            "RH2M": np.random.uniform(40, 80, 60),
            "Month": dates.month,
            "Year": dates.year,
            "Season": ["Dry_Season" if m in [6,7,8] else "Rainy" for m in dates.month]
        })

        # One-hot encode categorical feature 'Season'
        self.df = pd.get_dummies(self.df, columns=["Season"], drop_first=True)

    def test_run_regression_for_region_random_forest(self):
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        trained_model, metrics = run_regression_for_region(
            self.df, region="Dodoma", target="T2M", model=model
        )
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertGreater(metrics["rmse"], 0)
        self.assertTrue(hasattr(trained_model.named_steps["model"], "feature_importances_"))

        # Log user-friendly top 5 features
        feature_importances = trained_model.named_steps["model"].feature_importances_
        feature_names = trained_model.named_steps["model"].feature_names_in_
        top_features = sorted(zip(feature_names, feature_importances), key=lambda x: -x[1])[:5]

        print("\n🧠 Top 5 Important Features (User-Friendly Labels):")
        for feat, score in top_features:
            label = LABEL_MAP.get(feat, feat)
            print(f"  {label}: {score:.3f}")

if __name__ == "__main__":
    unittest.main()
