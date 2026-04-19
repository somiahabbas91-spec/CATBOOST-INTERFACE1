import pandas as pd
import joblib


class CatBoostUnifiedInterface:

    def __init__(self, clf_model_path, reg_model_path,
                 feature_names, label_names):

        self.clf_model = joblib.load(clf_model_path)
        self.reg_model = joblib.load(reg_model_path)

        self.feature_names = feature_names
        self.label_names = label_names

    def _merge_input(self, input_data):
        flat = {}
        for _, v in input_data.items():
            flat.update(v)
        return flat

    def _validate_input(self, flat):
        missing = [f for f in self.feature_names if f not in flat]
        if missing:
            raise ValueError(f"Missing features: {missing}")

    def predict_with_confidence(self, input_data):

        flat = self._merge_input(input_data)
        self._validate_input(flat)

        X = pd.DataFrame([flat])[self.feature_names]

        class_index = self.clf_model.predict(X).squeeze().astype(int).item()
        probs = self.clf_model.predict_proba(X).squeeze()

        failure_mode = self.label_names[class_index]
        confidence = float(max(probs))

        ultimate_load = float(self.reg_model.predict(X)[0])

        return {
            "failure_mode": failure_mode,
            "confidence": round(confidence, 3),
            "ultimate_load": round(ultimate_load, 2)
        }
