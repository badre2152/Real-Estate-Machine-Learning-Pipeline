"""
pipeline.py  (v2: intégration complète)
-----------------------------------------
Orchestrateur principal du pipeline ML immobilier Avito.

Nouveautés v2 :
  ✅ Config centralisée (config.yaml)
  ✅ Logging professionnel (logger_setup)
  ✅ Data Validation avant traitement
  ✅ Baseline Models (régression + classification)
  ✅ SMOTE via SmoteHandler
  ✅ MLflow Tracking
  ✅ SHAP Explainer
  ✅ Monitoring (chronomètre par étape, alertes)
  ✅ Prediction Intervals (CI 95%)
  ✅ Rapport HTML automatique
  ✅ Sauvegarde complète API-compatible
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config_loader import cfg
from logger_setup import get_logger, configure_root_logger

configure_root_logger()
logger = get_logger(__name__)

from classification import (
    evaluate_classification,
    get_feature_importance as clf_importance,
    save_model as _save_clf,
    train_classification,
)
from evaluate import run_full_evaluation
from extract import extract_obt
from features import engineer_features_train, engineer_features_test, add_classification_target
from prepare import clean_dataframe, split_data, prepare_data
from regression import (
    evaluate_regression,
    get_feature_importance,
    optimize_model,
    PriceScaleRegressor,
    save_model as _save_reg,
    train_regression,
)

from baselines import (
    run_regression_baselines,
    run_classification_baselines,
    compare_vs_regression_baseline,
    compare_vs_classification_baseline,
)
from data_validation import DataValidator
from mlflow_tracking import MLflowTracker
from mlflow_registry import MLflowRegistry, auto_register_and_promote
from monitoring import PipelineMonitor
from drift_detector import DriftDetector
from feature_store import FeatureStore, pipeline_write_features
from prediction_intervals import PredictionIntervalBuilder
from report_generator import ReportGenerator
from shap_explainer import SHAPExplainer
from smote_handler import SmoteHandler

def _save_artifact(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"   💾 Sauvegardé → {path}")

def run_pipeline(
    optimize: bool        = None,
    use_log_target: bool  = None,
    use_smote: bool       = None,
    use_calibration: bool = None,
    generate_plots: bool  = None,
    table: str            = None,
    input_parquet: str    = None,
    train_features: str   = None,
    test_features: str    = None,
    calibration_features: str = None,
    test_size: float      = None,
    random_state: int     = None,
) -> dict:
    """
    Lance le pipeline ML complet v2.
    Les paramètres None sont lus depuis config/config.yaml.
    """
    if calibration_features and not train_features:
        raise ValueError("Calibration features require prepared train and test features")
    if train_features and cfg.prediction_intervals.method == "quantile" and not calibration_features:
        raise ValueError("Quantile DVC training requires independent calibration features")

    if bool(train_features) != bool(test_features):
        raise ValueError('Specify both train_features and test_features, or neither')

    optimize        = optimize        if optimize        is not None else cfg.pipeline.optimize
    use_log_target  = use_log_target  if use_log_target  is not None else cfg.pipeline.use_log_target
    use_smote       = use_smote       if use_smote       is not None else cfg.pipeline.use_smote
    use_calibration = use_calibration if use_calibration is not None else cfg.pipeline.use_calibration
    generate_plots  = generate_plots  if generate_plots  is not None else cfg.pipeline.generate_plots
    table           = table           or cfg.database.table
    test_size       = test_size       if test_size       is not None else cfg.pipeline.test_size
    random_state    = random_state    if random_state    is not None else cfg.pipeline.random_state

    models_dir  = cfg.paths.models_dir
    plots_dir   = cfg.paths.plots_dir
    reports_dir = cfg.paths.reports_dir

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("🚀 " + "=" * 47)
    logger.info("   AVITO REAL ESTATE: ML PIPELINE v2")
    logger.info(f"   Options : optimize={optimize} | log_target={use_log_target} | smote={use_smote}")
    logger.info("=" * 50)

    monitor = PipelineMonitor(output_dir=reports_dir)
    tracker = MLflowTracker(run_name="pipeline_v2")
    tracker.start()

    tracker.log_params({
        "pipeline.optimize"        : optimize,
        "pipeline.use_log_target"  : use_log_target,
        "pipeline.use_smote"       : use_smote,
        "pipeline.use_calibration" : use_calibration,
        "pipeline.test_size"       : test_size,
        "pipeline.random_state"    : random_state,
        "pipeline.table"           : table,
    })

    shap_plots          = {}
    pi_builder          = None
    pi_examples         = []
    smote_report        = {}
    validation_report   = None
    baseline_reg        = {}
    baseline_clf        = {}
    reg_metrics         = {}
    clf_metrics         = {}
    report_path         = ""
    monitoring_path     = ""
    pi_cover            = {}
    reg_registry_result = {"version": None, "promoted": False, "reason": "not_run"}
    clf_registry_result = {"version": None, "promoted": False, "reason": "not_run"}
    drift_report        = None
    fs_stats            = {}

    try:

        with monitor.step("1_extraction"):
            logger.info("\n" + "=" * 50)
            logger.info("📥 ÉTAPE 1: Extraction OBT")
            logger.info("=" * 50)
            df = pd.read_parquet(input_parquet) if input_parquet else extract_obt(table=table)

        if df.empty:
            raise ValueError("Input dataset is empty")

        tracker.log_params({"data.raw_rows": len(df), "data.raw_cols": len(df.columns)})

        with monitor.step("2_validation"):
            logger.info("\n" + "=" * 50)
            logger.info("🛡️  ÉTAPE 2: Validation des données")
            logger.info("=" * 50)
            validator         = DataValidator()
            validation_report = validator.validate(df, stage="input")
            val_path          = validator.save_report(validation_report, output_dir=reports_dir)
            tracker.log_artifact(val_path, "validation")
            tracker.log_metrics({
                "validation.passed"    : int(validation_report.passed),
                "validation.n_failed"  : validation_report.n_failed,
                "validation.n_warnings": validation_report.n_warnings,
            })

        if not validation_report.passed:
            logger.warning("⚠️  Validation échouée: vérifier les données avant de continuer.")

        with monitor.step("3_cleaning"):
            logger.info("\n" + "=" * 50)
            logger.info("🧹 ÉTAPE 3: Nettoyage")
            logger.info("=" * 50)
            if not train_features:
                df = clean_dataframe(df)

        with monitor.step("4_split"):
            logger.info("\n" + "=" * 50)
            logger.info("✂️  ÉTAPE 4: Split train/test")
            logger.info("=" * 50)
            if train_features:
                df_train = pd.read_parquet(train_features)
                df_test = pd.read_parquet(test_features)
                if calibration_features and cfg.prediction_intervals.method == "quantile":
                    calibration_df = pd.read_parquet(calibration_features)
                    if calibration_df.empty:
                        raise ValueError("Quantile calibration feature data must not be empty")
                if df_train.empty or df_test.empty:
                    raise ValueError('Prepared feature datasets must not be empty')
            else:
                df_train, df_test = split_data(df, test_size=test_size, random_state=random_state)

        if not train_features:
            calibration_df = None
        if cfg.prediction_intervals.method == "quantile" and not train_features:
            if len(df_train) < 12:
                raise ValueError("Not enough rows for calibration")
            from sklearn.model_selection import train_test_split
            df_train, calibration_df = train_test_split(
                df_train, test_size=0.2, random_state=random_state
            )
            df_train = df_train.reset_index(drop=True)
            calibration_df = calibration_df.reset_index(drop=True)

        with monitor.step("5_feature_engineering"):
            logger.info("\n" + "=" * 50)
            logger.info("⚙️  ÉTAPE 5: Feature Engineering")
            logger.info("=" * 50)
            if not train_features:
                df_train, geo_stats = engineer_features_train(df_train)
                df_test = engineer_features_test(df_test, geo_stats)
                if calibration_df is not None:
                    calibration_df = engineer_features_test(calibration_df, geo_stats)
                if "categorie_prix" in df_train.columns and "categorie_prix" not in df_test.columns:
                    df_test = add_classification_target(df_test)
            else:
                logger.info('Using prepared DVC features without recomputing feature engineering')

        with monitor.step("6_encoding_scaling"):
            logger.info("\n" + "=" * 50)
            logger.info("🔧 ÉTAPE 6: Encoding + Scaling")
            logger.info("=" * 50)
            prepared = prepare_data(
                df_train, df_test,
                use_smote=False,
                random_state=random_state,
                save_preprocessor=True,
                calibration_df=calibration_df,
            )
            (X_train, X_test, y_reg_train, y_reg_test,
             y_clf_train, y_clf_test, feature_names, X_train_clf_raw) = prepared[:8]
            X_cal, y_cal = prepared[8:] if calibration_df is not None else (None, None)

        import os as _os
        _preproc_src = f"{models_dir}/preprocessor.pkl"
        if not _os.path.exists(_preproc_src):
            logger.warning("   ⚠️  preprocessor.pkl absent: il sera créé par prepare_data()")
        _save_artifact(feature_names, f"{models_dir}/feature_names.pkl")
        tracker.log_params({
            "data.n_train"   : len(X_train),
            "data.n_test"    : len(X_test),
            "data.n_features": len(feature_names),
        })

        classification_counts = (
            y_clf_train.astype("string").str.strip().str.lower().value_counts()
            if y_clf_train is not None else pd.Series(dtype="int64")
        )
        classification_eval_mask = None
        classification_enabled = len(classification_counts) >= 2 and classification_counts.min() >= 2
        if classification_enabled:
            normalized_test_labels = y_clf_test.astype("string").str.strip().str.lower()
            classification_eval_mask = normalized_test_labels.isin(classification_counts.index).fillna(False).to_numpy(dtype=bool)
            if not classification_eval_mask.any():
                logger.warning("Classification skipped: no known property types in evaluation data")
                classification_enabled = False
        if not classification_enabled:
            y_clf_train = None
            y_clf_test = None

        with monitor.step("7_baselines"):
            logger.info("\n" + "=" * 50)
            logger.info("📏 ÉTAPE 7: Baseline Models")
            logger.info("=" * 50)
            baseline_reg = run_regression_baselines(X_train, y_reg_train, X_test, y_reg_test)

            if y_clf_train is not None:
                baseline_clf = run_classification_baselines(
                    X_train_clf_raw,
                    y_clf_train,
                    X_test.iloc[classification_eval_mask],
                    y_clf_test.iloc[classification_eval_mask],
                )

            tracker.log_baseline_results(baseline_reg, baseline_clf or None)

        X_train_clf = X_train_clf_raw.copy()
        if y_clf_train is not None:
            with monitor.step("8_smote"):
                logger.info("\n" + "=" * 50)
                logger.info("⚖️  ÉTAPE 8: SMOTE")
                logger.info("=" * 50)
                smote_handler = SmoteHandler(random_state=random_state)
                X_train_clf, y_clf_train = smote_handler.fit_resample(
                    X_train_clf_raw, y_clf_train,
                    force=use_smote,
                )
                smote_report = smote_handler.get_report()
                tracker.log_params({
                    "smote.applied"  : smote_report["smote_applied"],
                    "smote.strategy" : smote_report["sampling_strategy"],
                })

        with monitor.step("9a_regression_train"):
            logger.info("\n" + "=" * 50)
            logger.info("📈 ÉTAPE 9A: Entraînement Régression")
            logger.info("=" * 50)
            import numpy as np
            regression_X = X_train
            regression_y = y_reg_train
            reg_model, reg_name, _ = train_regression(
                regression_X, regression_y, use_log_target=use_log_target
            )
            if optimize:
                y_opt = np.log1p(regression_y) if use_log_target else regression_y
                reg_model = optimize_model(reg_model, regression_X, y_opt)

        with monitor.step("9a_regression_eval"):
            reg_metrics = evaluate_regression(reg_model, X_test, y_reg_test, use_log_target)
            compare_vs_regression_baseline(reg_metrics, baseline_reg)
            monitor.check_regression_alert(r2=reg_metrics["R2"], mae=reg_metrics["MAE"])
            get_feature_importance(reg_model, feature_names)

        tracker.log_regression_results(reg_metrics, reg_name)

        reg_model_raw = reg_model
        reg_model = PriceScaleRegressor(reg_model_raw, log_target=bool(use_log_target))
        reg_model.estimator_ = reg_model_raw

        _save_reg(reg_model, f"{models_dir}/regression_model.pkl")
        _save_artifact(reg_model,   f"{models_dir}/best_regression_model.pkl")
        _save_artifact(reg_metrics, f"{models_dir}/regression_metrics.pkl")
        tracker.log_model(reg_model, "regression_model")

        reg_registry_result = {"version": None, "promoted": False, "reason": "skipped"}
        if tracker.run_id:
            reg_registry_result = auto_register_and_promote(
                run_id          = tracker.run_id,
                model_name      = "avito-regression",
                artifact_path   = "regression_model",
                primary_metric  = "reg/R2",
                higher_is_better= True,
                description     = f"{reg_name} | R²={reg_metrics.get('R2', 0):.4f}",
            )
            logger.info(
                f"   📋 Registry régression → "
                f"v{reg_registry_result['version']} | "
                f"{'🚀 promu Production' if reg_registry_result['promoted'] else '🟡 Staging'}"
            )

        clf_model = clf_name = label_enc = None

        if classification_enabled:
            with monitor.step("9b_classification_train"):
                logger.info("\n" + "=" * 50)
                logger.info("🧠 ÉTAPE 9B: Entraînement Classification")
                logger.info("=" * 50)
                clf_model, clf_name, label_enc = train_classification(
                    X_train_clf, y_clf_train, use_calibration=use_calibration
                )

            with monitor.step("9b_classification_eval"):
                clf_metrics = evaluate_classification(clf_model, X_test, y_clf_test, label_enc)
                compare_vs_classification_baseline(clf_metrics, baseline_clf)
                monitor.check_classification_alert(
                    f1=clf_metrics["F1"],
                    accuracy=clf_metrics["Accuracy"],
                )
                clf_importance(clf_model, feature_names)

            tracker.log_classification_results(clf_metrics, clf_name)

            _save_clf(clf_model, label_enc, f"{models_dir}/classification_model.pkl")
            _save_artifact(clf_model,   f"{models_dir}/best_classification_model.pkl")
            _save_artifact(label_enc,   f"{models_dir}/label_encoder.pkl")
            _save_artifact(clf_metrics, f"{models_dir}/classification_metrics.pkl")
            tracker.log_model(clf_model, "classification_model")

            clf_registry_result = {"version": None, "promoted": False, "reason": "skipped"}
            if tracker.run_id:
                clf_registry_result = auto_register_and_promote(
                    run_id          = tracker.run_id,
                    model_name      = "avito-classification",
                    artifact_path   = "classification_model",
                    primary_metric  = "clf/F1",
                    higher_is_better= True,
                    description     = f"{clf_name} | F1={clf_metrics.get('F1', 0):.4f}",
                )
                logger.info(
                    f"   📋 Registry classification → "
                    f"v{clf_registry_result['version']} | "
                    f"{'🚀 promu Production' if clf_registry_result['promoted'] else '🟡 Staging'}"
                )
        else:
            logger.warning("Classification skipped: missing target, fewer than two types, or rare labels")

        with monitor.step("10_prediction_intervals"):
            logger.info("\n" + "=" * 50)
            logger.info("📐 ÉTAPE 10: Intervalles de Prédiction (95% CI)")
            logger.info("=" * 50)
            pi_builder = PredictionIntervalBuilder(
                method           = cfg.prediction_intervals.method,
                confidence_level = float(cfg.prediction_intervals.confidence_level),
                n_bootstrap      = int(cfg.prediction_intervals.n_bootstrap),
                random_state     = random_state,
            )
            pi_builder.fit(reg_model, regression_X, regression_y, X_cal=X_cal, y_cal=y_cal)
            pi_df    = pi_builder.predict_with_interval(X_test)
            pi_cover = pi_builder.evaluate_coverage(X_test, y_reg_test)
            tracker.log_metrics({
                "pi/coverage"  : pi_cover["picp"],
                "pi/mean_width": pi_cover["mpiw"],
            })
            pi_examples = pi_df.head(10).to_dict(orient="records")

            _save_artifact(pi_builder, f"{models_dir}/pi_builder.pkl")

        with monitor.step("11_shap"):
            logger.info("\n" + "=" * 50)
            logger.info("🔍 ÉTAPE 11: SHAP Interprétabilité")
            logger.info("=" * 50)
            shap_exp = SHAPExplainer(
                model        = reg_model_raw,
                X_background = X_train,
                feature_names= feature_names,
            )
            shap_plots = shap_exp.run(X_test, output_dir=plots_dir, prefix="reg_")

            shap_df = shap_exp.get_feature_importance()
            if shap_df is not None:
                for _, row in shap_df.head(5).iterrows():
                    tracker.log_metrics({f"shap/top_{row['feature']}": row["shap_mean"]})
            for path in shap_plots.values():
                if path and str(path).endswith(".png"):
                    tracker.log_artifact(path, "shap_plots")

        if generate_plots:
            with monitor.step("12_plots"):
                logger.info("\n" + "=" * 50)
                logger.info("📊 ÉTAPE 12: Visualisations")
                logger.info("=" * 50)
                run_full_evaluation(
                    reg_model, clf_model, X_test,
                    y_reg_test, y_clf_test, label_enc, feature_names,
                )
                tracker.log_artifacts_dir(plots_dir, "plots")

        fs_stats = {}
        with monitor.step("12a_feature_store"):
            logger.info("\n" + "=" * 50)
            logger.info("🏪 ÉTAPE 12A: Feature Store (écriture features)")
            logger.info("=" * 50)
            try:
                X_train_df = (
                    pd.DataFrame(X_train, columns=feature_names)
                    if not isinstance(X_train, pd.DataFrame)
                    else X_train.copy()
                )

                store_path = os.path.join(models_dir, "..", "feature_store", "store.db")
                fs_stats = pipeline_write_features(
                    df_train     = X_train_df,
                    feature_names= feature_names,
                    store_path   = store_path,
                    version      = "v1",
                )

                tracker.log_metrics({
                    "feature_store/n_groups"  : len(fs_stats),
                    "feature_store/total_rows": sum(s.n_rows for s in fs_stats.values()),
                })
                tracker.set_tag("feature_store.version", "v1")
                logger.info(f"   ✅ {len(fs_stats)} groupes écrits dans le Feature Store")

            except Exception as fs_exc:
                logger.warning(f"   ⚠️  Feature Store ignoré : {fs_exc}")

        drift_report = None
        drift_report_path = None
        with monitor.step("12b_drift_detection"):
            logger.info("\n" + "=" * 50)
            logger.info("🔍 ÉTAPE 12B: Drift Detection (Train vs Test)")
            logger.info("=" * 50)
            try:
                X_train_df = pd.DataFrame(X_train, columns=feature_names) if not isinstance(X_train, pd.DataFrame) else X_train
                X_test_df  = pd.DataFrame(X_test,  columns=feature_names) if not isinstance(X_test,  pd.DataFrame) else X_test

                detector = DriftDetector(
                    reference_data=X_train_df,
                    output_dir=reports_dir,
                )

                reg_preds_train = reg_model.predict(X_train)
                reg_preds_test  = reg_model.predict(X_test)

                drift_report = detector.detect(
                    current_data    = X_test_df,
                    predictions_ref = reg_preds_train,
                    predictions_cur = reg_preds_test,
                    save_report     = True,
                )

                drift_report_path = drift_report.save(reports_dir)

                tracker.log_metrics({
                    "drift/dataset_psi"    : drift_report.dataset_psi,
                    "drift/n_drifted"      : drift_report.n_drifted,
                    "drift/n_warnings"     : drift_report.n_warnings,
                    "drift/pred_psi"       : drift_report.prediction_drift.psi if drift_report.prediction_drift else 0.0,
                    "drift/retrain_needed" : int(drift_report.needs_retraining),
                })
                tracker.set_tag("drift.recommendation", drift_report.recommendation)

                if drift_report_path:
                    tracker.log_artifact(drift_report_path, "drift")

                logger.info(f"   {drift_report.summary()}")

            except Exception as drift_exc:
                logger.warning(f"   ⚠️  Drift Detection ignorée : {drift_exc}")

        with monitor.step("13_report"):
            logger.info("\n" + "=" * 50)
            logger.info("📄 ÉTAPE 13: Rapport HTML")
            logger.info("=" * 50)

            tracker.log_metrics(monitor.get_step_durations())
            monitoring_path = monitor.save_report()

            gen = ReportGenerator()
            report_path = gen.build(
                reg_metrics       = reg_metrics,
                clf_metrics       = clf_metrics,
                baseline_reg      = baseline_reg,
                baseline_clf      = baseline_clf or None,
                validation_report = validation_report,
                shap_plots        = shap_plots,
                monitoring_path   = monitoring_path,
                pi_examples       = pi_examples,
                smote_report      = smote_report,
                output_dir        = reports_dir,
            )
            tracker.log_artifact(report_path,    "reports")
            tracker.log_artifact(monitoring_path, "reports")

        monitor.print_summary()
        tracker.end(success=True)

        logger.info("\n" + "=" * 50)
        logger.info("✅ PIPELINE v2 TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 50)
        logger.info(
            f"   📈 Régression     → R²={reg_metrics.get('R2', 0):.4f} | "
            f"MAE={reg_metrics.get('MAE', 0):,.0f} MAD | "
            f"MAPE={reg_metrics.get('MAPE', 0):.1f}%"
        )
        if clf_metrics:
            logger.info(
                f"   🧠 Classification → F1={clf_metrics.get('F1', 0):.4f} | "
                f"Accuracy={clf_metrics.get('Accuracy', 0):.4f}"
            )
        logger.info(f"   📐 PI Coverage    → {pi_cover.get('picp', 0):.1%} (cible 95%)")
        logger.info(f"   ⏱  Durée totale  → {monitor.total_duration_s:.0f}s")
        logger.info(f"   📄 Rapport       → {report_path}")
        logger.info(f"   🔬 MLflow UI     → mlflow ui --backend-store-uri {cfg.paths.mlflow_uri}")
        logger.info("=" * 50)

        results = {
            "pipeline_version": "2.0",
            "total_duration_s": round(monitor.total_duration_s, 2),
            "step_durations_s": monitor.get_step_durations(),
            "options": {
                "optimize"       : optimize,
                "use_log_target" : use_log_target,
                "use_smote"      : use_smote,
                "use_calibration": use_calibration,
                "test_size"      : test_size,
                "random_state"   : random_state,
            },
            "data": {
                "n_train"   : len(X_train),
                "n_test"    : len(X_test),
                "n_features": len(feature_names),
            },
            "regression"          : {**reg_metrics, "model": reg_name},
            "classification"      : ({**clf_metrics, "model": clf_name} if clf_metrics else None),
            "baselines_reg"       : baseline_reg,
            "prediction_intervals": pi_cover,
            "validation"          : {
                "passed"   : validation_report.passed if validation_report else None,
                "n_failed" : validation_report.n_failed if validation_report else 0,
            },
            "smote"     : smote_report,
            "reports"   : {"html": report_path, "monitoring": monitoring_path},
            "mlflow_run_id": tracker.run_id,
            "registry": {
                "regression"    : reg_registry_result,
                "classification": clf_registry_result if clf_metrics else {"version": None, "promoted": False},
            },
            "drift": drift_report.to_dict() if drift_report else None,
            "feature_store": {
                g: {"n_rows": s.n_rows, "n_features": s.n_features, "checksum": s.checksum}
                for g, s in fs_stats.items()
            },
        }

        results_path = f"{models_dir}/results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"   📋 Résultats JSON → {results_path}")

        return results

    except Exception as exc:
        logger.error(f"\n❌ PIPELINE ÉCHOUÉ : {exc}", exc_info=True)
        tracker.end(success=False)
        monitor.print_summary()
        raise

def _parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline ML v2: Prix Immobilier Avito Maroc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python pipeline.py                       # paramètres depuis config.yaml\n"
            "  python pipeline.py --optimize            # avec optimisation hyperparamètres\n"
            "  python pipeline.py --smote --log-target  # SMOTE + transformation log\n"
            "  python pipeline.py --no-plots            # sans visualisations\n"
        ),
    )
    p.add_argument("--optimize",   action="store_true", default=None)
    p.add_argument("--log-target", action="store_true", default=None)
    p.add_argument("--smote",      action="store_true", default=None)
    p.add_argument("--calibrate",  action="store_true", default=None)
    p.add_argument("--no-plots",   action="store_true")
    p.add_argument("--table",      default=None, help="Table OBT (défaut : config.yaml)")
    p.add_argument("--input-parquet", default=None, help="Read extracted OBT from Parquet instead of PostgreSQL")
    p.add_argument("--train-features", default=None, help="Prepared training features Parquet")
    p.add_argument("--test-features", default=None, help="Prepared test features Parquet")
    p.add_argument("--calibration-features", default=None, help="Independent prepared calibration features Parquet")
    p.add_argument("--test-size",  type=float, default=None)
    p.add_argument("--seed",       type=int,   default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        optimize        = args.optimize   or None,
        use_log_target  = args.log_target or None,
        use_smote       = args.smote      or None,
        use_calibration = args.calibrate  or None,
        generate_plots  = not args.no_plots,
        table           = args.table,
        input_parquet   = args.input_parquet,
        train_features  = args.train_features,
        test_features   = args.test_features,
        calibration_features = args.calibration_features,
        test_size       = args.test_size,
        random_state    = args.seed,
    )
