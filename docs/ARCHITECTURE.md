# Architecture: how the pieces connect

The core rule is **run inference once, analyze many times.** Models are loaded in one place, which writes prediction files. Everything downstream reads those files, so it needs no GPU or weights.

```mermaid
flowchart LR
  subgraph PRODUCE["Inference: run once"]
    PM["inference/predict_models.py"]
    RE["training/run_experiments.py"]
    EXP["train_yolo.export_predictions_for_split"]
    LPC["load_predictions_with_classes_from_model"]
    SPY["save_predictions_yolo_format"]
  end

  CACHE[("model_id/predictions/*.txt")]

  subgraph READ["train_yolo.py readers"]
    LPY["load_predictions_yolo_format<br/>masks, confs"]
    LPP["load_prediction_polygons<br/>+ polygon_to_mask"]
  end

  subgraph CONSUME["Analysis: run many times, no GPU"]
    AN["layer_count_stats · rank_predictions<br/>confidence_threshold_sweep<br/>multilayer_role_analysis"]
    BSP["inference/batch_size_profile.py<br/>--predictions-dir"]
    HA["annotation/hitl_annotator.py<br/>--predictions-dir"]
  end

  RAM["inference.review_and_measure"]
  IRO["interactive_review_objects<br/>+ save_review_decisions"]
  MORPH["analysis/morphology.py<br/>analyze_instances · save/draw/plot"]
  AGG["analysis/aggregate_size_results.py"] --> CSP["visualization/compare_size_profiles"]
  HL["annotation/hitl_launcher.py"] --> HA
  LPM["load_predictions_from_model<br/>(fallback: no .txt for an image)"]

  CG["analysis/cross_grid_comparison.py<br/>(one-off)"] --> EIY["inference.extract_instances_yolo"]
  MCP["analysis/multilayer_containment_predictions.py<br/>(candidate R3)"] --> LPC

  PM --> EXP
  RE --> EXP
  EXP --> LPC
  EXP --> SPY --> CACHE
  PM --> LPP
  CACHE --> LPY
  CACHE --> LPP
  LPY --> AN
  LPP --> BSP
  LPP --> HA
  HA -. fallback .-> LPM
  BSP --> RAM
  RAM --> IRO
  RAM --> MORPH
  BSP -. CSVs .-> AGG

  classDef yolo fill:#f6c667,stroke:#b7791f,color:#1f1a10
  classDef cache fill:#9fd8cf,stroke:#2a7f73,color:#0f2b27,stroke-width:2px
  classDef r3 fill:#fff8e8,stroke:#b7791f,stroke-width:2px,stroke-dasharray:6 3
  class LPC,LPM,EIY yolo
  class CACHE cache
  class MCP r3
```

**Key:** amber = loads a model and runs inference · teal = prediction files · dashed outline = still runs live inference and is a cleanup candidate (see [`CLEANUP_CANDIDATES.md`](../CLEANUP_CANDIDATES.md)).

## Separate workflow
`embedding/` (DINO embeddings + clustering) is independent of this graph. It shares only `data_utils.prepare_all_layer_singleclass` (category map), the geometry in `analysis/multilayer_containment.py` (`classify_roles()`, `find_containment()` and its own `polygon_to_mask()`), and `training.monitor_run.get_gpu_info`. See `embedding/README.md`.

## History
The pre-cleanup layout (U-Net stack, live inference in profiling and annotation, package `__init__` side effects) is preserved at git tag `pre-cleanup`.
