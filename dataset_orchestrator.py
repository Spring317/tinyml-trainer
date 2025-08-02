import os
import traceback

from dataset_builder import (
    load_config,
    run_analyze_dataset,
    run_copy_matched_species,
    run_cross_reference,
    run_manifest_generator,
    run_visualization,
    run_web_crawl,
    validate_config,
    venn_diagram,
)
from dataset_builder.core import ConfigError, FailedOperation, PipelineError
from dataset_builder.core.constants import CLASS_LIST
from dataset_builder.core.utility import banner


def run_stage(stage_name: str, func):
    banner(stage_name)

    try:
        func()
        print("\n")
    except FailedOperation as e:
        print(f"FailedOperation during {stage_name}:\n{e}")
        raise
    except PipelineError as e:
        print(f"PipelineError during {stage_name}:\n{e}")
        raise
    except Exception as e:
        print(f"Unexpected error in {stage_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise FailedOperation(f"Unhandled exception in {stage_name}")


try:
    config = load_config("./config.yaml")
    # validate_config(config)
except ConfigError as e:
    print(e)
    exit()


# Global
verbose = config["global"]["verbose"]
target_classes = config["global"]["included_classes"]
overwrite = config["global"]["overwrite"]

# Paths
src_dataset_path = config["paths"]["src_dataset"]
dst_dataset_path = config["paths"]["dst_dataset"]
src_dataset_name = os.path.basename(src_dataset_path)
dst_dataset_name = os.path.basename(dst_dataset_path)
output_path = config["paths"]["output_dir"]
matched_species_file = f"matched_species_{src_dataset_name}_{dst_dataset_name}.json"
matched_species_path = os.path.join(output_path, matched_species_file)
src_dataset_json = os.path.join(output_path, f"{src_dataset_name}_species.json")
dst_dataset_json = os.path.join(output_path, f"{dst_dataset_name}_species.json")
dst_properties_path = os.path.join(output_path, f"{dst_dataset_name}_composition.json")
src_properties_path = os.path.join(output_path, f"{src_dataset_name}_composition.json")

# Web Crawl
base_url = config["web_crawl"]["base_url"]
total_pages = config["web_crawl"]["total_pages"]
delay = config["web_crawl"]["delay_between_requests"]
web_crawl_output_path = config["paths"]["web_crawl_output_json"]

# Train and validate split
train_size = config["train_val_split"]["train_size"]
randomness = config["train_val_split"]["random_state"]
dominant_threshold = config["train_val_split"]["dominant_threshold"]

os.makedirs(dst_dataset_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

try:
    # run_stage(
    #     "Crawling the web to obtain the dataset",
    #     lambda: run_web_crawl(
    #         base_url, web_crawl_output_path, delay, total_pages, overwrite, verbose
    #     ),
    # )
    # run_stage(
    #     "Getting datasets properties",
    #     lambda: run_analyze_dataset(
    #         web_crawl_output_path,
    #         output_path,
    #         dst_dataset_name,
    #         target_classes,
    #         verbose,
    #         overwrite,
    #     ),
    # )
    # run_stage(
    #     "Getting datasets properties",
    #     lambda: run_analyze_dataset(
    #         src_dataset_path,
    #         output_path,
    #         src_dataset_name,
    #         target_classes,
    #         verbose,
    #         overwrite,
    #     ),
    # )
    # run_stage(
    #     "Matching species between datasets",
    #     lambda: run_cross_reference(
    #         matched_species_path,
    #         src_dataset_json,
    #         web_crawl_output_path,
    #         src_dataset_name,
    #         dst_dataset_name,
    #         target_classes,
    #         verbose,
    #         overwrite,
    #     ),
    # )
    # run_stage(
    #     "Copying matching species",
    #     lambda: run_copy_matched_species(
    #         src_dataset_path,
    #         dst_dataset_path,
    #         matched_species_path,
    #         target_classes,
    #         verbose,
    #     ),
    # )
    run_stage(
        "Getting datasets properties",
        lambda: run_analyze_dataset(
            dst_dataset_path,
            output_path,
            dst_dataset_name,
            target_classes,
            verbose,
            True,
        ),
    )
    # run_stage(
    #     "Getting datasets properties",
    #     lambda: run_analyze_dataset(
    #         src_dataset_path,
    #         output_path,
    #         src_dataset_name,
    #         target_classes,
    #         verbose,
    #         True,
    #     ),
    # )
    run_stage(
        "Generating dataset manifests",
        lambda: run_manifest_generator(
            dst_dataset_path,
            dst_dataset_path,
            dst_properties_path,
            train_size,
            randomness,
            target_classes,
            dominant_threshold,
        ),
    )
    # run_stage(
    #     "Generating visualization",
    #     lambda: run_visualization(
    #         src_dataset_path,
    #         dst_dataset_path,
    #         output_path,
    #         CLASS_LIST,
    #         target_classes,
    #         verbose,
    #         overwrite,
    #     ),
    # )
    # run_stage(
    #     "Generating PPF for Aves and Insecta",
    #     lambda: visualizing_ppf_many(
    #         src_properties_path,
    #         CLASS_LIST,
    #     )
    # )
    
    # run_stage(
    #     "Generating Venn diagram",
    #     lambda: venn_diagram(
    #         src_dataset_json,
    #         web_crawl_output_path,
    #         src_dataset_name,
    #         dst_dataset_name,
    #         "Species Overlap Between Datasets",
    #         CLASS_LIST,
    #         os.path.join(output_path, f"{src_dataset_name}_vs_{dst_dataset_name}_venn.png"),
    #         verbose,
    #         overwrite
    #     )
    # )
    # run_stage(
    #     "Generating Venn diagram",
    #     lambda: venn_diagram(
    #         src_dataset_json,
    #         web_crawl_output_path,
    #         src_dataset_name,
    #         dst_dataset_name,
    #         "Species Overlap Between Datasets",
    #         target_classes,
    #         os.path.join(output_path, f"{src_dataset_name}_vs_{dst_dataset_name}_venn_target.png"),
    #         verbose,
    #         overwrite
    #     )
    # )

except FailedOperation as failedOp:
    print(failedOp, "\n")
    print(traceback.format_exc())
    exit()
