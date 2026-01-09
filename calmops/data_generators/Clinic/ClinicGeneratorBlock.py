import os
import pandas as pd
from typing import List, Dict, Optional, Any
from calmops.data_generators.Synthetic.SyntheticBlockGenerator import (
    SyntheticBlockGenerator,
)
from calmops.data_generators.DriftInjection.DriftInjector import DriftInjector
from calmops.data_generators.Dynamics.DynamicsInjector import DynamicsInjector
from calmops.data_generators.Clinic.ClinicGenerator import ClinicGenerator
from calmops.data_generators.Clinic.ClinicReporter import ClinicReporter


class ClinicGeneratorBlock(SyntheticBlockGenerator):
    """
    Generator for Clinical data blocks.
    Wraps SyntheticBlockGenerator logic but utilizes ClinicGenerator for feature mapping
    and ClinicReporter for specialized reporting.
    """

    def generate(
        self,
        output_dir: str,
        filename: str,
        n_blocks: int,
        total_samples: int,
        n_samples_block,
        generators,
        target_col="target",
        balance: bool = False,
        date_start: str = None,
        date_step: dict = None,
        date_col: str = "timestamp",
        generate_report: bool = True,
        drift_config: Optional[List[Dict]] = None,
        dynamics_config: Optional[Dict] = None,
        block_labels: Optional[List[Any]] = None,
    ) -> str:
        # Reuse helper from parent to ensure lists
        n_samples_block = self._ensure_list(n_samples_block, n_blocks)
        generators = self._ensure_list(generators, n_blocks)
        if len(set(type(g) for g in generators)) > 1:
            raise ValueError("All generator instances must be of the same type.")

        if sum(n_samples_block) != total_samples:
            raise ValueError(
                f"Total samples ({total_samples}) must equal the sum of instances per block ({sum(n_samples_block)})"
            )

        if block_labels:
            if len(block_labels) != n_blocks:
                raise ValueError(
                    f"Length of block_labels ({len(block_labels)}) must match n_blocks ({n_blocks})."
                )
        else:
            block_labels = list(range(1, n_blocks + 1))

        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)

        block_dates = None
        if date_start:
            start_ts = pd.to_datetime(date_start)
            step = pd.DateOffset(**(date_step or {"days": 1}))
            block_dates = [start_ts + step * i for i in range(n_blocks)]

        all_data = []
        # USE CLINIC GENERATOR HERE
        clinic_generator = ClinicGenerator()

        for i in range(n_blocks):
            gen = generators[i]
            n_samples_this_block = n_samples_block[i]
            current_block_label = block_labels[i]

            block_df = clinic_generator.generate(
                generator_instance=gen,
                metadata_generator_instance=gen,
                output_dir=output_dir,
                filename=f"block_{str(current_block_label)}.csv",
                n_samples=n_samples_this_block,
                target_col=target_col,
                balance=balance,
                date_start=block_dates[i].isoformat() if block_dates else None,
                date_every=n_samples_this_block,
                date_col=date_col,
                save_dataset=False,
                generate_report=False,  # We aggregate at the end
            )
            block_df["block"] = current_block_label
            all_data.append(block_df)

        df = pd.concat(all_data, ignore_index=True)

        # --- Dynamics Injection ---
        if dynamics_config:
            injector = DynamicsInjector()
            if "evolve_features" in dynamics_config:
                evolve_args = dynamics_config["evolve_features"]
                df = injector.evolve_features(df, time_col=date_col, **evolve_args)
            if "construct_target" in dynamics_config:
                target_args = dynamics_config["construct_target"]
                df = injector.construct_target(df, **target_args)

        # --- Drift Injection ---
        if drift_config:
            injector = DriftInjector(
                original_df=df,
                output_dir=output_dir,
                generator_name="ClinicGeneratorBlock_Drifted",
                target_column=target_col,
                block_column="block",
                time_col=date_col,
            )
            for drift_conf in drift_config:
                method_name = drift_conf.get("method")
                params = drift_conf.get("params", {})
                if hasattr(injector, method_name):
                    drift_method = getattr(injector, method_name)
                    try:
                        if "df" not in params:
                            params["df"] = df
                        res = drift_method(**params)
                        if isinstance(res, pd.DataFrame):
                            df = res
                    except Exception as e:
                        print(f"Failed to apply drift {method_name}: {e}")
                        raise e
                else:
                    print(f"Drift method '{method_name}' not found.")

        df.to_csv(full_path, index=False)
        print(f"Generated {total_samples} samples in {n_blocks} blocks at: {full_path}")

        if generate_report:
            # USE CLINIC REPORTER HERE
            reporter = ClinicReporter(verbose=True)
            reporter.generate_report(
                synthetic_df=df,
                generator_name="ClinicGeneratorBlock",
                output_dir=output_dir,
                target_column=target_col,
                block_column="block",
                time_col=date_col,
            )

        return full_path
