#!/usr/bin/env python3
"""
Dataset categorization for RBM training.

This module defines categories for datasets to control sampling behavior:
- success: Datasets containing primarily successful trajectories
- failure: Datasets containing failure trajectories
- preference_only: Datasets that should only generate preference samples (no progress)
- paired: Datasets containing paired human/robot trajectories
"""

from typing import List

ALL_DATASOURCES = [
    "auto_eval_rfm",
    "failsafe",
    "fino_net",
    "h2r",
    "metaworld_train",
    "molmoact_dataset_household",
    "molmoact_dataset_tabletop",
    "motif_rfm",
    "oxe_aloha_mobile",
    "oxe_austin_buds_dataset_converted_externally_to_rlds",
    "oxe_bc_z",
    "oxe_berkeley_cable_routing",
    "oxe_berkeley_fanuc_manipulation",
    "oxe_berkeley_mvp_converted_externally_to_rlds",
    "oxe_berkeley_rpt_converted_externally_to_rlds",
    "oxe_bridge_v2",
    "oxe_dlr_edan_shared_control_converted_externally_to_rlds",
    "oxe_droid",
    "oxe_fractal20220817_data",
    "oxe_furniture_bench_dataset_converted_externally_to_rlds",
    "oxe_iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "oxe_imperialcollege_sawyer_wrist_cam",
    "oxe_jaco_play",
    "oxe_language_table",
    "oxe_nyu_rot_dataset_converted_externally_to_rlds",
    "oxe_robo_set",
    "oxe_stanford_hydra_dataset_converted_externally_to_rlds",
    "oxe_tokyo_u_lsmo_converted_externally_to_rlds",
    "oxe_toto",
    "oxe_ucsd_kitchen_dataset_converted_externally_to_rlds",
    "oxe_utaustin_mutex",
    "ph2d",
    "racer_train",
    "rh20t_human",
    "rh20t_robot",
    "roboarena",
    "soar_rfm",
    "usc_koch_human_robot_paired_human",
    "usc_koch_human_robot_paired_robot",
    "hand_paired_human",
    "hand_paired_robot",
    "armnetbench",
]

DATASET_CATEGORY = {
    "success": [
        # Most datasets are success by default, so this list is intentionally minimal
        # Only explicitly list if needed for special handling
    ],
    "failure": [
        "ykorkmaz_libero_failure_rfm_libero_90_failure",
        "ykorkmaz_libero_failure_rfm_libero_10_failure",
    ],
    "preference_only": [
        "jesbu1_oxe_rfm_oxe_bc_z",
        "jesbu1_oxe_rfm_oxe_dlr_edan_shared_control_converted_externally_to_rlds",
        "jesbu1_roboarena_0825_rfm_roboarena",
        "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist",
        # "roboarena",
        "auto_eval_rfm",
        "jesbu1_epic_rfm_epic",
    ],
    "paired": [
        "jesbu1_h2r_rfm_h2r",
        "jesbu1_motif_rfm_motif_rfm",
        "anqil_rh20t_subset_rfm_rh20t_human",
        "anqil_rh20t_subset_rfm_rh20t_robot",
        "jesbu1_ph2d_rfm_ph2d",
        "jesbu1_hand_paired_rfm_hand_paired_human",
        "jesbu1_hand_paired_rfm_hand_paired_robot",
        "jesbu1_usc_koch_human_robot_paired_human",
        "jesbu1_usc_koch_human_robot_paired_robot",
    ],
}

DATASET_MAP = {
    "armnetbench_v01": {
        "train": ["villekuosmanen_armnetbench_robometer_v01_so101"],
        "eval": ["villekuosmanen_armnetbench_robometer_v01_so101"],
    },
    "others": {
        "train": [
            "jesbu1_molmoact_rfm_molmoact_dataset_household",
            "jesbu1_molmoact_rfm_molmoact_dataset_tabletop",
            # "jesbu1_egodex_rfm_egodex_part1",
            # "jesbu1_egodex_rfm_egodex_part2",
            # "jesbu1_egodex_rfm_egodex_part3",
            # "jesbu1_egodex_rfm_egodex_part4",
            # "jesbu1_egodex_rfm_egodex_part5",
            "abraranwar_agibotworld_alpha_headcam_rfm_agibotworld",
            "jesbu1_humanoid_everyday_rfm_humanoid_everyday_rfm",
            "jesbu1_epic_rfm_epic",
            "jesbu1_galaxea_rfm_galaxea_part1_r1_lite",
            "jesbu1_galaxea_rfm_galaxea_part2_r1_lite",
            "jesbu1_galaxea_rfm_galaxea_part3_r1_lite",
            "jesbu1_galaxea_rfm_galaxea_part4_r1_lite",
            "jesbu1_galaxea_rfm_galaxea_part5_r1_lite",
        ],
        # "eval": [
        #    #"jesbu1_egodex_rfm_egodex_test",
        # ],
    },
    "oxe": {
        "train": [
            "jesbu1_oxe_rfm_oxe_aloha_mobile",
            "jesbu1_oxe_rfm_oxe_austin_buds_dataset_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_bc_z",
            "jesbu1_oxe_rfm_oxe_berkeley_cable_routing",
            "jesbu1_oxe_rfm_oxe_berkeley_fanuc_manipulation",
            "jesbu1_oxe_rfm_oxe_berkeley_mvp_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_berkeley_rpt_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_bridge_v2",
            "jesbu1_oxe_rfm_oxe_dlr_edan_shared_control_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_droid",
            "jesbu1_oxe_rfm_oxe_fractal20220817_data",
            "jesbu1_oxe_rfm_oxe_furniture_bench_dataset_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_iamlab_cmu_pickup_insert_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_imperialcollege_sawyer_wrist_cam",
            "jesbu1_oxe_rfm_oxe_jaco_play",
            "jesbu1_oxe_rfm_oxe_language_table",
            "jesbu1_oxe_rfm_oxe_nyu_rot_dataset_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_robo_set",
            "jesbu1_oxe_rfm_oxe_stanford_hydra_dataset_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_tokyo_u_lsmo_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_toto",
            "jesbu1_oxe_rfm_oxe_ucsd_kitchen_dataset_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_utaustin_mutex",
        ],
        "eval": [
            "jesbu1_oxe_rfm_eval_oxe_bc_z_eval",
            "jesbu1_oxe_rfm_eval_oxe_berkeley_cable_routing_eval",
            "jesbu1_oxe_rfm_eval_oxe_bridge_v2_eval",
            "jesbu1_oxe_rfm_eval_oxe_jaco_play_eval",
            "jesbu1_oxe_rfm_eval_oxe_toto_eval",
            "jesbu1_oxe_rfm_eval_oxe_viola_eval",
        ],
    },
    "mw": {
        "train": [
            "aliangdw_metaworld_metaworld_train",
        ],
        "eval": [
            "aliangdw_metaworld_metaworld_eval",
        ],
    },
    "reward_alignment": {
        "eval": [
            # "aliangdw_metaworld_metaworld_eval",
            "abraranwar_usc_koch_rewind_rfm_usc_koch_rewind",
            "jesbu1_roboarena_0825_rfm_roboarena",
            "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist",
            "jesbu1_soar_rfm_soar_rfm",
            # "jesbu1_egodex_rfm_egodex_test",
        ]
    },
    "quality_preference": {
        "eval": [
            # "aliangdw_metaworld_metaworld_eval",
            "jesbu1_roboarena_0825_rfm_roboarena",
            "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist",
            # "aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking",
            # "aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking",
            # "aliangdw_utd_so101_policy_ranking_utd_so101_policy_ranking",
            "jesbu1_soar_rfm_soar_rfm",
        ]
    },
    "policy_ranking": {
        "eval": [
            # "aliangdw_metaworld_metaworld_eval",
            # "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist",
            # "jesbu1_roboarena_0825_rfm_roboarena",
            "aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking",
            "aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking",
            "aliangdw_utd_so101_policy_ranking_utd_so101_policy_ranking",
            "jesbu1_soar_rfm_soar_rfm",
            "jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm",
            "jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top",
            "jesbu1_utd_so101_clean_policy_ranking_wrist_utd_so101_clean_policy_ranking_wrist",
            "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all",
            # "jesbu1_roboreward_rfm_roboreward_test",
            "jesbu1_roboreward_rfm_roboreward_test",
            # ["abraranwar_libero_rfm_libero256_10", "ykorkmaz_libero_failure_rfm_libero_10_failure"],
        ]
    },
    "paired": {
        "train": [
            "jesbu1_h2r_rfm_h2r",
            "jesbu1_motif_rfm_motif_rfm",
            "anqil_rh20t_subset_rfm_rh20t_human",
            "anqil_rh20t_subset_rfm_rh20t_robot",
            "jesbu1_ph2d_rfm_ph2d",
        ],
        "eval": [
            # "jesbu1_egodex_rfm_egodex_test",
            ["jesbu1_hand_paired_rfm_hand_paired_human", "jesbu1_hand_paired_rfm_hand_paired_robot"],
            [
                "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_human",
                "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_robot",
            ],
            ["anqil_rh20t_subset_rfm_rh20t_human", "anqil_rh20t_subset_rfm_rh20t_robot"],
            # "aliangdw_utd_so101_human_utd_so101_human",
            # "aliangdw_utd_so101_policy_ranking_utd_so101_policy_ranking",
        ],
    },
    "suboptimal_fail": {
        "train": [
            # "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist",
            "jesbu1_fino_net_rfm_fino_net",
            "jesbu1_failsafe_rfm_failsafe",
            "jesbu1_soar_rfm_soar_rfm",
            "jesbu1_auto_eval_rfm_auto_eval_rfm",
            "jesbu1_racer_rfm_racer_train",
            "jesbu1_roboarena_0825_rfm_roboarena",
            "jesbu1_roboreward_rfm_roboreward_train"
            # "jesbu1_roboreward_rfm_roboreward_train",
        ],
        "eval": [
            "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist",
            "jesbu1_racer_rfm_racer_val",
        ],
    },
    "franka": {
        "train": [
            "jesbu1_oxe_rfm_oxe_droid",
            "jesbu1_roboarena_0825_rfm_roboarena",
            "jesbu1_molmoact_rfm_molmoact_dataset_household",
            "jesbu1_molmoact_rfm_molmoact_dataset_tabletop",
        ],
        "eval": [
            "jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm",
            "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist",
            # "jesbu1_roboarena_0825_rfm_roboarena",
            "aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking",
            "aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking",
        ],
    },
    "libero": {
        "train": [
            "abraranwar_libero_rfm_libero256_90",
            "ykorkmaz_libero_failure_rfm_libero_90_failure",
        ],
        "eval": [
            "abraranwar_libero_rfm_libero256_10",
            "abraranwar_libero_rfm_libero256_object",
            "abraranwar_libero_rfm_libero256_spatial",
            "abraranwar_libero_rfm_libero256_goal",
            # "ykorkmaz_libero_failure_rfm_libero_90_failure",
            # "ykorkmaz_libero_failure_rfm_libero_10_failure",
        ],
    },
    "libero_pi0": {
        "train": [
            "abraranwar_libero_rfm_libero256_10",
            "abraranwar_libero_rfm_libero256_object",
            "abraranwar_libero_rfm_libero256_spatial",
            "abraranwar_libero_rfm_libero256_goal",
            "ykorkmaz_libero_failure_rfm_libero_10_failure",
            "ykorkmaz_libero_failure_rfm_libero_object_failure",
            "ykorkmaz_libero_failure_rfm_libero_spatial_failure",
            "ykorkmaz_libero_failure_rfm_libero_goal_failure",
        ],
        "eval": [
            [
                "abraranwar_libero_rfm_libero256_90",
                "ykorkmaz_libero_failure_rfm_libero_90_failure",
            ],
            [
                "abraranwar_libero_rfm_libero256_10",
                "ykorkmaz_libero_failure_rfm_libero_10_failure",
            ],
        ],
    },
    "libero_all": {
        "train": [
            "abraranwar_libero_rfm_libero256_10",
            "abraranwar_libero_rfm_libero256_object",
            "abraranwar_libero_rfm_libero256_spatial",
            "abraranwar_libero_rfm_libero256_goal",
            "ykorkmaz_libero_failure_rfm_libero_10_failure",
            "ykorkmaz_libero_failure_rfm_libero_object_failure",
            "ykorkmaz_libero_failure_rfm_libero_spatial_failure",
            "ykorkmaz_libero_failure_rfm_libero_goal_failure",
            "abraranwar_libero_rfm_libero256_90",
            "ykorkmaz_libero_failure_rfm_libero_90_failure",
        ],
        "eval": [
                "abraranwar_libero_rfm_libero256_90",
                "ykorkmaz_libero_failure_rfm_libero_90_failure",
        ],
    },
    "rbm-1m-id": {
        "train": [
            # franka
            "jesbu1_oxe_rfm_oxe_droid",
            "jesbu1_roboarena_0825_rfm_roboarena",
            "jesbu1_molmoact_rfm_molmoact_dataset_household",
            "jesbu1_molmoact_rfm_molmoact_dataset_tabletop",
            # oxe
            "jesbu1_oxe_rfm_oxe_aloha_mobile",
            "jesbu1_oxe_rfm_oxe_austin_buds_dataset_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_bc_z",
            "jesbu1_oxe_rfm_oxe_berkeley_cable_routing",
            "jesbu1_oxe_rfm_oxe_berkeley_fanuc_manipulation",
            "jesbu1_oxe_rfm_oxe_berkeley_mvp_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_berkeley_rpt_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_bridge_v2",
            "jesbu1_oxe_rfm_oxe_dlr_edan_shared_control_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_droid",
            "jesbu1_oxe_rfm_oxe_fractal20220817_data",
            "jesbu1_oxe_rfm_oxe_furniture_bench_dataset_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_iamlab_cmu_pickup_insert_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_imperialcollege_sawyer_wrist_cam",
            "jesbu1_oxe_rfm_oxe_jaco_play",
            "jesbu1_oxe_rfm_oxe_language_table",
            "jesbu1_oxe_rfm_oxe_nyu_rot_dataset_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_robo_set",
            "jesbu1_oxe_rfm_oxe_stanford_hydra_dataset_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_tokyo_u_lsmo_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_toto",
            "jesbu1_oxe_rfm_oxe_ucsd_kitchen_dataset_converted_externally_to_rlds",
            "jesbu1_oxe_rfm_oxe_utaustin_mutex",
            # others
            "jesbu1_molmoact_rfm_molmoact_dataset_household",
            "jesbu1_molmoact_rfm_molmoact_dataset_tabletop",
            "abraranwar_agibotworld_alpha_headcam_rfm_agibotworld",
            "jesbu1_humanoid_everyday_rfm_humanoid_everyday_rfm",
            "jesbu1_epic_rfm_epic",
            "jesbu1_galaxea_rfm_galaxea_part1_r1_lite",
            "jesbu1_galaxea_rfm_galaxea_part2_r1_lite",
            "jesbu1_galaxea_rfm_galaxea_part3_r1_lite",
            "jesbu1_galaxea_rfm_galaxea_part4_r1_lite",
            "jesbu1_galaxea_rfm_galaxea_part5_r1_lite",
            # libero
            "abraranwar_libero_rfm_libero256_10",
            "abraranwar_libero_rfm_libero256_object",
            "abraranwar_libero_rfm_libero256_spatial",
            "abraranwar_libero_rfm_libero256_goal",
            "ykorkmaz_libero_failure_rfm_libero_10_failure",
            "ykorkmaz_libero_failure_rfm_libero_object_failure",
            "ykorkmaz_libero_failure_rfm_libero_spatial_failure",
            # suboptimal_fail
            "jesbu1_fino_net_rfm_fino_net",
            "jesbu1_failsafe_rfm_failsafe",
            "jesbu1_soar_rfm_soar_rfm",
            "jesbu1_auto_eval_rfm_auto_eval_rfm",
            "jesbu1_racer_rfm_racer_train",
            "jesbu1_roboarena_0825_rfm_roboarena",
            "jesbu1_roboreward_rfm_roboreward_train",
            # paired
            "jesbu1_h2r_rfm_h2r",
            "jesbu1_motif_rfm_motif_rfm",
            "anqil_rh20t_subset_rfm_rh20t_human",
            "anqil_rh20t_subset_rfm_rh20t_robot",
            "jesbu1_ph2d_rfm_ph2d",
            # mw
            "aliangdw_metaworld_metaworld_train",
        ],
        "eval": [
            # "abraranwar_libero_rfm_libero256_10",
            # "abraranwar_libero_rfm_libero256_object",
            # "abraranwar_libero_rfm_libero256_spatial",
            # "abraranwar_libero_rfm_libero256_goal",
            # "ykorkmaz_libero_failure_rfm_libero_10_failure",
            # "ykorkmaz_libero_failure_rfm_libero_object_failure",
            # "ykorkmaz_libero_failure_rfm_libero_spatial_failure",
            # "ykorkmaz_libero_failure_rfm_libero_goal_failure",
            # "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist",
            # "jesbu1_roboarena_0825_rfm_roboarena",
            "jesbu1_racer_rfm_racer_val",
            # "jesbu1_soar_rfm_soar_rfm",
            # "jesbu1_auto_eval_rfm_auto_eval_rfm",
            # "jesbu1_roboreward_rfm_roboreward_test",
            "jesbu1_oxe_rfm_eval_oxe_bc_z_eval",
            "jesbu1_oxe_rfm_eval_oxe_berkeley_cable_routing_eval",
            "jesbu1_oxe_rfm_eval_oxe_bridge_v2_eval",
            "jesbu1_oxe_rfm_eval_oxe_jaco_play_eval",
            "jesbu1_oxe_rfm_eval_oxe_toto_eval",
            "jesbu1_oxe_rfm_eval_oxe_viola_eval",
            "aliangdw_metaworld_metaworld_eval",
            "abraranwar_libero_rfm_libero256_90",
        ],
    },
    "rbm-1m-ood": {
        "eval": [
            # "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist",  # putting this here for now
            # "jesbu1_roboreward_rfm_roboreward_test",
            # "jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm",
            "ykorkmaz_usc_trossen_rfm_usc_trossen",
            "jesbu1_rfm_new_mit_franka_rfm_rfm_new_mit_franka_rfm",
            "jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top",
            "aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking",
            "aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking",
            "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all",
        ]
    },
    "libero_pi0_no_fail": {
        "train": [
            "abraranwar_libero_rfm_libero256_10",
            "abraranwar_libero_rfm_libero256_object",
            "abraranwar_libero_rfm_libero256_spatial",
            "abraranwar_libero_rfm_libero256_goal",
        ],
        "eval": [
            [
                "abraranwar_libero_rfm_libero256_90",
                "ykorkmaz_libero_failure_rfm_libero_90_failure",
            ],
            [
                "abraranwar_libero_rfm_libero256_10",
                "ykorkmaz_libero_failure_rfm_libero_10_failure",
            ],
        ],
    },
}


DATA_SOURCE_CATEGORY = {
    "success": [
        # All datasets are success by default, so this list is intentionally minimal
        # Only explicitly list if needed for special handling
    ],
    "failure": [
        "libero_90_failure",
        "libero_10_failure",
        "libero_object_failure",
        "libero_spatial_failure",
        "libero_goal_failure",
    ],
    "preference_only": [
        "oxe_bc_z",
        "oxe_dlr_edan_shared_control_converted_externally_to_rlds",
        "auto_eval_rfm",
        # "humanoid_everyday_rfm",
        "galaxea_part1_r1_lite",
        "galaxea_part2_r1_lite",
        "galaxea_part3_r1_lite",
        "galaxea_part4_r1_lite",
        "galaxea_part5_r1_lite",
        "epic",
        "roboreward_roboarena",
        "roboreward_dlr_edan_shared_control_converted_externally_to_rldsroboarena",
        "roboarena",
        "roboarena_eval_debug_nowrist",
    ],
    "paired": [
        "h2r",
        "motif_rfm",
        "rh20t_human",
        "rh20t_robot",
        "ph2d",
        "hand_paired_human",
        "hand_paired_robot",
        "usc_koch_human_robot_paired_human",
        "usc_koch_human_robot_paired_robot",
    ],
    "suboptimal_fail": [
        "roboarena",
        "fino_net",
        "failsafe",
        "soar_rfm",
        "auto_eval_rfm",
        "racer_train",
        "racer_val",
        "roboarena_eval_debug_nowrist",
        "armnetbench",
    ],
    "franka": [
        "oxe_droid",
        "roboarena",
        "molmoact_dataset_household",
        "molmoact_dataset_tabletop",
    ],
}

# DATASET helper functions


def get_preference_only_datasets() -> List[str]:
    """Get list of datasets that should only generate preference samples."""
    return DATASET_CATEGORY.get("preference_only", [])


def get_paired_datasets() -> List[str]:
    """Get list of datasets containing paired human/robot trajectories."""
    return DATASET_CATEGORY.get("paired", [])


def get_failure_datasets() -> List[str]:
    """Get list of datasets containing failure trajectories."""
    return DATASET_CATEGORY.get("failure", [])


def get_success_datasets() -> List[str]:
    """Get list of datasets containing successful trajectories."""
    return DATASET_CATEGORY.get("success", [])


def is_preference_only(dataset_name: str) -> bool:
    """Check if a dataset should only generate preference samples."""
    return dataset_name in DATASET_CATEGORY.get("preference_only", [])


def is_paired(dataset_name: str) -> bool:
    """Check if a dataset contains paired human/robot trajectories."""
    return dataset_name in DATASET_CATEGORY.get("paired", [])


def is_failure(dataset_name: str) -> bool:
    """Check if a dataset contains failure trajectories."""
    return dataset_name in DATASET_CATEGORY.get("failure", [])


def is_success(dataset_name: str) -> bool:
    """Check if a dataset contains successful trajectories."""
    return dataset_name in DATASET_CATEGORY.get("success", [])


# DATA_SOURCE helper functions


def get_preference_only_ds() -> List[str]:
    """Get list of data sources that should only generate preference samples."""
    return DATA_SOURCE_CATEGORY.get("preference_only", [])


def get_paired_ds() -> List[str]:
    """Get list of data sources containing paired human/robot trajectories."""
    return DATA_SOURCE_CATEGORY.get("paired", [])


def get_failure_ds() -> List[str]:
    """Get list of data sources containing failure trajectories."""
    return DATA_SOURCE_CATEGORY.get("failure", [])


def get_success_ds() -> List[str]:
    """Get list of data sources containing successful trajectories."""
    return DATA_SOURCE_CATEGORY.get("success", [])


def is_preference_only_ds(data_source_name: str) -> bool:
    """Check if a data source should only generate preference samples."""
    return data_source_name in DATA_SOURCE_CATEGORY.get("preference_only", [])


def is_paired_ds(data_source_name: str) -> bool:
    """Check if a data source contains paired human/robot trajectories."""
    return data_source_name in DATA_SOURCE_CATEGORY.get("paired", [])


def is_failure_ds(data_source_name: str) -> bool:
    """Check if a data source contains failure trajectories."""
    return data_source_name in DATA_SOURCE_CATEGORY.get("failure", [])


def is_success_ds(data_source_name: str) -> bool:
    """Check if a data source contains successful trajectories."""
    return data_source_name in DATA_SOURCE_CATEGORY.get("success", [])


def get_suboptimal_fail_ds() -> List[str]:
    """Get list of data sources containing suboptimal/failure trajectories."""
    return DATA_SOURCE_CATEGORY.get("suboptimal_fail", [])


def is_suboptimal_fail_ds(data_source_name: str) -> bool:
    """Check if a data source contains suboptimal/failure trajectories."""
    return data_source_name in DATA_SOURCE_CATEGORY.get("suboptimal_fail", [])
