import os
import copy
import torch
import random
import numpy as np
from torch import nn
from typing import Any, List, Dict, Tuple
import pytest # Import pytest


from aero.helper import seed_everything, get_model_outputs, get_single_roi_extractor

def run_model(seed_feat: int, seed_roi: int, rois_tensor: torch.Tensor):
    """Helper function to create model, inputs, and get output."""
    seed_everything(seed_feat)
    single_roi_extractor = get_single_roi_extractor()
    out_channels = single_roi_extractor.out_channels

    # Feats generation is based on seed_feat
    feats = [
        torch.rand((1, out_channels, 200, 336)),
        torch.rand((1, out_channels, 100, 168)),
        torch.rand((1, out_channels, 50, 84)),
        torch.rand((1, out_channels, 25, 42)),
    ]

    seed_everything(seed_roi) # Seed for ROI might affect other things if not careful, but for just the tensor, it's less critical here
    # Use the passed rois_tensor directly
    rois = rois_tensor 

    model_inputs = {'feats': feats, 'rois': rois}
    output = get_model_outputs(single_roi_extractor, 'forward', model_inputs)
    return output, out_channels, rois.shape[0]


# --- Function to generate expected output for a given test case ---
def generate_expected_output_for_case(seed_feat: int, seed_roi: int, rois: torch.Tensor) -> torch.Tensor:
    """
    Generates the ground truth expected output for a given set of seeds and ROIs.
    This should be run once to capture the exact output for static expected values.
    """
    # Ensure seeding is consistent for generating the reference
    seed_everything(seed_feat)
    temp_single_roi_extractor = get_single_roi_extractor()
    temp_out_channels = temp_single_roi_extractor.out_channels
    temp_feats = [
        torch.rand((1, temp_out_channels, 200, 336)),
        torch.rand((1, temp_out_channels, 100, 168)),
        torch.rand((1, temp_out_channels, 50, 84)),
        torch.rand((1, temp_out_channels, 25, 42)),
    ]
    seed_everything(seed_roi)
    temp_rois = rois # Use the specific ROIs for this case

    temp_model_inputs = {'feats': temp_feats, 'rois': temp_rois}
    precise_expected_output = get_model_outputs(temp_single_roi_extractor, 'forward', temp_model_inputs)
    return precise_expected_output

# --- Define your test cases here ---

# Example ROIs for different tests
rois_case1 = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])
rois_case2 = torch.tensor([[0.0000, 662.433475, 124.496475, 811.643425, 269.208425]])
rois_case3 = torch.tensor([[0.0000, 513.223525, -20.215575, 960.853375, 413.920475]])
rois_case4 = torch.tensor([[0.0000, 700.0109625, 160.6744625, 774.0659375, 233.0304375]])
rois_case5 = torch.tensor([[0.0000, 438.61855, -92.57145, 1035.45835, 486.27635]])


# Dictionary to store the precise expected outputs.
# You need to fill this by running 'generate_expected_output_for_case' for each test case below.
# Example:
# torch.set_printoptions(precision=10)
# print(f"\n--- Generating expected output for Case 1 (feat:1234, roi:5678, rois_case1) ---")
# print(repr(generate_expected_output_for_case(1234, 5678, rois_case1)))

# print(f"\n--- Generating expected output for Case 2 (feat:1111, roi:2222, rois_case2) ---")
# print(repr(generate_expected_output_for_case(1111, 2222, rois_case2)))

# print(f"\n--- Generating expected output for Case 3 (feat:3333, roi:4444, rois_case3) ---")
# print(repr(generate_expected_output_for_case(3333, 4444, rois_case3)))

# print(f"\n--- Generating expected output for Case 4 (feat:5555, roi:6666, rois_case4) ---")
# print(repr(generate_expected_output_for_case(5555, 6666, rois_case4)))

# print(f"\n--- Generating expected output for Case 5 (feat:7777, roi:8888, rois_case5) ---")
# print(repr(generate_expected_output_for_case(7777, 8888, rois_case5)))
# and so on for all 5 cases.

expected_outputs_map = {
    "test_case_1": torch.tensor([[[[0.3465849757, 0.5128397346, 0.5110899806, 0.5482604504, 0.5056343079,
                                    0.5311501026, 0.6108055115],
                                    [0.5049589872, 0.4759848118, 0.6118276119, 0.3697176278, 0.3654493093,
                                    0.3364364207, 0.3461944461],
                                    [0.5681046844, 0.6062791348, 0.7786518335, 0.4520654678, 0.3312771320,
                                    0.4873413444, 0.4783429503],
                                    [0.6419377327, 0.6105794311, 0.6516363621, 0.5381178856, 0.4221541286,
                                    0.5601430535, 0.4478777945],
                                    [0.6376824975, 0.7091851234, 0.5606944561, 0.5813844204, 0.3035770059,
                                    0.5155543089, 0.3908332586],
                                    [0.4552578926, 0.5963292122, 0.6055278182, 0.5186353922, 0.3104445636,
                                    0.5164959431, 0.3619483113],
                                    [0.4917810559, 0.6542190909, 0.5295952559, 0.5941148996, 0.3207262456,
                                    0.4624288678, 0.4704295397]]]]), # Mocked
    "test_case_2": torch.tensor([[[[0.5269232988, 0.5064573884, 0.3129107952, 0.4318993688, 0.3199926615,
                                    0.3705605567, 0.6434665918],
                                    [0.4162221253, 0.4983438253, 0.5078741908, 0.3952633142, 0.3732519150,
                                    0.6506257057, 0.5086625218],
                                    [0.5377349854, 0.4755926430, 0.5341442823, 0.5052004457, 0.4427971244,
                                    0.4476976991, 0.5875057578],
                                    [0.6038594246, 0.5937914848, 0.5945696235, 0.5981339216, 0.5503132343,
                                    0.4188967943, 0.4090564251],
                                    [0.6374881864, 0.4397063553, 0.4768432379, 0.5272981524, 0.5492674112,
                                    0.5947561264, 0.4933271110],
                                    [0.3965471685, 0.4097659588, 0.4988619089, 0.4831130505, 0.4926891923,
                                    0.5253052711, 0.3164408803],
                                    [0.6492080688, 0.6357449889, 0.4159278572, 0.4158120751, 0.4951737523,
                                    0.5031883121, 0.5185737610]]]]), 
    "test_case_3": torch.tensor([[[[0.6223646402, 0.6566191316, 0.5974565744, 0.5538887978, 0.6200559139,
                                    0.5089467764, 0.3062607944],
                                    [0.4443974793, 0.4190994799, 0.7471989989, 0.5919553638, 0.4612738490,
                                    0.5550785065, 0.3769874573],
                                    [0.3752906322, 0.5031515956, 0.3258121610, 0.6284471750, 0.4673784971,
                                    0.3898342252, 0.4885273576],
                                    [0.4760673344, 0.4859030247, 0.3244329095, 0.5232285857, 0.5860123634,
                                    0.4935473502, 0.5364913344],
                                    [0.4883972406, 0.5156449080, 0.5280382037, 0.4890205264, 0.4814634323,
                                    0.4846664965, 0.5394366384],
                                    [0.4999735355, 0.5451403856, 0.3636220396, 0.4708228111, 0.5234858990,
                                    0.6388503909, 0.5037521124],
                                    [0.5550362468, 0.4391868114, 0.6094511747, 0.5598125458, 0.6597266197,
                                    0.4356312454, 0.5823569298]]]]),
    "test_case_4": torch.tensor([[[[0.5939379334, 0.4246172309, 0.3196474016, 0.2963446379, 0.5565569997,
                                    0.3756339550, 0.4800677896],
                                    [0.5011804700, 0.5793734193, 0.6723834276, 0.6148723960, 0.5288816690,
                                    0.5261361003, 0.2640726566],
                                    [0.5115469098, 0.3905014098, 0.5536121130, 0.6036446095, 0.7521017194,
                                    0.3811443448, 0.4452955723],
                                    [0.4652964771, 0.4717595577, 0.5139939785, 0.5330227017, 0.6282523870,
                                    0.3962504566, 0.5124903321],
                                    [0.5296352506, 0.5297706127, 0.5432767868, 0.5962522626, 0.6128376722,
                                    0.5721481442, 0.4223083258],
                                    [0.4321259558, 0.5023046732, 0.6533561945, 0.4042465091, 0.5352458954,
                                    0.3861585855, 0.5142617226],
                                    [0.4984455705, 0.4363396168, 0.5231940150, 0.5231662393, 0.6889831424,
                                    0.3346422017, 0.5530680418]]]]),
    "test_case_5": torch.tensor([[[[0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000,
                                    0.0000000000, 0.0000000000],
                                    [0.4525978565, 0.5257676840, 0.3865423799, 0.4027224779, 0.4597852826,
                                    0.5302714705, 0.5448592305],
                                    [0.5002649426, 0.6229658127, 0.4296990037, 0.5012235641, 0.4842900038,
                                    0.5267749429, 0.4884766936],
                                    [0.4942498803, 0.4771137834, 0.4424464703, 0.7212434411, 0.3657824099,
                                    0.3467974067, 0.4949225485],
                                    [0.5416152477, 0.5504436493, 0.4371532500, 0.3133995533, 0.5553355813,
                                    0.4549269080, 0.5355077386],
                                    [0.6354122758, 0.5728309751, 0.6854389906, 0.5386965275, 0.4933396578,
                                    0.6835676432, 0.3895457387],
                                    [0.3872296810, 0.5079483390, 0.6748830080, 0.7942975163, 0.5559839010,
                                    0.5412411094, 0.4170230627]]]])
}


# List of test parameters: (test_id, seed_feat, seed_roi, rois_tensor)
test_cases = [
    ("test_case_1", 1234, 5678, rois_case1),
    ("test_case_2", 1111, 2222, rois_case2),
    ("test_case_3", 3333, 4444, rois_case3),
    ("test_case_4", 5555, 6666, rois_case4),
    ("test_case_5", 7777, 8888, rois_case5),
]


@pytest.mark.parametrize("test_id, seed_feat, seed_roi, rois", test_cases)
def test_single_roi_extractor_deterministic_output_parametrized(test_id, seed_feat, seed_roi, rois):
    print(f"\nRunning test case: {test_id} with seed_feat={seed_feat}, seed_roi={seed_roi}, rois shape={rois.shape}")

    output1, out_channels, num_rois = run_model(seed_feat=seed_feat, seed_roi=seed_roi, rois_tensor=rois)
    output2, _, _ = run_model(seed_feat=seed_feat, seed_roi=seed_roi, rois_tensor=rois)

    expected_model_output = expected_outputs_map[test_id]

    assert isinstance(output1, torch.Tensor)
    assert isinstance(output2, torch.Tensor)
    assert isinstance(expected_model_output, torch.Tensor)

    assert output1.shape == expected_model_output.shape, f"Shape mismatch for {test_id}: {output1.shape} vs {expected_model_output.shape}"
    assert output2.shape == expected_model_output.shape, f"Shape mismatch for {test_id}: {output2.shape} vs {expected_model_output.shape}"

    assert torch.allclose(output1, output2, atol=1e-7, rtol=1e-5), \
        f"Internal non-determinism in run_model for {test_id}! Max diff: {(output1 - output2).abs().max()}"

    assert torch.allclose(output1, expected_model_output, atol=1e-6, rtol=1e-5), \
        f"Output mismatch for {test_id}! Max diff: {(output1 - expected_model_output).abs().max()}"

    assert torch.isfinite(output1).all()
    assert torch.isfinite(output2).all()
