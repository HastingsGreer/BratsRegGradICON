import argparse
import glob
import os

import icon_registration.pretrained_models
import itk
import numpy as np
import pandas as pd
import torch
from icon_registration import itk_wrapper


def get_model():

    # Read in your trained model
    model = icon_registration.pretrained_models.brain_registration_model(
        pretrained=True
    )
    # model.regis_net.load_state_dict(torch.load("/usr/bin/brain_registration_model.trch"))
    return model


def generate_output(args):
    """
    Generates landmarks, detJ, deformation fields (optional), and followup_registered_to_baseline images (optional) for challenge submission
    """
    print("generate_output called")

    input_path = os.path.abspath(args["input"])
    output_path = os.path.abspath(args["output"])

    print(
        f"* Found following data in the input path {input_path}=",
        os.listdir(input_path),
    )  # Found following data in the input path /input= ['BraTSReg_001', 'BraTSReg_002']
    print(
        "* Output will be written to=", output_path
    )  # Output will be written to= /output

    model = get_model()

    # Now we iterate through each subject folder under input_path
    for subj_path in glob.glob(os.path.join(input_path, "BraTSReg*")):
        subj = os.path.basename(subj_path)
        print(
            f"Now performing registration on {subj}"
        )  # Now performing registration on BraTSReg_001

        # Read in your data
        input_pre_path = glob.glob(os.path.join(subj_path, f"{subj}_00_*_t1.nii.gz"))[0]
        input_pre = itk.imread(input_pre_path)
        input_pre = icon_registration.pretrained_models.brain_network_preprocess(input_pre)
        input_post_path = glob.glob(os.path.join(subj_path, f"{subj}_01_*_t1.nii.gz"))[
            0
        ]
        input_post = itk.imread(input_post_path)
        input_post = icon_registration.pretrained_models.brain_network_preprocess(input_post)

        phi_pre_post, phi_post_pre = itk_wrapper.register_pair(
            model, input_pre, input_post, finetune_steps=None
        )

        # Make your prediction segmentation file for case BraTSReg_001

        ## 1. calculate the output landmark points
        post_landmarks_path = glob.glob(
            os.path.join(subj_path, f"{subj}_01_*_landmarks.csv")
        )[0]
        post_landmarks = pd.read_csv(post_landmarks_path).values

        pre_landmarks = np.array(
            [
                list(phi_pre_post.TransformPoint(t[1:]))
                for t in post_landmarks * 1.0
            ]
        )

        pre_landmarks_ind = post_landmarks.copy()
        pre_landmarks_ind[:, 1:] = pre_landmarks

        np.savetxt(pre_landmarks_ind, os.path.join(args["output"], f"{subj}.csv"), header="Landmark,X,Y,Z", delimiter=",")

        itk.transformwrite([phi_pre_post], os.path.join(output_path, f"{subj}_df_f2b.hdf5"))

        ## 2. calculate the determinant of jacobian of the deformation field
        import SimpleITK as sitk

        phi_pre_post_sitk = sitk.ReadTransform(os.path.join(output_path, f"{subj}_df_f2b.hdf5"))
        filt = sitk.TransformToDisplacementFieldFilter()
        filt.SetReferenceImage(
            sitk.ReadImage(
                glob.glob(os.path.join(subj_path, f"{subj}_00_*_t1.nii.gz"))[0]
            )
        )
        displacement_image_sitk = filt.Execute(phi_pre_post_sitk)
        det_sitk = sitk.DisplacementFieldJacobianDeterminantFilter().Execute(
            displacement_image_sitk
        )

        ## write your output_detj to the output folder as BraTSReg_001.nii.gz
        sitk.WriteImage(det_sitk, os.path.join(args["output"], f"{subj}_detj.nii.gz"))

        if args["def"]:
            # write both the forward and backward deformation fields to the output/ folder
            print("--def flag is set to True")
            # write(output_def_followup_to_baseline, os.path.join(args["output"], f"{subj}_df_f2b.nii.gz"))
            # write(output_def_baseline_to_followup, os.path.join(args["output"], f"{subj}_df_b2f.nii.gz"))

        if args["reg"]:
            # write the followup_registered_to_baseline sequences (all 4 sequences provided) to the output/ folder
            print("--reg flag is set to True")
            # write(output_followup_to_baseline_t1ce, os.path.join(args["output"], f"{subj}_t1ce_f2b.nii.gz"))
            # write(output_followup_to_baseline_t1, os.path.join(args["output"], f"{subj}_t1_f2b.nii.gz"))
            # write(output_followup_to_baseline_t2, os.path.join(args["output"], f"{subj}_t2_f2b.nii.gz"))
            # write(output_followup_to_baseline_flair, os.path.join(args["output"], f"{subj}_flair_f2b.nii.gz"))


def apply_deformation(args):
    """
    Applies a deformation field on an input image and saves/returns the output
    """
    print("apply_deformation called")

    # Read the field
    f = read(path_to_deformation_field)

    # Read the input image
    i = read(path_to_input_image)

    # apply field on image and get output
    o = apply_field_on_image(f, i, interpolation_type)

    # If a save_path is provided then write the output there, otherwise return the output
    if save_path:
        write(o, savepath)
    else:
        return o


if __name__ == "__main__":
    # You can first check what devices are available to the singularity
    # setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Additional Info when using cuda
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Parse the input arguments

    parser = argparse.ArgumentParser(
        description="Argument parser for BraTS_Reg challenge"
    )

    subparsers = parser.add_subparsers()

    command1_parser = subparsers.add_parser("generate_output")
    command1_parser.set_defaults(func=generate_output)
    command1_parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="/input",
        help="Provide full path to directory that contains input data",
    )
    command1_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="/output",
        help="Provide full path to directory where output will be written",
    )
    command1_parser.add_argument(
        "-d",
        "--def",
        action="store_true",
        help="Output forward and backward deformation fields",
    )
    command1_parser.add_argument(
        "-r",
        "--reg",
        action="store_true",
        help="Output followup scans registered to baseline",
    )

    command2_parser = subparsers.add_parser("apply_deformation")
    command2_parser.set_defaults(func=apply_deformation)
    command2_parser.add_argument(
        "-f",
        "--field",
        type=str,
        required=True,
        help="Provide full path to deformation field",
    )
    command2_parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="Provide full path to image on which field will be applied",
    )
    command2_parser.add_argument(
        "-t",
        "--interpolation",
        type=str,
        required=True,
        help="Should be nearest_neighbour (for segmentation mask type images) or trilinear etc. (for normal scans). To be handled inside apply_deformation() function",
    )
    command2_parser.add_argument(
        "-p",
        "--path_to_output_nifti",
        type=str,
        default=None,
        help="Format: /path/to/output_image_after_applying_deformation_field.nii.gz",
    )

    args = vars(parser.parse_args())

    print("* Received the following arguments =", args)

    args["func"](args)
