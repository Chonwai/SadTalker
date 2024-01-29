from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from inference import main
from argparse import Namespace
from typing import List, Optional
import torch
import shutil
import os
import uuid

app = FastAPI()


class InferenceRequest(BaseModel):
    # driven_audio: str
    # source_image: str
    ref_eyeblink: Optional[str] = None
    ref_pose: Optional[str] = None
    checkpoint_dir: str = "./checkpoints"
    result_dir: str = "./results"
    pose_style: int = 0
    batch_size: int = 2
    size: int = 256
    expression_scale: float = 1.0
    input_yaw: Optional[List[int]] = None
    input_pitch: Optional[List[int]] = None
    input_roll: Optional[List[int]] = None
    enhancer: Optional[str] = None
    background_enhancer: Optional[str] = None
    cpu: bool = False
    face3dvis: bool = False
    still: bool = True
    preprocess: str = "crop"
    verbose: bool = False
    old_version: bool = False
    net_recon: str = "resnet50"
    init_path: Optional[str] = None
    use_last_fc: bool = False
    bfm_folder: str = "./checkpoints/BFM_Fitting/"
    bfm_model: str = "BFM_model_front.mat"
    focal: float = 1015.0
    center: float = 112.0
    camera_d: float = 10.0
    z_near: float = 5.0
    z_far: float = 15.0


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/inference")
async def run_inference(
    driven_audio: UploadFile = File(...),
    source_image: UploadFile = File(...),
    ref_eyeblink: Optional[str] = Form(None),
    ref_pose: Optional[str] = Form(None),
    checkpoint_dir: str = Form("./checkpoints"),
    # result_dir: str = Form("./results"),
    pose_style: int = Form(0),
    batch_size: int = Form(2),
    size: int = Form(256),
    expression_scale: float = Form(1.0),
    input_yaw: Optional[List[int]] = Form(None),
    input_pitch: Optional[List[int]] = Form(None),
    input_roll: Optional[List[int]] = Form(None),
    enhancer: Optional[str] = Form(None),
    background_enhancer: Optional[str] = Form(None),
    cpu: bool = Form(False),
    face3dvis: bool = Form(False),
    still: bool = Form(True),
    preprocess: str = Form("crop"),
    verbose: bool = Form(False),
    old_version: bool = Form(False),
    net_recon: str = Form("resnet50"),
    init_path: Optional[str] = Form(None),
    use_last_fc: bool = Form(False),
    bfm_folder: str = Form("./checkpoints/BFM_Fitting/"),
    bfm_model: str = Form("BFM_model_front.mat"),
    focal: float = Form(1015.0),
    center: float = Form(112.0),
    camera_d: float = Form(10.0),
    z_near: float = Form(5.0),
    z_far: float = Form(15.0),
):
    result_folder_name = str(uuid.uuid4())
    print(result_folder_name)
    audio_path = f"./results/{result_folder_name}/temp_{driven_audio.filename}"
    image_path = f"./results/{result_folder_name}/temp_{source_image.filename}"
    with open(audio_path, "wb") as audio_file:
        shutil.copyfileobj(driven_audio.file, audio_file)
    with open(image_path, "wb") as image_file:
        shutil.copyfileobj(source_image.file, image_file)

    try:
        args = Namespace(
            driven_audio=audio_path,
            source_image=image_path,
            ref_eyeblink=ref_eyeblink,
            ref_pose=ref_pose,
            checkpoint_dir=checkpoint_dir,
            result_dir="./results/" + str(result_folder_name),
            pose_style=pose_style,
            batch_size=batch_size,
            size=int(size),
            expression_scale=expression_scale,
            input_yaw=input_yaw,
            input_pitch=input_pitch,
            input_roll=input_roll,
            enhancer=enhancer,
            background_enhancer=background_enhancer,
            cpu=cpu,
            face3dvis=face3dvis,
            still=still,
            preprocess=preprocess,
            verbose=verbose,
            old_version=old_version,
            net_recon=net_recon,
            init_path=init_path,
            use_last_fc=use_last_fc,
            bfm_folder=bfm_folder,
            bfm_model=bfm_model,
            focal=focal,
            center=center,
            camera_d=camera_d,
            z_near=z_near,
            z_far=z_far,
        )

        if torch.cuda.is_available() and not args.cpu:
            args.device = "cuda"
        else:
            args.device = "cpu"

        print(args)

        main(args)  # 調用main函數

        os.remove(audio_path)
        os.remove(image_path)

        result_file_path = f"./results/{result_folder_name}".format(
            result_folder_name=result_folder_name
        )
        mp4_files = [f for f in os.listdir(result_file_path)]

        print(mp4_files)

        # return {"message": "Inference completed successfully"}
        return FileResponse(
            path=result_file_path, filename=mp4_files[0], media_type="video/mp4"
        )
    except Exception as e:
        os.remove(audio_path)
        os.remove(image_path)
        raise HTTPException(status_code=500, detail=str(e.message))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
