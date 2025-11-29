import io
import os
import zipfile
import tempfile
from typing import Annotated

import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from crop import crop_to_square

app = FastAPI(
    title="FaceCropper",
    summary="Center and Crop an Image of One Face into a Square",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/crop/", responses={200: {"content": {"image/jpeg": {}}}}, response_class=Response
)
async def crop(
    files: Annotated[
        list[UploadFile],
        File(description=f"Upload image(s) to crop."),
    ],
):
    if len(files) == 1:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, files[0].filename)
            with open(temp_file_path, "wb") as temp_file:
                contents = await files[0].read()
                temp_file.write(contents)

            cropped_img = crop_to_square(img_path=temp_file_path)
            _, im_buf_arr = cv2.imencode(f"crop.jpg", cropped_img)

        return Response(content=im_buf_arr.tobytes(), media_type="image/png")

    zip_filename = "archive.zip"
    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    for idx, file in enumerate(files):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, file.filename)
                with open(temp_file_path, "wb") as temp_file:
                    contents = await file.read()
                    temp_file.write(contents)

                cropped_img = crop_to_square(img_path=temp_file_path)

                with open(temp_file_path, "wb") as temp_file:
                    temp_file.flush()
                    _, im_buf_arr = cv2.imencode(f"crop_{idx}.jpg", cropped_img)
                    temp_file.write(im_buf_arr.tobytes())

                zf.write(temp_file_path, f"crop_{idx}.jpg")

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error reading file at index {idx}: {file.filename}. | {e}",
            )

    zf.close()

    return Response(
        s.getvalue(),
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment;filename={zip_filename}"},
    )
