# Home_Working_Status_Analysis

Final project for 2021 PKUSZ@Digital Image Processing

## Prepare Dataset

Dataset is created from video frames. First copy corresponding video into the folder `data/video/{label_name}`, then run the script below.

```bash
cd data
python create_data.py -r 20
python split_train_test.py
```
