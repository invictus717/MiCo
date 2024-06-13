### Data Preparation

#### Install Download Tools
```bash
pip install video2dataset
```
or
```bash
git clone https://github.com/iejMac/video2dataset
cd video2dataset
pip install -e .
```
#### Download Metadata
```bash
wget -O hdvila100m.zip https://hdvila.blob.core.windows.net/dataset/hdvila100m.zip?sp=r&st=2022-06-28T03:33:11Z&se=2026-01-01T11:33:11Z&spr=https&sv=2021-06-08&sr=b&sig=VaqQkLFDqKinfkaPNs1jJ1EQIYCB%2FUPYiqFqmjWye6Y%3D
```
Then unzip the metadata zip file.
```bash
unzip hdvilla100m.zip
```
With the metadata, we will deal with these data into parquet files by running this code:
```bash
python makeparquet.py
```
Once you run this, you should have a file `hd_vila.parquet` with all the relevant metadata. The files are organized as:
```bash
data
├── caption_config
├── model
├── scripts
├── utils
├── makeparquet.py
├── config.yaml
├── download_hdvila.sh
├── hdvila
│   ├── hdvila_part0.jsonl 
│   ├── hdvila_part1.jsonl 
│   ├── hdvila_part2.jsonl 
│   ├── hdvila_part3.jsonl 
│   ├── hdvila_part4.jsonl
│   ├── hdvila_part5.jsonl
│   ├── hdvila_part6.jsonl
│   ├── hdvila_part7.jsonl
│   ├── hdvila_part8.jsonl
│   ├── hdvila_part9.jsonl
│   ├── hdvila_part10.jsonl
│   ├── hd_vila.parquet
```
#### Download HDVILA-100M Source Data
Please check your path in `download_hdvila.sh` before running the script for downloading the dataset:
```bash
bash download_hdvila.sh
```
#### Annotate Your Videos
1. Download Pretrained Captioners for Videos (Images) and Audio.
    ```bash
    pip install gdown
    gdown https://drive.google.com/file/d/1vYqb0Lb_3sQ5bo6XV-FQ4n7k_0M9UMU3/view?usp=sharing
    tar -xvf audio_captioner.tar.gz
    gdown https://drive.google.com/file/d/1ZFCWZ8csMWLYsg9CWt71PJmKYpSn-FMt/view?usp=sharing
    tar -xvf vision_captioner.tar.gz
    ```
2. Deploy captioners for data annotation
    Set up the python environment for captioner.
    ```bash
    bash setup_env.sh
    ```

    Video Annotation with Captions
    ```bash 
    bash scripts/run_vision_captioner.sh
    ```

    Audio Annotation with Captions
    ```bash 
    bash scripts/run_audio_captioner.sh
    ```
3. (Optional) Deploy Depth Estimator to annotate 3D contents
*We highly recommend you to use [GeoWizard](https://github.com/fuxiao0719/GeoWizard) to generate high-quality 3D contents*.
while the shortage of *GeoWizard* is the inference speed of generative models. Therefore, in our practice, we use the [DPT](https://github.com/EPFL-VILAB/omnidata) to annotate major data.