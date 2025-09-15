<p align="center" width="60%">
<img src="assets/logo.png" alt="OpenEMMA" style="width: 35%; min-width: 200px; display: block; margin: auto; background-color: transparent;">
</p>

<p align="center">
    <a href="README.md"><strong>English</strong></a> | <a href="README_zh-CN.md"><strong>中文</strong></a> | <a href="README_ja-JP.md"><strong>日本語</strong></a>
</p>

<div id="top" align="center">

![Code License](https://img.shields.io/badge/Code%20License-Apache%202.0-brightgreen)
[![arXiv](https://img.shields.io/badge/arXiv-2412.15208-b31b1b.svg)](https://arxiv.org/abs/2412.15208)

</div>


# OpenEMMA: オープンソースのエンドツーエンド自動運転のためのマルチモーダルモデル
**OpenEMMA**は、[Waymoのエンドツーエンドマルチモーダルモデル（EMMA）](https://waymo.com/blog/2024/10/introducing-emma/)のオープンソース実装であり、自動運転車のモーションプランニングのためのエンドツーエンドフレームワークを提供します。**OpenEMMA**は、GPT-4やLLaVAなどのビジョンランゲージモデル（VLM）の事前学習された世界知識を活用し、テキストと前方カメラ入力を統合して、将来の自車ウェイポイントを正確に予測し、意思決定の理由を提供します。私たちの目標は、研究者や開発者が自動運転の研究と応用を進めるためのアクセス可能なツールを提供することです。

<div align="center">
  <img src="assets/EMMA-Paper-1__3_.webp" alt="EMMA diagram" width="800"/>
  <p><em>図1. EMMA: Waymoのエンドツーエンドマルチモーダルモデル。</em></p>
</div>

<div align="center">
  <img src="assets/openemma-pipeline.png" alt="OpenEMMA diagram" width="800"/>
  <p><em>図2. OpenEMMA: 事前学習されたVLMに基づくオープンソースのエンドツーエンド自動運転フレームワーク。</em></p>
</div>

### ニュース
- **[2025/1/12]** 🔥**OpenEMMA** は PyPI パッケージとして利用可能です！正式にインストールするには `pip install openemma` を使用してください。
- **[2024/12/19]** 🔥**OpenEMMA**をリリースしました。エンドツーエンドのモーションプランニング自動運転タスクのためのオープンソースプロジェクトです。詳細は[論文](https://arxiv.org/abs/2412.15208)をご覧ください。

### 目次
- [デモ](#デモ)
- [インストール](#インストール)
- [使用方法](#使用方法)
- [連絡先](#連絡先)
- [引用](#引用)

### デモ
![](assets/scene-0061.gif)

![](assets/scene-0103.gif)

![](assets/scene-1077.gif)

### インストール  
OpenEMMAを始めるには、以下の手順に従って環境と依存関係を設定してください。

1. **環境設定**  
   Python 3.8を使用してOpenEMMAのためのConda環境を設定します：
   ```bash
   conda create -n openemma python=3.8
   conda activate openemma
   ```
2. **OpenEMMA をインストールする**   
    PyPI を使用して、次のコマンドで OpenEMMA をインストールできます：
    ```bash
    pip install openemma
    ```
    または、次の手順に従ってください：
    - **OpenEMMAリポジトリのクローン**   
        OpenEMMAリポジトリをクローンし、ルートディレクトリに移動します：
        ```bash
        git clone git@github.com:taco-group/OpenEMMA.git
        cd OpenEMMA
        ```

    - **依存関係のインストール**  
        cudatoolkitがインストールされていることを確認します。インストールされていない場合は、以下のコマンドを使用します：
        ```bash
        conda install nvidia/label/cuda-12.4.0::cuda-toolkit
        ```
        OpenEMMAに必要なコアパッケージをインストールするには、以下のコマンドを実行します：
        ```bash
        pip install -r requirements.txt
        ```
        これにより、YOLO-3Dなどの重要なオブジェクト検出ツールの依存関係がすべてインストールされます。YOLO-3Dを実行するために必要なウェイトは、最初の実行時に自動的にダウンロードされます。

4. **GPT-4 APIアクセスの設定**  
    GPT-4の推論機能を有効にするために、OpenAIからAPIキーを取得します。コード内で直接APIキーを追加するか、環境変数として設定します：
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```
    これにより、OpenEMMAがGPT-4にアクセスして将来のウェイポイントと意思決定の理由を生成できるようになります。

### 使用方法  
環境を設定した後、以下の手順に従ってOpenEMMAを使用できます：

1. **入力データの準備**   
    [nuScenesデータセット](https://www.nuscenes.org/nuscenes#download)をダウンロードして解凍します。
    
2. **OpenEMMAの実行**  
    OpenEMMAのメインスクリプトを実行するには、以下のコマンドを使用します：
    - PyPI:
    ```bash
    openemma \
        --model-path qwen \
        --dataroot [dir-of-nuscnse-dataset] \
        --version [vesion-of-nuscnse-dataset] \
        --method openemma
    ```
    - Github Repo:
    ```bash
    python main.py \
        --model-path qwen \
        --dataroot [dir-of-nuscnse-dataset] \
        --version [vesion-of-nuscnse-dataset] \
        --method openemma
    ```

    現在、以下のモデルをサポートしています：`GPT-4o`、`LLaVA-1.6-Mistral-7B`、`Llama-3.2-11B-Vision-Instruct`、および`Qwen2-VL-7B-Instruct`。特定のモデルを使用するには、`--model-path`引数に`gpt`、`llava`、`llama`、および`qwen`を渡します。

3. **出力の解釈**   
    モデルを実行した後、OpenEMMAは`./qwen-reults`ディレクトリに以下の出力を生成します：

    - **ウェイポイント**: 自車の将来の軌道を予測するウェイポイントのリスト。

    - **意思決定の理由**: モデルの推論のテキスト説明。シーンのコンテキスト、重要なオブジェクト、および行動の決定を含みます。

    - **注釈付き画像**: 元の画像に重ねられた計画された軌道と検出された重要なオブジェクトの視覚化。

    - **コンパイルされたビデオ**: 注釈付き画像から作成されたビデオ（例：`output_video.mp4`）。時間経過とともに予測された経路を示します。

## 連絡先
このパッケージの使用に関するヘルプや問題については、GitHubのissueを提出してください。

このプロジェクトに関連する個人的な連絡は、Shuo Xing（shuoxing@tamu.edu）までお願いします。

## 引用
このコードがあなたの仕事に役立つことを嬉しく思います。
私たちのコードを使用するか、私たちの仕事を拡張する場合は、私たちの論文を引用してください：

```bibtex

@article{openemma,
	author = {Xing, Shuo and Qian, Chengyuan and Wang, Yuping and Hua, Hongyuan and Tian, Kexin and Zhou, Yang and Tu, Zhengzhong},
	title = {OpenEMMA: Open-Source Multimodal Model for End-to-End Autonomous Driving},
	journal = {arXiv},
	year = {2024},
	month = dec,
	eprint = {2412.15208},
	doi = {10.48550/arXiv.2412.15208}
}

```
