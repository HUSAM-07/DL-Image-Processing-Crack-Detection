# Deep Learning for Image Processing & Crack-Detection
Image Processing for Crack Detection  using XFEM, Machine Learning &amp; Deep Learning Techniques

For Extended Summary, Refer to [Notes](/notes.md)

*_The Repo is obviously not well structured(at least to my liking), This will be imporved and properly documented after the research reaches it's ideal stages and a paper is submitted_*

### Summarized Poster:
![Poster](/Explanation_Poster.png)

```mermaid
flowchart TB
    A[Input Image] --> B[Preprocessing]
    B --> C[Feature Extraction]
    C --> D[Crack Path Extraction]
    D --> E[Output Image]

    subgraph Preprocessing
        B --> B1[Resize Image 224x224]
        B1 --> B2[Convert to Grayscale]
        B2 --> B3[Edge Detection]
        B3 --> B4[Hough Transform]
        B4 --> B5[Processed Image]
    end

    subgraph "Feature Extraction"
        C --> C1[VGG-16 Convolutional Layers]
        C1 --> C2[Extracted Features]
    end

    subgraph "Crack Path Extraction"
        D --> D1[Dense Layer 512 units]
        D1 --> D2[Dense Layer 256 units]
        D2 --> D3[Dense Layer 128 units]
        D3 --> D4[Batch Normalization]
        D4 --> D5[Dropout 0.3]
        D5 --> D6[Output Dense Layer 224x224]
        D6 --> D7[Sigmoid Activation]
        D7 --> D8[Reshape Layer 224x224x1]
        D8 --> D9[Crack Path Output]
    end
```

## Updated High-Level Architecture

```mermaid

flowchart TD
subgraph Data[Data Pipeline]
A[Initial Images<br/>frame_0] --> B[Image Preprocessing]
C[Final Images<br/>frame_-1] --> B
B --> D[Train/Val/Test Split]
end
subgraph Model[Model Architecture]
E[Encoder<br/>CNN Layers] --> F[Latent Space]
F --> G[Decoder<br/>Transpose CNN]
end
subgraph Training[Training Process]
H[Loss Function<br/>MSE/MAE] --> I[Optimization<br/>Adam]
I --> J[Validation]
end
D --> E
G --> H
J --> |Feedback|E
subgraph Evaluation[Model Evaluation]
K[Test Set<br/>Predictions] --> L[Metrics<br/>SSIM/PSNR]
L --> M[Visual<br/>Comparison]
end

```
