# WildfireSimulator

## Environment
```bash
conda env create -f environment.yml
conda activate wildfire_env
pip install -e .
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # or thru here: https://pytorch.org/get-started/locally/
```

## Data
Data is not traditionally saved in a `.npy` file for training. Rather, `.tifs` are directly converted into the correct format for training in RAM through a DataLoader. Create a `.env` file that points to these tifs:

```bash
LANDSCAPE=/run/media/hrotovb001/9021-552E/FireDataset/palisades.tif
TRIALS=/run/media/hrotovb001/9021-552E/FireDataset/Trials/
IGNITIONS=/run/media/hrotovb001/9021-552E/FireDataset/Ignitions/
```

Currently there are 3112 samples. Each on the Palisades landscape, with parameterized:
- Ignition point (ign)
- Wind speed (ws)
- Wind direction (wd)
- Moisture (moisture)

```mermaid
graph TD
    subgraph "Raw Data (on disk)"
        A["palisades.tif<br/>(8-band landscape)"]
        B["Ignitions/*.shp<br/>(50 point shapefiles)"]
        C["Trials/*.tif<br/>(3211 arrival time rasters)"]
    end

    subgraph "Data Loading"
        D["WildfireDataLoader"]
        E["TrialCollection<br/>(lazy file loader)"]
    end

    subgraph "Dataset Assembly"
        F["WildfireDataset<br/>→ (13, 500, 500) per sample"]
        G["MinMaxPerChannel<br/>→ normalized [0,1]"]
        H["TransformedDataset"]
    end

    subgraph "Training-Time Processing"
        I["BurnerBatchProcessor(t)"]
        J["ForwardBurnProcess<br/>(slices fire state at time t)"]
        K["Input: (14, 512, 512)<br/>[fire@t + landscape + t_channel]"]
        L["Target: (2, 512, 512)<br/>[fire_mask@(t+dt), arrival@(t+dt)]"]
    end

    subgraph "Model"
        M["MK_UNet_Regression<br/>in=14, out=2"]
    end

    A --> D
    B --> D
    C --> E
    E --> D
    D --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    J --> L
    K --> M
    M -->|"pred (2, 512, 512)"| I
```

## Run training
Simply through `python run.py`.
