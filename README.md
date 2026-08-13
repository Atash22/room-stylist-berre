# Room Stylist — Berre.ca Furniture Recommender

An AI interior-style classifier and furniture recommender built for Berre.ca,
covering the full Full Stack Data Science Systems pipeline:
**Notebook → Script → Flask API → Docker → Azure (live public URL)**.

Given a photo of a room, the model classifies its interior style into one of
six categories and recommends real, in-catalog Berre products that match.

**Live demo API:** `http://room-stylist-final-atash.canadacentral.azurecontainer.io:5000/predict`
*(Azure Container Instance — may be stopped between demos to save cost; see
"Running the live deployment" below to bring it back up.)*

## Structure

```
room_stylist/
  notebook/
    train_benchmark.py        # local/Jupyter training+benchmarking script
    room_style_colab.ipynb    # the actual notebook used (Colab, T4 GPU,
                               # includes dropout + early stopping)
  app/
    predict.py                # inference script — loads weights once,
                               # exposes predict_style(image_bytes) -> dict
    catalog.py                # product lookup keyed by predicted style
                               # (real Berre.ca products, prices, images)
    app.py                    # Flask API: POST /predict, GET /health, CORS enabled
    Dockerfile
    requirements.txt
    style_classifier_effnet_b0.pt   # exported winning model weights
  results/
    benchmark_results.json          # accuracy / macro F1 / latency summary
    full_benchmark_results.json     # + confusion matrices
    classification_reports.txt      # full per-class precision/recall/F1
  tidy_dataset.py              # reorganizes the raw 19-class Kaggle dataset
                                # into the 6 Berre-aligned classes below
  room_stylist_demo_v2.html    # standalone browser demo (fetch-based,
                                # shows style bars + real product photo cards)
  data/                        # NOT committed — see Data section below
```

## Data

Source: the Houzz-sourced "Interior design styles" Kaggle dataset — 19 raw
style folders (asian, coastal, contemporary, craftsman, eclectic, farmhouse,
french-country, industrial, mediterranean, mid-century-modern, modern,
rustic, scandinavian, shabby-chic-style, southwestern, traditional,
transitional, tropical, victorian), each with `dataset_train` and
`dataset_test` splits.

`tidy_dataset.py` consolidates those 19 raw labels into **6 classes that
match how Berre.ca actually groups its furniture**, and carves a validation
split out of train:

| Berre-aligned class | Raw classes folded in |
|---|---|
| `mid_century_modern` | mid-century-modern, modern |
| `contemporary_scandinavian` | contemporary, scandinavian |
| `traditional_classic` | traditional, victorian, french-country, mediterranean, asian |
| `rustic_farmhouse` | farmhouse, rustic, craftsman, southwestern |
| `coastal_tropical` | coastal, tropical |
| `eclectic_industrial` | eclectic, transitional, shabby-chic-style, industrial |

Result: `data/train`, `data/val`, `data/test`, each with the 6 class
subfolders above. **`data/` is git-ignored** — regenerate it locally with
`tidy_dataset.py` rather than committing raw images (also keeps the repo
under GitHub's file size limits — don't zip and commit the dataset).

## Training & Benchmarking

Run in `notebook/room_style_colab.ipynb` (Colab, T4 GPU — training on CPU is
impractical at this dataset size). Compares three approaches:

- **ResNet-18** — transfer learning, frozen backbone + `Dropout(0.3) → Linear` head
- **EfficientNet-B0** — same setup, frozen backbone + dropout head
- **CLIP zero-shot** (ViT-B/32) — no training, prompt-based baseline

Both trained models use dropout regularization and early stopping
(patience=3 on validation accuracy, restores best-epoch weights).

### Results

| Model | Accuracy | Macro F1 | Latency (ms/img) |
|---|---|---|---|
| ResNet-18 | 0.3816 | 0.3283 | 4.3 |
| **EfficientNet-B0** | **0.4073** | **0.3736** | 4.4 |
| CLIP zero-shot | 0.3341 | 0.3108 | 10.8 |

**EfficientNet-B0 was selected for deployment** — best accuracy and macro F1
with no latency penalty vs. ResNet-18. Notably, CLIP zero-shot underperformed
both fine-tuned models here: ~2,100 images/class was enough signal for
transfer learning to beat a general-purpose foundation model on this
fine-grained, visually-overlapping-classes task. Full per-class reports and
confusion matrices are in `results/`.

## Running it locally

**1. Test the inference script directly:**
```
cd app
pip install -r requirements.txt --break-system-packages
python predict.py path/to/room.jpg
```

**2. Run the Flask API:**
```
python app.py
curl -X POST -F "image=@room.jpg" http://localhost:5000/predict
```

**3. Build and run the Docker container:**
```
docker build -t room-stylist .
docker run -p 5000:5000 room-stylist
curl -X POST -F "image=@room.jpg" http://localhost:5000/predict
```

## Deployment (Azure)

Hosted on **Azure Container Registry** + **Azure Container Instances**
(not Render/Railway/Fly.io — moved to Azure since it integrates with the
course's Azure for Students credit).

```
# Push a new image version
docker tag room-stylist roomstylist.azurecr.io/room-stylist:vN
docker push roomstylist.azurecr.io/room-stylist:vN

# Deploy (delete any existing instance of the same name first)
az container create --resource-group room-stylist-rg --name room-stylist-final \
  --image roomstylist.azurecr.io/room-stylist:vN --os-type Linux \
  --cpu 1 --memory 3.5 \
  --registry-login-server roomstylist.azurecr.io \
  --registry-username roomstylist --registry-password <see Azure Portal Access keys> \
  --dns-name-label room-stylist-final-atash --ports 5000
```

### Pausing / resuming the live deployment

To avoid burning Azure credit between demos, stop rather than delete the
container instance — it keeps its full configuration and comes back up at
the same URL:
```
az container stop --resource-group room-stylist-rg --name room-stylist-final
az container start --resource-group room-stylist-rg --name room-stylist-final
```

## Browser demo

`room_stylist_demo_v2.html` is a standalone page (no server needed — just
open it in a browser) that calls the live Azure endpoint directly via
`fetch()`, and renders:
- the predicted style + confidence
- a bar chart of all 6 class probabilities
- recommended Berre products as real photo cards (image, name, price, link)

This requires CORS to be enabled on the Flask API (`flask-cors`, already
wired up in `app.py`) since the page and the API are on different origins.

## Presentation notes

- **ML Canvas**: multi-class image classification; input = room photo,
  output = 1-of-6 style label + confidence; metrics = accuracy + macro F1
  (classes are imbalanced after the 19→6 consolidation, so macro F1 matters
  alongside accuracy).
- **Benchmarking**: EfficientNet-B0 wins on accuracy and macro F1 with no
  latency cost; CLIP zero-shot is a legitimate baseline to include even
  though it underperformed here — that's itself a finding worth presenting.
- **Licensing caveat**: the Houzz-sourced dataset is for academic/research
  use only — don't redistribute the images; the trained weights and results
  are fine to share, the raw dataset is not.

## Known limitations / honest caveats

- Test accuracy (~41%) is well above the 17% random baseline for 6 classes,
  but interior style classification is a genuinely hard, somewhat subjective
  task — `contemporary_scandinavian` and `mid_century_modern` in particular
  get confused with each other (see `results/classification_reports.txt`).
- The product catalog in `catalog.py` uses two fully-verified real Berre
  products (with real images/prices/URLs) applied across all 6 style
  buckets for demo purposes — a production version would need a larger,
  per-style-mapped catalog.
