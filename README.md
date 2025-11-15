# AI CUP 2025 玉山人工智慧公開挑戰賽 — GraphSAGE-based GNN

## 介紹
本專案為「2025 玉山人工智慧公開挑戰賽」的實作範本，展示如何利用交易資料構建帳戶圖，並透過 Graph Neural Network（GraphSAGE）進行二元分類。

專案包含三個核心 Python 模組：

- `Preprocess/data_preprocess.py`：資料前處理與帳戶特徵工程  
- `Model/graph_based_model.py`：圖建構、GraphSAGE 模型、AMP、訓練  
- `main.py`：完整 pipeline（訓練與預測 `result.csv`）


---

## 資料說明
比賽提供某段期間的跨帳戶交易資料，目標是預測玉山帳戶在未來是否成為警示帳戶。

資料包含：

### 交易資料（`acct_transaction.csv`）
- 約 400 萬筆，每筆為一筆跨帳戶交易紀錄，含匯款帳戶、收款帳戶、交易日期時間、金額、幣別等

### 警示帳戶（`acct_alert.csv`）
- 約 1000 筆，每筆為一個警示帳戶及對應之警示日

### 需預測帳戶（`acct_predict.csv`）
- 約 4000 筆，需預測其是否在未來 1 個月內成為警示帳戶

專案會自動處理欄名不一致、資料 mapping、帳戶對齊等問題。


---

## 專案架構

```
.
├── Model/
│   ├── graph_based_model.py         # GraphSAGE 模型、AMP、訓練流程
│   └── README.md
│
├── Preprocess/
│   ├── data_preprocess.py           # CSV 載入、欄位對應、帳戶特徵萃取
│   └── README.md
│
├── main.py                          # 主程式：完整 pipeline 與結果輸出
├── requirements.txt                 # 套件需求
└── README.md                        # 本檔案
```

---

## 主要功能

- 自動辨識欄位名稱（如 `src` / `dst` / `amt`）
- 交易資料聚合 → 帳戶向量特徵
- 建構 Compresses Sparse Row (CSR) Adjacency（含雙向邊＋self-loop）
- GraphSAGE full-batch 訓練
- 在 Recall 不低於最低要求的前提下，從多個 threshold 中挑選 F1-score 最佳時的 threshold
- 輸出比賽格式之 `result.csv`


---

## 使用方式

### 1. 安裝套件
請使用 Python 3.10 或以上版本，及下列套件：

- pandas==2.0.0  
- scikit-learn==1.3.2  
- numpy==1.26.4  
- torch==2.1.0  

安裝指令：

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```


### 2. 準備資料
請將以下 CSV 放在 `dir_path`（預設為 `../preliminary_data/`）：

- `acct_transaction.csv`
- `acct_alert.csv`
- `acct_predict.csv`

若資料位置與預設不同，請在執行程式時於指令最後加上 `--data_dir dir_path` 來指定資料所在位置。


### 3. 執行程式

```bash
python main.py --device auto --epochs 3000 --patience 200 --hidden 128 --layers 3 --dropout 0.3 --lr 3e-3 --weight_decay 5e-4 --label_smoothing 0.02 --eval_every 10 --grad_ckpt
```

執行後會產生 `result.csv`。


---

## Preprocess/data_preprocess.py

這個 Python 程式負責比賽資料的前處理工作。它會讀取三個主要資料集、辨識不同版本資料中的欄位名稱，並將多筆交易紀錄聚合為帳戶層級特徵，最終輸出可直接供 GNN 使用的節點特徵矩陣。本模組是整個 GNN 模型的資料基礎。  

### 主要函式
- `load_csvs(data_dir)`：讀取三大資料集  
- `find_col(df, candidates)`：自動辨識欄位名稱  
- `resolve_columns(tx, alert, predict)`：統一欄位命名（來源、目的、金額等）  
- `build_account_features(tx, col)`：將交易紀錄轉成帳戶層級特徵  
  - 包含轉入/轉出金額統計、degree 特徵、交易次數、log1p特徵、是否為玉山銀行帳戶推斷  


---

## Model/graph_based_model.py

這個 Python 程式實作整個 GNN 模型的核心，包括圖結構建置、GraphSAGE 模型架構、閾值搜尋工具，以及完整的全圖訓練流程。 它負責將帳戶特徵與交易圖結合，並在 GPU/CPU 上執行高效率的全圖訓練與推論。  

### 主要元件
- `build_graph(tx, col, acct2idx)`：建立帳戶間的 CSR adjacency（支援雙向、自動加入 self-loop）
- `SAGELayer`：實作 GraphSAGE 的鄰居特徵聚合
- `SAGEBlock`：LayerNorm + PReLU + Dropout + Residual
- `SAGEModel`：多層 GraphSAGE（支援 gradient checkpointing）
- `train_model_fullbatch(...)`：全圖訓練、AMP、pos_weight、early stopping
- `select_conservative_threshold(...)`：以 Recall 下限條件搜尋最佳 threshold


---

## main.py

這個 Python 程式整合了本專案的所有流程，包含資料讀取、欄位解析、特徵工程、圖建構、模型訓練、閾值搜尋與最終結果輸出。 將前處理模組與 GNN 模型結合，形成以下流程。

1. 載入三大資料集  
2. 自動解析欄位名稱  
3. 建立帳戶特徵  
4. 將每一個帳戶的字串 ID 轉換成連續的編號，使帳戶能作為圖神經網路中的節點被使用
5. Construct CSR adjacency  
6. Full-batch GraphSAGE 訓練（AMP + early stopping）  
7. Threshold 搜尋  
8. 輸出 `result.csv`


---

## 實驗結果

- Validation F1 = **0.6739**（threshold = 0.500）  
- Conservative search 最佳 threshold = **0.696**  
- 使用 threshold = 0.696 → Validation F1 = **0.7029**  
- Model 預測 **122 個帳戶** 為 alert / suspicious  




