# AI CUP 2025 玉山人工智慧公開挑戰賽 — Model/graph_based_model.py

## 檔案說明
`graph_based_model.py`包含本專案使用的 **圖神經模型建構與訓練流程**，將「帳戶交易資料」轉成圖結構，再利用 **GraphSAGE** 做「警示帳戶」預測。模型相關程式碼皆由 `main.py` 呼叫並執行訓練。  

---

## 主要功能
- 自動混合精度（Automatic Mixed Precision, AMP）： 在適當的運算路徑中自動切換使用 float16／bfloat16 與 float32，以降低記憶體使用量並提升訓練效能，同時封裝不同 PyTorch 版本的 AMP 介面差異
- 圖結構建立： 從帳戶轉帳紀錄建立含雙向邊與 self-loop 的 CSR adjacency matrix，作為圖神經網路的輸入
- 多層 GraphSAGE 模型： 實作 GraphSAGE 多層架構，透過聚合自身特徵與鄰居特徵，並結合 LayerNorm、PReLU、Dropout 與殘差連接，對每個帳戶節點進行警示與否的二元分類
- 全圖訓練流程： 提供 full-batch 訓練與 early stopping，整合 class weight、label smoothing 與 AMP
- Threshold 搜尋： 在給定的 threshold 區間內，排除 train 與 validation 上 recall 過低選項，再由 validation F1 選取最佳 threshold 

---


## 自動混合精度（AMP）

為了在不同版本的 PyTorch 上穩定啟用 AMP，本模組提供一層輕量封裝：

- 優先使用 `torch.amp.autocast`、`torch.amp.GradScaler`
- 若環境不支援，退回 `torch.cuda.amp.autocast`、`torch.cuda.amp.GradScaler`
- 若仍不可用，相關程式碼自動退為 no-op，不影響訓練流程

這讓主訓練程式可以用統一界面開啟混合精度訓練，而不需要額外處理 PyTorch 版本差異。

---


## 圖結構建立：`build_graph(...)`

```python
A = build_graph(tx, col, acct2idx, undirected=True)
```

此函式會將帳戶間的轉帳紀錄轉換為 CSR 格式的圖鄰接矩陣，流程如下：

1. 從 `tx` DataFrame 中取出 `(src, dst)` pair，移除 NaN 與重複紀錄  
2. 只保留出現在 `acct2idx` 中的帳戶  
3. 若 `undirected=True`，對每條邊 `u → v` 補上一條 `v → u`，形成無向圖  
4. 對每個節點加入 self-loop（`i → i`）  
5. 計算每個節點向外連接的邊數，並設定該節點列的邊權重為 `1 / out_degree`，完成列正規化  
6. 回傳形狀為 `(N, N)` 的 CSR 稀疏張量，可直接輸入 GraphSAGE 模型

若遭遇資料中完全沒有邊的極端情況，函式會退回只包含 self-loop 的「近似單位矩陣」圖。

---

## GraphSAGE 模型架構

### A. `SAGELayer`

單一 GraphSAGE layer 會分別對「節點自身特徵向量」與「該節點鄰居聚合後的特徵向量」做線性轉換，最後將兩者相加：

$$
h_i' = W_{\text{self}} h_i + W_{\text{neigh}} \sum_j A_{ij} h_j
$$

- `lin_self`：對節點本身特徵的線性層  
- `lin_neigh`：對鄰居聚合特徵（`A @ x`）的線性層  
- 假設 `A` 已是 row-normalized adjacency，因此不額外處理 normalization

### B. `SAGEBlock`

`SAGEBlock` 的結構為：
- `SAGELayer`  
- `LayerNorm` 
- `PReLU` 
- `Dropout` 
- `Residual`

`Residual` 為殘差連接，將輸出與原始輸入透過 skip connection 相加：
- 若輸入與輸出維度不同，先對輸入套用 `Linear(in_dim, out_dim)` 再相加
- 若維度相同，則使用 `Identity` 當作 shortcut


### C. `SAGEModel`

```python
model = SAGEModel(
    in_dim=feat_dim,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3,
    use_ckpt=False,
)
```

- 主體結構：由 `num_layers` 個 `SAGEBlock` 依序堆疊而成，每層的隱藏維度皆為 `hidden_dim` 
- Classification Head：將前面 GraphSAGE 得到的節點向量，轉換為可判斷警示帳戶的分數
  - `Linear(hidden_dim, hidden_dim)`
  - `LayerNorm`
  - `PReLU`
  - `Dropout`
  - `Linear(hidden_dim, 1)` → 對每個節點輸出單一 logit 分數  

  訓練時搭配 `BCEWithLogitsLoss`，推論與 threshold 搜尋時則以 `torch.sigmoid` 將 logit 轉成機率。

- 初始化與訓練穩定性：
  - 所有 `nn.Linear` 權重使用 **Xavier uniform** 初始化，bias 全部設為 0
  - `use_ckpt=True` 時，在訓練模式下使用 `torch.utils.checkpoint` 對各 block 做梯度重計算，可顯著降低 GPU 記憶體用量  

---

## 全圖訓練流程：`train_model_fullbatch(...)`

`train_model_fullbatch` 在單一裝置上進行 **full-batch 訓練**，整張圖一次送入模型。常見呼叫方式如下：

```python
model, best_val_thr, best_val_f1 = train_model_fullbatch(
    model=model,
    x_all=x_all,
    A=A,
    y_all=y_all,
    train_idx=train_idx,
    val_idx=val_idx,
    lr=3e-3,
    weight_decay=5e-4,
    epochs=1200,
    patience=120,
    label_smoothing=0.02,
    eval_every=10,
    device=device,
    amp_dtype=torch.bfloat16,
)
```

流程概要：

1. **裝置與 dtype 設定**  
   - 將 `model`, `x_all`, `A`, `y_all` 搬到指定 `device`（CPU 或 CUDA）  
   - 若為 CUDA 並指定 `amp_dtype`，會嘗試將模型與輸入轉為對應 dtype（如 `bfloat16`）

2. **處理類別不平衡**  
   - 以 `train_idx` 中正負樣本比例計算 `pos_weight`：  
     - `pos_weight = sqrt(max(neg / pos, 1.0))`  
   - 使用 `F.binary_cross_entropy_with_logits` 時帶入此 `pos_weight`

3. **Label smoothing**  
   - 若 `label_smoothing > 0`，將 label 略為往 0.5 收斂，減少過度自信預測

4. **優化與學習率調整**  
   - Optimizer：`AdamW`  
   - Scheduler：`ReduceLROnPlateau`，以 validation F1 為監控指標（`mode="max"`）

5. **AMP 與梯度裁切**  
   - 若使用 CUDA，依照 `amp_dtype` 適度啟用 `autocast` 與 `GradScaler`  
   - 每次反向傳播後使用 `clip_grad_norm_(model.parameters(), max_grad_norm=5.0)` 防止梯度爆炸

6. **Validation 與 early stopping**  
   - 每 `eval_every` 個 epoch 在 validation 上評估一次，threshold 固定為 0.5  
   - 以 validation F1 作為 early stopping 主指標  
   - 若連續多次評估 F1 無提升，觸發 early stopping  
   - 始終保留 validation F1 最佳時的模型權重

7. **回傳**  
   - 最終將最佳權重載回模型，並轉回 CPU＋float32 後回傳  
   - 同時回傳：
     - `best_val_thr`：目前實作中固定為 0.5，用於訓練內部評估  
     - `best_val_f1`：訓練過程中觀察到的最佳 validation F1

---

## Threshold 搜尋

模組提供兩種 threshold 工具，用於從預測機率中選擇分類門檻：

### `find_best_threshold(y_true, y_prob, steps=400)`

- 在 `[0.01, 0.99]` 區間中均勻掃描多個 threshold
- 對每個 threshold 計算 F1 score
- 回傳 F1 最高的 `(best_thr, best_f1)`

此函式適合在單一資料集（例如 validation）上做純 F1 最佳化。

### `select_conservative_threshold(train_y, train_p, val_y, val_p, thr_min=0.3, thr_max=0.7, steps=200, min_recall=0.35)`

本專案在最終預測時，會使用此函式來選擇較保守的決策門檻。流程如下：

1. 只在 `[thr_min, thr_max]` 區間內掃描 threshold（預設 0.3～0.7）
2. 對每個候選 threshold 計算：
   - train 上的 recall
   - validation 上的 recall
3. 若任一側 recall 低於 `min_recall`，即捨棄此 threshold
4. 在「train 與 validation 超過 `min_recall`」的候選集合中：
   - 選擇 validation F1 最佳的 threshold
   - 若 F1 相近則以保守為主，偏好較低 threshold

若沒有任何 threshold 通過 recall 下限，則退回預設門檻（如 0.5）。


---
