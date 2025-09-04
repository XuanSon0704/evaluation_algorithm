import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
import time
import warnings
import matplotlib.pyplot as plt  

# Bỏ qua các cảnh báo
warnings.filterwarnings('ignore')

# THAY ĐỔI: Thay thế LoOPAlgorithm bằng LOFAlgorithm
class LOFAlgorithm:
    
    def __init__(self, k=20, contamination=0.1):
        # Chuyển đổi k thành n_neighbors
        self.n_neighbors = k
        self.contamination = contamination
        # novelty=True là bắt buộc để có thể dự đoán trên dữ liệu mới
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination, novelty=True)
        
    def fit(self, X_train_scaled):
        # Cần đủ mẫu để fit
        if len(X_train_scaled) < self.n_neighbors:
            return False
        try:
            self.model.fit(X_train_scaled)
            return True
        except Exception as e:
            print(f"Lỗi khi fit LOF: {e}")
            return False
        
    def predict(self, X_test_scaled):
        # predict trả về 1 cho inlier (bình thường), -1 cho outlier (bất thường)
        predictions = self.model.predict(X_test_scaled)
        # Chuyển đổi: 1 (inlier) -> 0 (BENIGN), -1 (outlier) -> 1 (ATTACK)
        return (predictions == -1).astype(int)

# Class StreamingAnomalyDetector 
class StreamingAnomalyDetector:
    def __init__(self, window_size=180, k_neighbors=15, contamination=0.01):
        self.window_size = window_size
        self.k_neighbors = k_neighbors
        self.contamination = contamination
        self.results = {'timestamps': [], 'true_labels': [], 'predictions': []}

    def detect_anomalies(self, df, selected_features, attack_label='ATTACK'):

       # Chuẩn bị dữ liệu 
        df = df.copy()
        df['Second'] = pd.to_datetime(df['Second'])
        
        # Tạo label dựa trên timestamp
        attack_start_time = pd.to_datetime('10:01:26').time()
        df['Label'] = np.where(df['Second'].dt.time >= attack_start_time, attack_label, 'BENIGN')
        
        # Preprocessing
        for col in selected_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=selected_features + ['Label'], inplace=True)
        
        # Log transformation(Lấy log feature->giảm những outlier có biến động mạnh)
        for col in selected_features:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        
        df = df.reset_index(drop=True)
        
        start_time = df['Second'].min()
        end_time = df['Second'].max()
        current_time = start_time
        
        print(f"Xử lý từ {start_time} đến {end_time}")
        
        # Main processing loop
        while current_time <= end_time:
            window_start_time = current_time - pd.Timedelta(seconds=self.window_size - 1)
            
            # Lấy dữ liệu cửa sổ hiện tại
            window_df = df[(df['Second'] >= window_start_time) & (df['Second'] <= current_time)]
            current_point_df = df[df['Second'] == current_time]
            
            # Tiến trình xử lí
            if (current_time - start_time).total_seconds() % 60 == 0:
                progress = (current_time - start_time).total_seconds() / (end_time - start_time).total_seconds() * 100
                print(f"Tiến độ: {progress:.1f}% - Thời gian: {current_time.time()}")
            
            if not current_point_df.empty:
                # Chỉ dùng BENIGN data để tìm đặc điểm "bình thường"
                train_df = window_df[window_df['Label'] == 'BENIGN']
                
                if len(train_df) >= self.k_neighbors + 5:
                    X_train = train_df[selected_features].values
                    X_current = current_point_df[selected_features].values
                    
                    # Chuản hóa sử dụng RobustScaler
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_current_scaled = scaler.transform(X_current)
                    
                    # KNNAlgorithm
                    model = LOFAlgorithm(k=self.k_neighbors, contamination=self.contamination)
                    
                    if model.fit(X_train_scaled):
                        prediction = model.predict(X_current_scaled)[0] 
                    else:
                        prediction = 0
                else:
                    prediction = 0

                true_label = 1 if current_point_df['Label'].values[0] == attack_label else 0
                
                self.results['timestamps'].append(current_time)
                self.results['true_labels'].append(true_label)
                self.results['predictions'].append(prediction) 
            
            current_time += pd.Timedelta(seconds=1)
        return self.results

    def evaluate_performance(self):
        y_true = np.array(self.results['true_labels'])
        y_pred = np.array(self.results['predictions'])
        
        attack_code = np.max(y_true) if np.any(y_true > 0) else 1

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0, average='binary', pos_label=attack_code)
        recall = recall_score(y_true, y_pred, zero_division=0, average='binary', pos_label=attack_code)
        f1 = f1_score(y_true, y_pred, zero_division=0, average='binary', pos_label=attack_code)
        cm = confusion_matrix(y_true, y_pred, labels=[0, attack_code])
        
        attack_indices = np.where(y_true == attack_code)[0]
        detected_indices = np.where((y_true == attack_code) & (y_pred == attack_code))[0]
        
        detection_delay = (self.results['timestamps'][detected_indices[0]] - self.results['timestamps'][attack_indices[0]]).total_seconds() if len(attack_indices) > 0 and len(detected_indices) > 0 else float('inf')
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'confusion_matrix': cm, 'detection_delay': detection_delay, 'total_samples': len(y_true), 'attack_samples': np.sum(y_true == attack_code), 'detected_attacks': np.sum((y_true == attack_code) & (y_pred == attack_code))}

# FUNCTION VẼ BIỂU ĐỒ
def plot_multi_k_analysis_lof(performance_results):
    """
    Vẽ 4 biểu đồ riêng biệt, mỗi metric một biểu đồ với các đường k khác nhau
    """
    window_sizes = [90, 100, 120, 150, 180, 200]
    k_values = sorted(list(performance_results.keys()))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Màu sắc cho các đường  
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "#9467bd", "#8c564b", '#e377c2', '#7f7f7f', '#bcbd22', "#17becf"]

    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        # Vẽ các đường cho từng k
        for i, k in enumerate(k_values):
            values = []
            for window_size in window_sizes:
                if window_size in performance_results[k]:
                    values.append(performance_results[k][window_size][metric] * 100)
                else:
                    values.append(0)  # Giá trị mặc định nếu không có dữ liệu
            
            # Vẽ đường 
            linewidth = 3.0 + (i * 0.3)  # Độ dày khác nhau
            linestyle = '-' if i < 4 else '--' if i < 6 else '-.'  # Style khác nhau
            line = ax.plot(window_sizes, values, 
                          linewidth=linewidth, 
                          linestyle=linestyle,
                          color=colors[i % len(colors)], 
                          label=f'k={k}',
                          alpha=0.85)  # Độ trong suốt để nhìn thấy đường bên dưới
            
            # Thêm chú thích k trên đường (ở điểm cuối) 
            if values:  # Kiểm tra nếu có dữ liệu
                ax.annotate(f'k={k}', 
                           xy=(window_sizes[-1], values[-1]), 
                           xytext=(15, 0), 
                           textcoords='offset points',
                           fontsize=11, 
                           fontweight='bold',
                           color='white',  # Chữ trắng
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=colors[i % len(colors)], 
                                   alpha=0.9, 
                                   edgecolor='white',
                                   linewidth=1.5))
        
        # Tùy chỉnh từng subplot
        ax.set_xlabel('Window Size (giây)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric_name} (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} theo Window Size và K', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 105)
        ax.set_xticks(window_sizes)
        
        # Legend với style đẹp hơn
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, 
                 fontsize=11, ncol=2, columnspacing=0.8)
    
    plt.suptitle('Phân tích hiệu suất thuật toán LOF theo K và Window Size', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('lof_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig('lof_metrics.png')
    
    
def main():

    # Cấu hình tham số
    K_VALUES = [2,4,6,8,10,12,14,16,18,20] 
    WINDOW_SIZES = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]  
    CONTAMINATION = 0.02
    
    selected_features = ['Packets_s', 'Bytes_s', 'New_Flow_s']
    ATTACK_LABEL = 'Portmap'
    data_file_path = '/home/xson/LOF-LoOP/evaluation/Portmap_BENIGN_ATTACK1.csv'
    
    print("=" * 80)
    print("PHÂN TÍCH ĐA K CHO THUẬT TOÁN LOF")
    print(f"K values: {K_VALUES}")
    print(f"Window sizes: {WINDOW_SIZES}")
    print("=" * 80)
    

    df = pd.read_csv(data_file_path)
    
    # Dictionary để lưu kết quả: {k_value: {window_size: performance}}
    all_results = {}
    
    total_combinations = len(K_VALUES) * len(WINDOW_SIZES)
    current_combination = 0
    
    # Test với từng combination của k và window_size
    for k in K_VALUES:
        print(f"\n--- Testing K = {k} ---")
        all_results[k] = {}
        
        for window_size in WINDOW_SIZES:
            current_combination += 1
            progress = (current_combination / total_combinations) * 100
            print(f"[{progress:5.1f}%] K={k}, Window={window_size}s...", end=" ")
            
            detector = StreamingAnomalyDetector(
                window_size=window_size, 
                k_neighbors=k, 
                contamination=CONTAMINATION
            )
            
            start_time = time.time()
            detector.detect_anomalies(df.copy(), selected_features, ATTACK_LABEL)
            end_time = time.time()
            
            performance = detector.evaluate_performance()
            all_results[k][window_size] = performance
            
            print(f"F1={performance['f1_score']:.3f} ({end_time-start_time:.1f}s)")
    
    # Vẽ biểu đồ 4 panel
    plot_multi_k_analysis_lof(all_results)
    
    # In kết quả
    print("\n" + "=" * 80)
    print("KẾT QUẢ TỔNG HỢP - F1 SCORE")
    print("=" * 80)
    print("K\\Window".ljust(8), end="")
    for w in WINDOW_SIZES:
        print(f"{w:>8}", end="")
    print()
    print("-" * (8 + 8 * len(WINDOW_SIZES)))
    
    for k in K_VALUES:
        print(f"{k:<8}", end="")
        for window_size in WINDOW_SIZES:
            f1 = all_results[k][window_size]['f1_score']
            print(f"{f1:>8.3f}", end="")
        print()
    
    # Tìm combination tốt nhất
    best_f1 = 0
    best_k = 0
    best_window = 0
    for k in K_VALUES:
        for window_size in WINDOW_SIZES:
            f1 = all_results[k][window_size]['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_k = k
                best_window = window_size
    
    print(f"\nCombination tốt nhất: K={best_k}, Window={best_window}s (F1-Score: {best_f1:.4f})")

if __name__ == "__main__":
    main()
