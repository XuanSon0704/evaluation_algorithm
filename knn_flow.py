import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import time
import warnings
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import gc

warnings.filterwarnings('ignore')


class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train_ = None
        self.y_train_ = None
        self.knn_model = None
        
    def fit(self, X_train_scaled, y_train):
        self.X_train_ = X_train_scaled
        self.y_train_ = y_train
        n_samples_train = self.X_train_.shape[0]
        
        if n_samples_train <= self.k:
            return False
            
        k_actual = min(self.k, n_samples_train - 1)
        
        self.knn_model = KNeighborsClassifier(
            n_neighbors=k_actual,
            weights='distance',
            algorithm='auto',
            n_jobs=1
        )
        
        try:
            self.knn_model.fit(self.X_train_, self.y_train_)
            return True
        except:
            return False
        
    def predict(self, X_test_scaled):
        if self.knn_model is None:
            return np.zeros(X_test_scaled.shape[0])
        
        try:
            predictions = self.knn_model.predict(X_test_scaled)
            return predictions
        except:
            return np.zeros(X_test_scaled.shape[0])

    def predict_proba(self, X_test_scaled):
        if self.knn_model is None:
            return np.zeros((X_test_scaled.shape[0], 2))
        
        try:
            probabilities = self.knn_model.predict_proba(X_test_scaled)
            return probabilities
        except:
            return np.zeros((X_test_scaled.shape[0], 2))

def process_single_configuration_fixed(args):
  
    k, window_size, data_file_path, selected_features, attack_label, sample_ratio = args
    
    process_id = os.getpid()
    print(f"Process {process_id}: Bắt đầu k={k}, ws={window_size}")
    start_time = time.time()
    
    try:
        # Load dữ liệu
        df = pd.read_csv(data_file_path)
        original_size = len(df)
        
        # Chuẩn hóa tên cột
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Sampling nếu cần
        if sample_ratio < 1.0:
            df_benign = df[df['Label'] == 'BENIGN'].sample(frac=sample_ratio, random_state=42)
            df_attack = df[df['Label'] != 'BENIGN'].sample(frac=sample_ratio, random_state=42)
            df = pd.concat([df_benign, df_attack], ignore_index=True)
            print(f"Process {process_id}: Sampled từ {original_size} xuống {len(df)} flows")

        # Kiểm tra các cột cần thiết
        missing_cols = [col for col in selected_features + ['Label'] if col not in df.columns]
        if missing_cols:
            print(f"Process {process_id}: Missing columns: {missing_cols}")
            return None
        
        # Xử lý dữ liệu
        for col in selected_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=selected_features + ['Label'])
        
        if len(df) == 0:
            print(f"Process {process_id}: Không còn dữ liệu sau preprocessing")
            return None
        
        # Log transformation
        for col in selected_features:
            df[col] = np.log1p(df[col].clip(lower=0))

        # Giữ nguyên thứ tự dataset gốc
        print(f"Process {process_id}: Dataset shape: {df.shape}")
        
        # Phân tích dataset
        benign_count = len(df[df['Label'] == 'BENIGN'])
        attack_count = len(df[df['Label'] != 'BENIGN'])
        print(f"Process {process_id}: BENIGN: {benign_count}, ATTACK: {attack_count}")
        
       
        # Kiểm tra xem có đủ BENIGN ở đầu không
        first_benign_flows = df.head(window_size * 2)  # Lấy 2x để đảm bảo
        benign_in_first = len(first_benign_flows[first_benign_flows['Label'] == 'BENIGN'])
        
        if benign_in_first < window_size:
            print(f"Process {process_id}: Không đủ BENIGN trong {window_size*2} flows đầu (chỉ có {benign_in_first})")
            return None
        
        # Lấy window_size flows BENIGN đầu tiên để tính toán
        benign_mask = df['Label'] == 'BENIGN'
        benign_indices = df[benign_mask].head(window_size).index
        training_window_df = df.loc[benign_indices].copy()
        
        print(f"Process {process_id}: Training window: {len(training_window_df)} BENIGN flows")
        
        # Label encoding
        le = LabelEncoder()
        le.fit(df['Label'].unique())
        benign_class_code = le.transform(['BENIGN'])[0]
        attack_class_code = le.transform([attack_label])[0] if attack_label in le.classes_ else 1
        
        # SLIDING WINDOW TRÊN TOÀN BỘ DATASET 
        training_window = deque(training_window_df.to_dict('records'), maxlen=window_size)
        
        y_true = []
        y_pred = []
        
        # Bắt đầu test từ flow thứ window_size
        start_idx = window_size
        total_test_flows = len(df) - start_idx
        
        print(f"Process {process_id}: Bắt đầu test từ flow {start_idx}, tổng {total_test_flows} flows")
        
        # Batch processing để tối ưu
        batch_size = min(1000, max(100, total_test_flows // 20))
        
        for batch_start in range(start_idx, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            
            if (batch_start - start_idx) % 5000 == 0:
                elapsed = time.time() - start_time
                progress = ((batch_start - start_idx) / total_test_flows) * 100
                print(f"Process {process_id}: k={k}, ws={window_size}: {progress:.1f}% ({elapsed:.1f}s)")
            
            # tính toán theo batch
            current_train_df = pd.DataFrame(list(training_window))
            X_train = current_train_df[selected_features].values
            
            # Tạo labels cho training data (tất cả BENIGN)
            y_train = np.full(len(current_train_df), benign_class_code)
            
            # Standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train KNN model
            model = KNN(k=k)
            model_fitted = model.fit(X_train_scaled, y_train)
            
            # Test trên batch
            for idx in range(batch_start, batch_end):
                flow = df.iloc[idx]
                
                # Prediction
                if model_fitted:
                    X_current = np.array([[flow[col] for col in selected_features]])
                    X_current_scaled = scaler.transform(X_current)
                    
                    # Sử dụng KNN để dự đoán
                    try:
                        # Tính khoảng cách đến k neighbors gần nhất
                        distances, indices = model.knn_model.kneighbors(X_current_scaled, n_neighbors=min(k, len(X_train_scaled)))
                        avg_distance = np.mean(distances[0])
                        
                        # Sử dụng threshold dựa trên khoảng cách trung bình của training data
                        # Tính khoảng cách trung bình trong training set
                        train_distances, _ = model.knn_model.kneighbors(X_train_scaled, n_neighbors=min(k, len(X_train_scaled)))
                        train_avg_distance = np.mean(train_distances)
                        threshold = train_avg_distance * 1.5  # Có thể điều chỉnh 
                        
                        prediction = attack_class_code if avg_distance > threshold else benign_class_code
                    except:
                        prediction = benign_class_code
                else:
                    prediction = benign_class_code
                
                # Ground truth
                true_label = le.transform([flow['Label']])[0]
                y_true.append(true_label)
                y_pred.append(prediction)
                
                
                # Chỉ thêm BENIGN flows vào training window
                if flow['Label'] == 'BENIGN':
                    training_window.append(flow.to_dict())
            
            # Memory management
            if (batch_start - start_idx) % 10000 == 0:
                gc.collect()

        # Tính toán metrics
        if len(set(y_true)) > 1:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, pos_label=attack_class_code, zero_division=0)
            recall = recall_score(y_true, y_pred, pos_label=attack_class_code, zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label=attack_class_code, zero_division=0)
        else:
            accuracy = precision = recall = f1 = 0.0
        
       
        attack_count_test = sum(1 for label in y_true if label == attack_class_code)
        benign_count_test = len(y_true) - attack_count_test
        true_positives = sum(1 for t, p in zip(y_true, y_pred) 
                           if t == attack_class_code and p == attack_class_code)
        false_positives = sum(1 for t, p in zip(y_true, y_pred) 
                            if t == benign_class_code and p == attack_class_code)
        
        elapsed_time = time.time() - start_time
        result = {
            'k': k,
            'window_size': window_size,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_test_flows': len(y_true),
            'attack_flows': attack_count_test,
            'benign_flows': benign_count_test,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'processing_time': elapsed_time
        }
        
        print(f"Process {process_id}: Hoàn thành k={k}, ws={window_size}: " + 
              f"F1={f1:.4f}, Acc={accuracy:.4f} " +
              f"({true_positives}/{attack_count_test} attacks detected, {elapsed_time:.1f}s)")
        
        return result
        
    except Exception as e:
        import traceback
        print(f"Process {process_id}: Error k={k}, ws={window_size}: {str(e)}")
        print(f"Process {process_id}: Traceback: {traceback.format_exc()}")
        return None
    finally:
        gc.collect()

# Vẽ biểu đồ kết quả
def plot_results_corrected(results_df, attack_label):
    """Vẽ biểu đồ kết quả với format chuẩn"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'KNN Performance Analysis - {attack_label} Detection', 
                 fontsize=14, fontweight='bold')
    
    k_values = sorted(results_df['k'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(k_values)))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy vs Window Size', 'Precision vs Window Size', 
              'Recall vs Window Size', 'F1-Score vs Window Size']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        for i, k in enumerate(k_values):
            k_data = results_df[results_df['k'] == k].sort_values('window_size')
            if len(k_data) > 0:
                ax.plot(k_data['window_size'], k_data[metric], 
                       color=colors[i], linewidth=2, marker='o', 
                       markersize=5, label=f'K={k}', alpha=0.8)
        
        ax.set_xlabel('Window Size (flows)', fontsize=11)
        ax.set_ylabel(f'{metric.title()}', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(f'KNN_{attack_label}_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Biểu đồ detection details
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle(f'Detection Analysis - {attack_label}', fontsize=14, fontweight='bold')
    
    # Detection rate
    for i, k in enumerate(k_values):
        k_data = results_df[results_df['k'] == k].sort_values('window_size')
        if len(k_data) > 0:
            detection_rate = k_data['true_positives'] / k_data['attack_flows']
            ax1.plot(k_data['window_size'], detection_rate, 
                    color=colors[i], linewidth=2, marker='s', 
                    markersize=4, label=f'K={k}')
    
    ax1.set_xlabel('Window Size', fontsize=11)
    ax1.set_ylabel('Detection Rate (TP/Total Attacks)', fontsize=11)
    ax1.set_title('Attack Detection Rate', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # False positive rate
    for i, k in enumerate(k_values):
        k_data = results_df[results_df['k'] == k].sort_values('window_size')
        if len(k_data) > 0:
            fpr = k_data['false_positives'] / k_data['benign_flows']
            ax2.plot(k_data['window_size'], fpr, 
                    color=colors[i], linewidth=2, marker='^', 
                    markersize=4, label=f'K={k}')
    
    ax2.set_xlabel('Window Size', fontsize=11)
    ax2.set_ylabel('False Positive Rate (FP/Total Benign)', fontsize=11)
    ax2.set_title('False Positive Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'KNN_{attack_label}_detection_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Tham số
    WINDOW_SIZES = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    K_NEIGHBORS_LIST = [2, 4, 6, 8, 10, 12, 14, 16]
    
    selected_features = [
        'Flow_Packets/s', 'Flow_Duration', 'Packet_Length_Mean',
        'Flow_Bytes/s', 'Flow_IAT_Mean'
    ]
    
    ATTACK_LABEL = 'Portmap'
    data_file_path = '/home/xson/LOF-LoOP/Portmap_Cut.csv'
    sample_ratio = 1.0
    
    max_workers = min(mp.cpu_count() - 1, 8)
    
    # Tạo tất cả cấu hình
    configurations = []
    for k in K_NEIGHBORS_LIST:
        for ws in WINDOW_SIZES:
            configurations.append((k, ws, data_file_path, selected_features, ATTACK_LABEL, sample_ratio))
    
    # Chạy song song
    results = []
    start_time = time.time()
    
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_config = {executor.submit(process_single_configuration_fixed, config): config 
                          for config in configurations}
        
        completed = 0
        for future in as_completed(future_to_config):
            try:
                result = future.result(timeout=3600)
                if result is not None:
                    results.append(result)
                completed += 1
                progress = (completed / len(configurations)) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (len(configurations) - completed) if completed > 0 else 0
                print(f"TIẾN TRÌNH: {completed}/{len(configurations)} ({progress:.1f}%) - ETA: {eta/60:.1f} phút")
            except Exception as exc:
                config = future_to_config[future]
                print(f"Cấu hình {config[:2]} thất bại: {exc}")
                completed += 1
    
    total_time = time.time() - start_time
    print(f"\n=== HOÀN THÀNH TRONG {total_time/60:.2f} PHÚT ===")
    
    if not results:
        print("Không có kết quả nào!")
        return
    
    # Phân tích kết quả
    results_df = pd.DataFrame(results)
    
    print(f"\nĐã thu thập {len(results)} kết quả từ {len(configurations)} cấu hình")
    print("\n=== TOP 10 CẤU HÌNH TỐT NHẤT (theo F1-Score) ===")
    top_results = results_df.nlargest(10, 'f1')
    print(top_results[['k', 'window_size', 'accuracy', 'precision', 'recall', 'f1', 
                      'true_positives', 'attack_flows', 'processing_time']].round(4))
    
    # Thống kê
    print(f"\nF1-Score cao nhất: {results_df['f1'].max():.4f}")
    print(f"F1-Score trung bình: {results_df['f1'].mean():.4f}")
    print(f"Thời gian xử lý trung bình: {results_df['processing_time'].mean():.2f}s")
    
    # Lưu kết quả
    results_df.to_csv(f'KNN_{ATTACK_LABEL}_results.csv', index=False)
    print(f"\nĐã lưu kết quả vào 'KNN_{ATTACK_LABEL}_results.csv'")
    
    # Vẽ biểu đồ
    plot_results_corrected(results_df, ATTACK_LABEL)
    

if __name__ == "__main__":
    main()