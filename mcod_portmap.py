import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import warnings
from collections import deque
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import gc

warnings.filterwarnings('ignore')


class MicroCluster:
    #Cấu trúc micro-cluster cho MCOD
    def __init__(self, center, radius, creation_time):
        self.center = np.array(center)
        self.radius = radius
        self.creation_time = creation_time
        self.last_update_time = creation_time
        
        self.points = [(np.array(center), creation_time)]  
        
        self.weight = len(self.points)
       
    def add_point(self, point, timestamp):
        # Thêm 1 điểm mới vào micro-cluster
        self.points.append((np.array(point), timestamp))
        self.weight = len(self.points)
        self.last_update_time = timestamp
        
        self._update_center_radius()
    
    def _update_center_radius(self):
        if len(self.points) >0:
            points_array = np.array([p for p, _ in self.points])    
            self.center = np.mean(points_array, axis=0)
            if len(self.points) > 1:
                dists = np.linalg.norm(points_array - self.center, axis=1)
                self.radius = np.max(dists)
            else:
                self.radius = 0.1  # bán kính tối thiểu
                  
    def remove_old_points(self, current_time, window_size):
        cutoff = current_time - window_size
        
        #Loại bỏ các điểm cũ hơn cutoff
        self.points = [(p, t) for (p, t) in self.points if t > cutoff]
        if not self.points:
            return True # cluster rồng -> xóa
        
        # update thuộc tính mới
        self.weight = len(self.points)
        self._update_center_radius()
        self.creation_time = min(t for _, t in self.points)
        self.last_update_time = max(t for _, t in self.points)
        
        return False
    
    
    def distance_to_center(self, point):
        #Tính khoảng cách từ điểm hiện tại đến center của micro-cluster
        return np.linalg.norm(np.array(point) - self.center)
    
    def can_absorb(self, point):
        #Kiểm tra xem điểm có nằm trong micro-cluster không
        return self.distance_to_center(point) <= self.radius


class MCOD:
    def __init__(self, radius=0.5, min_points=5, window_size=1000):
        """
        Tham số:
        - radius(bán kính): bán kính tối đa để một điểm được xem là thuộc về micro-cluster
        - min_points: số điểm tối thiểu cần thiết để tạo thành 1 cụm
        - window_size: Sliding window size 
        """
        self.radius = radius
        self.min_points = min_points
        self.window_size = window_size
        self.micro_clusters = []
        self.timestamp = 0
        
    def fit(self, X_train):
        """Initialize MCOD with training data"""
        self.micro_clusters = []
        self.timestamp = 0
        
        # Process each training point
        for i, point in enumerate(X_train):
            self._process_point(point, i)
        
        return len(self.micro_clusters) > 0
    
    
    def _process_point(self, point, timestamp):
        """Process a single point using MCOD algorithm"""
        self.timestamp = timestamp
        
        self._maintain_window()
        
        # Find nearest micro-cluster
        nearest_cluster = None
        min_distance = float('inf')
        
        for cluster in self.micro_clusters:
            distance = cluster.distance_to_center(point)
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = cluster
        
        # Check if point can be absorbed by existing cluster
        if nearest_cluster and nearest_cluster.can_absorb(point) :
            nearest_cluster.add_point(point, timestamp)
        else:
            # Create new micro-cluster or mark as outlier
            new_cluster = MicroCluster(point, self.radius, timestamp)
            self.micro_clusters.append(new_cluster)
            
    def _maintain_window(self):
        
        current = self.timestamp
        new_clusters = []
        for cluster in self.micro_clusters:
            empty = cluster.remove_old_points(current, self.window_size)
            if not empty:
                new_clusters.append(cluster)
        self.micro_clusters = new_clusters
    
    
    def decision_function(self, X_test):
        if not self.micro_clusters:
            return np.zeros(X_test.shape[0])
        
        outlier_scores = np.zeros(X_test.shape[0])
        
        for i, point in enumerate(X_test):
            min_distance = float('inf')
            nearest_cluster_radius = None
            
            # Find distance to nearest micro-cluster
            for cluster in self.micro_clusters:
                distance = cluster.distance_to_center(point)
                if distance < min_distance:
                    min_distance = distance
                    nearest_cluster_radius = max(cluster.radius, 0.1)

            # chuẩn hóa distance(khoảng cách) by cluster radius
            if nearest_cluster_radius is not None:
        
                score = min_distance / nearest_cluster_radius
                outlier_score = 1.0 / (1.0 + np.exp(-score + 1))
                outlier_scores[i] = outlier_score
            else:
                outlier_scores[i] = 1.0  # Max outlier score
            
        return outlier_scores
    
    def predict(self, X_test, threshold=1.0):
        score = self.decision_function(X_test)
        return (score > threshold).astype(int)
    

def process_single_configuration_fixed(args):
    """Xử lý một cấu hình MCOD (radius, window_size) cho dataset tĩnh"""
    radius, window_size, data_file_path, selected_features, attack_label, sample_ratio = args
    
    process_id = os.getpid()
    print(f"Process {process_id}: Bắt đầu radius={radius:.2f}, ws={window_size}")
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
        
         # Tạo mask cho BENIGN
        benign_mask = df['Label'] == 'BENIGN'

       # Lấy tất cả index của BENIGN
        benign_indices = df[benign_mask].index

        # Kiểm tra có đủ số lượng BENIGN để tạo window hay không
        if len(benign_indices) < window_size:
            print(f"Process {process_id}: Không đủ BENIGN trong toàn bộ dataset (chỉ có {len(benign_indices)})")
            return None

       # Lấy ngẫu nhiên window_size BENIGN từ toàn dataset
        training_window_df = df.loc[benign_indices].sample(window_size, random_state=42).copy()
        
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
        threshold = 0.5  # MCOD threshold
        
        for batch_start in range(start_idx, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            
            if (batch_start - start_idx) % 5000 == 0:
                elapsed = time.time() - start_time
                progress = ((batch_start - start_idx) / total_test_flows) * 100
                print(f"Process {process_id}: radius={radius:.2f}, ws={window_size}: {progress:.1f}% ({elapsed:.1f}s)")
            
            # Tính toán theo batch
            current_train_df = pd.DataFrame(list(training_window))
            X_train = current_train_df[selected_features].values
            
            # Standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train MCOD model
            model = MCOD(radius=radius, min_points=5, window_size=window_size)
            model_fitted = model.fit(X_train_scaled)
            
            # Tính toán threshold từ training data
            if model_fitted:
                train_scores = model.decision_function(X_train_scaled)
                # Sử dụng percentile để xác định threshold
                threshold = np.percentile(train_scores, 75)  # Top 25% là outliers
            
            # Test trên batch
            for idx in range(batch_start, batch_end):
                flow = df.iloc[idx]
                
                # Prediction
                if model_fitted:
                    X_current = np.array([[flow[col] for col in selected_features]])
                    X_current_scaled = scaler.transform(X_current)
                    current_score = model.decision_function(X_current_scaled)[0]
                    prediction = attack_class_code if current_score > threshold else benign_class_code
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
            'radius': radius,
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
        
        print(f"Process {process_id}: Hoàn thành radius={radius:.2f}, ws={window_size}: " + 
              f"F1={f1:.4f}, Acc={accuracy:.4f} " +
              f"({true_positives}/{attack_count_test} attacks detected, {elapsed_time:.1f}s)")
        
        return result
        
    except Exception as e:
        import traceback
        print(f"Process {process_id}: Error radius={radius:.2f}, ws={window_size}: {str(e)}")
        print(f"Process {process_id}: Traceback: {traceback.format_exc()}")
        return None
    finally:
        gc.collect()

# Vẽ biểu đồ kết quả
def plot_results_corrected(results_df, attack_label):
   
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'MCOD Performance Analysis - {attack_label} Detection', 
                 fontsize=14, fontweight='bold')
    
    radius_values = sorted(results_df['radius'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(radius_values)))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy vs Window Size', 'Precision vs Window Size', 
              'Recall vs Window Size', 'F1-Score vs Window Size']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        for i, radius in enumerate(radius_values):
            radius_data = results_df[results_df['radius'] == radius].sort_values('window_size')
            if len(radius_data) > 0:
                ax.plot(radius_data['window_size'], radius_data[metric], 
                       color=colors[i], linewidth=2, marker='o', 
                       markersize=5, label=f'R={radius:.2f}', alpha=0.8)
        
        ax.set_xlabel('Window Size (flows)', fontsize=11)
        ax.set_ylabel(f'{metric.title()}', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(f'MCOD_{attack_label}_performance.png', dpi=300, bbox_inches='tight')
    
    # Biểu đồ detection details
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle(f'Detection Analysis - {attack_label} (MCOD)', fontsize=14, fontweight='bold')
    
    # Detection rate
    for i, radius in enumerate(radius_values):
        radius_data = results_df[results_df['radius'] == radius].sort_values('window_size')
        if len(radius_data) > 0:
            detection_rate = radius_data['true_positives'] / radius_data['attack_flows']
            ax1.plot(radius_data['window_size'], detection_rate, 
                    color=colors[i], linewidth=2, marker='s', 
                    markersize=4, label=f'R={radius:.2f}')
    
    ax1.set_xlabel('Window Size', fontsize=11)
    ax1.set_ylabel('Detection Rate (TP/Total Attacks)', fontsize=11)
    ax1.set_title('Attack Detection Rate', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # False positive rate
    for i, radius in enumerate(radius_values):
        radius_data = results_df[results_df['radius'] == radius].sort_values('window_size')
        if len(radius_data) > 0:
            fpr = radius_data['false_positives'] / radius_data['benign_flows']
            ax2.plot(radius_data['window_size'], fpr, 
                    color=colors[i], linewidth=2, marker='^', 
                    markersize=4, label=f'R={radius:.2f}')
    
    ax2.set_xlabel('Window Size', fontsize=11)
    ax2.set_ylabel('False Positive Rate (FP/Total Benign)', fontsize=11)
    ax2.set_title('False Positive Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'MCOD_{attack_label}_detection_analysis.png', dpi=300, bbox_inches='tight')

def main():
    # Tham số cấu hình cho MCOD
    WINDOW_SIZES = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    RADIUS_LIST = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5]  # Micro-cluster radius 
    
    selected_features = [
        'Flow_Packets/s', 'Flow_Duration', 'Packet_Length_Mean',
        'Flow_Bytes/s', 'Flow_IAT_Mean'
    ]
    
    ATTACK_LABEL = 'Portmap'
    data_file_path = '/home/xson/evaluation_algorithm/portmap.csv'
    sample_ratio = 1.0
    
    max_workers = min(mp.cpu_count() - 1, 8)
    
    # Tạo tất cả cấu hình
    configurations = []
    for radius in RADIUS_LIST:
        for ws in WINDOW_SIZES:
            configurations.append((radius, ws, data_file_path, selected_features, ATTACK_LABEL, sample_ratio))
    
    # Chạy song song
    results = []
    start_time = time.time()
    
    print(f"Radius values: {RADIUS_LIST}")
    print(f"Window sizes: {WINDOW_SIZES}")
    print(f"Tổng cộng {len(configurations)} cấu hình")
    
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
    
    plot_results_corrected(results_df, ATTACK_LABEL)
    
    print("=== HOÀN THÀNH ===")

if __name__ == "__main__":
    main()