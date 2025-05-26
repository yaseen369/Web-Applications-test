import requests
import time
import sqlite3
import psutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime, timedelta
import threading
import seaborn as sns
import queue
from collections import defaultdict
import statistics
import os

# ====== Configuration ======
TEST_CONFIG = {
    "load_test": {
        "concurrent_users": [1, 5, 10, 20],
        "duration": 60,  # seconds
        "ramp_up_time": 10
    },
    "rate_limit_test": {
        "burst_size": 250,
        "burst_interval": 1  # seconds between bursts
    },
    "stability_test": {
        "duration": 300,  # 5 minutes
        "interval": 2  # seconds between requests
    },
    "failure_injection": {
        "endpoints": ["/not-found", "/timeout", "/error"],
        "invalid_hosts": ["http://invalid.local", "http://timeout.test"]
    }
}

# ====== Paths ======
DB_PATH = "test_results.db"
PDF_PATH = "test_report.pdf"
CSV_PATH = "test_results.csv"
SUMMARY_CSV = "summary_by_test_type.csv"
ENHANCED_SUMMARY_CSV = "enhanced_summary.csv"
ENHANCED_PDF_PATH = "enhanced_test_report.pdf"
SYSTEM_METRICS_CSV = "system_metrics.csv"

# Global variables for resource monitoring
system_metrics_queue = queue.Queue()
monitoring_active = False
baseline_metrics = {"cpu": 0, "ram": 0}

# ====== URL Validation and Normalization ======
def normalize_url(url):
    """Normalize URL by adding scheme if missing"""
    url = url.strip()
    
    # Check if URL already has a scheme
    if url.startswith(('http://', 'https://')):
        return url
    
    # Add https:// by default for security
    return f'https://{url}'

def validate_url(url):
    """Validate URL format"""
    try:
        from urllib.parse import urlparse
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# ====== Database Setup ======
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Drop existing tables if they exist to ensure correct schema
    c.execute('DROP TABLE IF EXISTS results')
    c.execute('DROP TABLE IF EXISTS system_metrics')
    
    # Main results table
    c.execute('''CREATE TABLE results
                 (timestamp TEXT, name TEXT, url TEXT, test_type TEXT,
                  status_code INTEGER, response_time REAL, error_type TEXT, 
                  bytes_received INTEGER, request_id TEXT)''')
    
    # System metrics table
    c.execute('''CREATE TABLE system_metrics
                 (timestamp TEXT, cpu_percent REAL, ram_percent REAL, 
                  disk_io_read REAL, disk_io_write REAL, network_sent REAL, network_recv REAL)''')
    
    conn.commit()
    conn.close()

# ====== Resource Monitoring ======
def collect_baseline_metrics():
    """Collect baseline system metrics before testing"""
    global baseline_metrics
    print("📊 Collecting baseline system metrics...")
    
    cpu_samples = []
    ram_samples = []
    
    for _ in range(10):
        cpu_samples.append(psutil.cpu_percent(interval=0.1))
        ram_samples.append(psutil.virtual_memory().percent)
        time.sleep(0.1)
    
    baseline_metrics = {
        "cpu": statistics.mean(cpu_samples),
        "ram": statistics.mean(ram_samples)
    }
    
    print(f"📊 Baseline - CPU: {baseline_metrics['cpu']:.1f}%, RAM: {baseline_metrics['ram']:.1f}%")

def monitor_system_resources():
    """Background thread to monitor system resources"""
    global monitoring_active
    
    # Get initial network/disk stats
    net_io_start = psutil.net_io_counters()
    disk_io_start = psutil.disk_io_counters()
    
    while monitoring_active:
        try:
            # CPU and RAM
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram_percent = psutil.virtual_memory().percent
            
            # Network and Disk I/O
            net_io = psutil.net_io_counters()
            disk_io = psutil.disk_io_counters()
            
            net_sent = net_io.bytes_sent - net_io_start.bytes_sent if net_io_start else 0
            net_recv = net_io.bytes_recv - net_io_start.bytes_recv if net_io_start else 0
            disk_read = disk_io.read_bytes - disk_io_start.read_bytes if disk_io_start else 0
            disk_write = disk_io.write_bytes - disk_io_start.write_bytes if disk_io_start else 0
            
            # Store metrics
            system_metrics_queue.put({
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'ram_percent': ram_percent,
                'disk_io_read': disk_read,
                'disk_io_write': disk_write,
                'network_sent': net_sent,
                'network_recv': net_recv
            })
            
            time.sleep(0.5)  # Sample every 500ms
            
        except Exception as e:
            print(f"⚠️ Error monitoring system resources: {e}")
            time.sleep(1)

def save_system_metrics():
    """Save collected system metrics to database"""
    if system_metrics_queue.empty():
        return
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    while not system_metrics_queue.empty():
        try:
            metrics = system_metrics_queue.get_nowait()
            c.execute("""INSERT INTO system_metrics VALUES 
                        (?, ?, ?, ?, ?, ?, ?)""",
                     (metrics['timestamp'], metrics['cpu_percent'], metrics['ram_percent'],
                      metrics['disk_io_read'], metrics['disk_io_write'], 
                      metrics['network_sent'], metrics['network_recv']))
        except queue.Empty:
            break
        except Exception as e:
            print(f"⚠️ Error saving system metrics: {e}")
    
    conn.commit()
    conn.close()

# ====== Result Logging ======
def log_result(name, url, test_type, status, duration, error_type="", bytes_received=0, request_id=""):
    """Log test result to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (datetime.now().isoformat(), name, url, test_type, status, duration, 
               error_type, bytes_received, request_id))
    conn.commit()
    conn.close()

# ====== Request Execution ======
def make_request(target, test_type="General", request_id="", timeout=10):
    """Make a single HTTP request with detailed error handling"""
    name, url = target["name"], target["url"]
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        response_time = time.time() - start_time
        
        # Categorize response
        if response.status_code == 200:
            error_type = ""
        elif response.status_code == 429:
            error_type = "rate_limited"
        elif 400 <= response.status_code < 500:
            error_type = "client_error"
        elif 500 <= response.status_code < 600:
            error_type = "server_error"
        else:
            error_type = "unknown_http_error"
        
        bytes_received = len(response.content) if response.content else 0
        
        log_result(name, url, test_type, response.status_code, response_time, 
                  error_type, bytes_received, request_id)
        
        return {
            'success': response.status_code == 200,
            'status_code': response.status_code,
            'response_time': response_time,
            'error_type': error_type,
            'bytes_received': bytes_received
        }
        
    except requests.exceptions.Timeout:
        # Record actual timeout duration, not the timeout limit
        response_time = time.time() - start_time if 'start_time' in locals() else timeout
        log_result(name, url, test_type, 0, response_time, "timeout", 0, request_id)
        return {'success': False, 'status_code': 0, 'response_time': response_time, 
                'error_type': 'timeout', 'bytes_received': 0}
        
    except requests.exceptions.ConnectionError:
        response_time = time.time() - start_time if 'start_time' in locals() else -1
        log_result(name, url, test_type, 0, response_time, "connection_error", 0, request_id)
        return {'success': False, 'status_code': 0, 'response_time': response_time, 
                'error_type': 'connection_error', 'bytes_received': 0}
        
    except Exception as e:
        response_time = time.time() - start_time if 'start_time' in locals() else -1
        error_type = f"exception_{type(e).__name__}"
        log_result(name, url, test_type, 0, response_time, error_type, 0, request_id)
        return {'success': False, 'status_code': 0, 'response_time': response_time, 
                'error_type': error_type, 'bytes_received': 0}

# ====== Safe Statistics Functions ======
def safe_mean(values):
    """Calculate mean safely, returning 0 if no values"""
    valid_values = [v for v in values if v is not None and v >= 0]
    return statistics.mean(valid_values) if valid_values else 0

def safe_median(values):
    """Calculate median safely, returning 0 if no values"""
    valid_values = [v for v in values if v is not None and v >= 0]
    return statistics.median(valid_values) if valid_values else 0

def safe_quantile(values, q):
    """Calculate quantile safely, returning 0 if no values"""
    # Handle empty values
    if values is None:
        return 0
    
    # Convert to pandas Series if not already
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    
    # Check if empty using pandas method
    if values.empty:
        return 0
    
    try:
        return values.quantile(q)
    except:
        return 0

def calculate_success_rate(test_df, test_type):
    """Calculate success rate based on test type context"""
    if len(test_df) == 0:
        return 0, 0, 0  # success_count, total_count, success_rate
    
    total_count = len(test_df)
    
    if test_type == "Failure Injection":
        # For failure injection, success means detecting failures (non-200 responses)
        success_count = len(test_df[test_df['status_code'] != 200])
        success_rate = (success_count / total_count) * 100
    else:
        # For other tests, success means getting 200 responses
        success_count = len(test_df[test_df['status_code'] == 200])
        success_rate = (success_count / total_count) * 100
    
    return success_count, total_count, success_rate

# ====== Test Implementations ======
def load_test(target, config=None):
    """Improved load test with true concurrency"""
    if config is None:
        config = TEST_CONFIG["load_test"]
    
    print(f"🚀 Starting Load Test for {target['url']}")
    
    for user_count in config["concurrent_users"]:
        print(f"📈 Testing with {user_count} concurrent users...")
        
        # Results collection
        results = []
        results_lock = threading.Lock()
        test_duration = config["duration"]
        
        def worker(worker_id):
            start_time = time.time()
            request_count = 0
            
            while time.time() - start_time < test_duration:
                request_id = f"load_{user_count}_{worker_id}_{request_count}"
                result = make_request(target, f"Load Test ({user_count} users)", request_id)
                
                with results_lock:
                    results.append(result)
                
                request_count += 1
                time.sleep(0.1)  # Small delay between requests per user
        
        # Start worker threads
        threads = []
        for i in range(user_count):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Print summary for this load level
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        if total > 0:
            successful_times = [r['response_time'] for r in results if r['success'] and r['response_time'] > 0]
            avg_response_time = safe_mean(successful_times)
            
            print(f"   📊 {user_count} users: {successful}/{total} successful ({100*successful/total:.1f}%), "
                  f"avg response: {avg_response_time:.3f}s")
        else:
            print(f"   📊 {user_count} users: No requests completed")
        
        time.sleep(2)  # Cool down between load levels

def failure_injection_test(target, config=None):
    """Test system's response to various failure scenarios"""
    if config is None:
        config = TEST_CONFIG["failure_injection"]
    
    print(f"🔧 Starting Failure Injection Test for {target['url']}")
    
    failure_scenarios = []
    
    # Add endpoint-based failures
    for endpoint in config["endpoints"]:
        failure_scenarios.append({
            "name": f"Missing Endpoint: {endpoint}",
            "url": target["url"] + endpoint
        })
    
    # Add invalid host failures
    for invalid_host in config["invalid_hosts"]:
        failure_scenarios.append({
            "name": f"Invalid Host: {invalid_host}",
            "url": invalid_host
        })
    
    results = []
    for i, scenario in enumerate(failure_scenarios):
        print(f"   🎯 Testing scenario: {scenario['name']}")
        
        for attempt in range(3):  # Multiple attempts per scenario
            request_id = f"failure_{i}_{attempt}"
            result = make_request(scenario, "Failure Injection", request_id, timeout=5)
            results.append(result)
    
    # Summary - count failures as successes for failure injection
    failures_detected = sum(1 for r in results if not r['success'] or r['status_code'] != 200)
    total = len(results)
    print(f"   📊 Failure detection: {failures_detected}/{total} scenarios failed as expected "
          f"({100*failures_detected/total:.1f}%)")

def rate_limit_test(target, config=None):
    """Test rate limiting behavior"""
    if config is None:
        config = TEST_CONFIG["rate_limit_test"]
    
    print(f"⚡ Starting Rate Limit Test for {target['url']}")
    
    burst_size = config["burst_size"]
    
    print(f"   🌊 Sending burst of {burst_size} requests...")
    
    results = []
    threads = []
    results_lock = threading.Lock()
    
    def burst_worker(worker_id):
        for i in range(burst_size // 10):  # Each worker handles 1/10th of requests
            request_id = f"rate_limit_{worker_id}_{i}"
            result = make_request(target, "Rate Limit Test", request_id, timeout=5)
            
            with results_lock:
                results.append(result)
    
    # Create 10 threads for burst
    for i in range(10):
        t = threading.Thread(target=burst_worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # Analyze results
    successful = sum(1 for r in results if r['success'])
    rate_limited = sum(1 for r in results if r['error_type'] == 'rate_limited' or r['status_code'] == 429)
    total = len(results)
    
    print(f"   📊 Results: {successful} successful, {rate_limited} rate-limited, "
          f"{total - successful - rate_limited} other errors")
    
    if rate_limited > 0:
        print(f"   ✅ Rate limiting detected: {rate_limited} requests were rate-limited")
    else:
        print(f"   ⚠️ No rate limiting detected - server accepted all {burst_size} rapid requests")

def stability_test(target, config=None):
    """Long-running stability test"""
    if config is None:
        config = TEST_CONFIG["stability_test"]
    
    print(f"📈 Starting Stability Test for {target['url']} ({config['duration']}s duration)")
    
    start_time = time.time()
    request_count = 0
    results = []
    
    while time.time() - start_time < config["duration"]:
        request_id = f"stability_{request_count}"
        result = make_request(target, "Stability Test", request_id)
        results.append(result)
        
        request_count += 1
        
        # Progress update every 30 seconds
        elapsed = time.time() - start_time
        if request_count % 15 == 0:  # Every 15 requests (roughly 30 seconds)
            successful = sum(1 for r in results if r['success'])
            print(f"   ⏱️ {elapsed:.0f}s elapsed: {successful}/{request_count} successful "
                  f"({100*successful/request_count:.1f}%)")
        
        time.sleep(config["interval"])
    
    # Final summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    successful_times = [r['response_time'] for r in results if r['response_time'] > 0]
    avg_response_time = safe_mean(successful_times)
    
    print(f"   📊 Final: {successful}/{total} successful ({100*successful/total:.1f}%), "
          f"avg response: {avg_response_time:.3f}s")

# ====== Data Validation ======
def validate_data():
    """Validate collected data for anomalies"""
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("❌ No data file found for validation")
        return
    
    print("\n🔍 Data Validation Results:")
    
    # Check for impossible values
    negative_times = df[df['response_time'] < 0]
    if not negative_times.empty:
        print(f"   ⚠️ Found {len(negative_times)} requests with negative response times")
    
    # Check for potential timeout artifacts (exact timeout values)
    timeout_artifacts = df[df['response_time'].isin([5.0, 10.0, 15.0, 30.0])]
    if not timeout_artifacts.empty:
        print(f"   ⚠️ Found {len(timeout_artifacts)} requests with potential timeout artifacts")
    
    # Check for outliers
    valid_times = df[df['response_time'] > 0]['response_time']
    if not valid_times.empty:
        q99 = safe_quantile(valid_times, 0.99)
        q01 = safe_quantile(valid_times, 0.01)
        if q99 > 0 and q01 > 0:
            # Convert to list for comparison to avoid pandas Series ambiguity
            valid_times_list = valid_times.tolist()
            outliers = [t for t in valid_times_list if t > q99 * 10 or t < q01 / 10]
            if outliers:
                print(f"   ⚠️ Found {len(outliers)} potential response time outliers")
    
    # Check timestamp sequences
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_gaps = df['timestamp'].diff()
    large_gaps = time_gaps[time_gaps > pd.Timedelta(minutes=5)]
    if not large_gaps.empty:
        print(f"   ⚠️ Found {len(large_gaps)} large time gaps (>5 min) in data")
    
    print("   ✅ Data validation complete")

# ====== CSV Export ======
def export_to_csv():
    """Export results and system metrics to CSV"""
    conn = sqlite3.connect(DB_PATH)
    
    # Export test results
    df_results = pd.read_sql_query("SELECT * FROM results", conn)
    df_results.to_csv(CSV_PATH, index=False)
    print(f"📄 Test results saved to: {CSV_PATH}")
    
    # Export system metrics
    df_metrics = pd.read_sql_query("SELECT * FROM system_metrics", conn)
    if not df_metrics.empty:
        df_metrics.to_csv(SYSTEM_METRICS_CSV, index=False)
        print(f"📄 System metrics saved to: {SYSTEM_METRICS_CSV}")
    
    conn.close()

# ====== Enhanced Graph Generation ======
def generate_enhanced_graphs(db_path=DB_PATH, output_path="enhanced_graphs.png"):
    """Generate comprehensive performance visualizations"""
    conn = sqlite3.connect(db_path)
    df_results = pd.read_sql_query("SELECT * FROM results", conn)
    df_metrics = pd.read_sql_query("SELECT * FROM system_metrics", conn)
    conn.close()
    
    if df_results.empty:
        print("⚠️ No test data to visualize.")
        return False
    
    # Convert timestamps
    df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])
    if not df_metrics.empty:
        df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'])
    
    # Filter successful requests for response time analysis
    df_success = df_results[df_results['response_time'] > 0].copy()
    
    # Create color mapping
    test_types = df_results['test_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_types)))
    color_map = dict(zip(test_types, colors))
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Response Time Over Time (top left)
    ax1 = plt.subplot(3, 3, 1)
    if not df_success.empty:
        for test_type in test_types:
            mask = df_success['test_type'] == test_type
            if mask.any():
                ax1.scatter(df_success[mask]['timestamp'], df_success[mask]['response_time'], 
                           label=test_type, alpha=0.7, c=[color_map[test_type]], marker='o', s=20)
    
    ax1.set_title('Response Time Over Time')
    ax1.set_ylabel('Response Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    if not df_success.empty:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax1.text(0.5, 0.5, 'No Successful Responses', ha='center', va='center', transform=ax1.transAxes)
    
    # 2. Success Rate by Test Type (top middle)
    ax2 = plt.subplot(3, 3, 2)
    success_rates = []
    for test_type in test_types:
        test_df = df_results[df_results['test_type'] == test_type]
        _, _, success_rate = calculate_success_rate(test_df, test_type)
        success_rates.append(success_rate)
    
    bars = ax2.bar(range(len(test_types)), success_rates, 
                  color=[color_map[t] for t in test_types])
    ax2.set_title('Success Rate by Test Type')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_xticks(range(len(test_types)))
    ax2.set_xticklabels(test_types, rotation=45, ha='right')
    ax2.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # 3. Error Type Distribution (top right)
    ax3 = plt.subplot(3, 3, 3)
    error_counts = df_results[df_results['error_type'] != '']['error_type'].value_counts()
    if not error_counts.empty:
        wedges, texts, autotexts = ax3.pie(error_counts.values, labels=error_counts.index, 
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Error Type Distribution')
    else:
        ax3.text(0.5, 0.5, 'No Errors Detected', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Error Type Distribution')
    
    # 4. System CPU Usage (middle left)
    ax4 = plt.subplot(3, 3, 4)
    if not df_metrics.empty:
        ax4.plot(df_metrics['timestamp'], df_metrics['cpu_percent'], 'r-', alpha=0.7, linewidth=2)
        ax4.axhline(y=baseline_metrics['cpu'], color='r', linestyle='--', alpha=0.5, label='Baseline')
        ax4.set_title('System CPU Usage')
        ax4.set_ylabel('CPU Usage (%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No System Metrics', ha='center', va='center', transform=ax4.transAxes)
    
    # 5. System RAM Usage (middle center)
    ax5 = plt.subplot(3, 3, 5)
    if not df_metrics.empty:
        ax5.plot(df_metrics['timestamp'], df_metrics['ram_percent'], 'b-', alpha=0.7, linewidth=2)
        ax5.axhline(y=baseline_metrics['ram'], color='b', linestyle='--', alpha=0.5, label='Baseline')
        ax5.set_title('System RAM Usage')
        ax5.set_ylabel('RAM Usage (%)')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'No System Metrics', ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Response Time Distribution (middle right)
    ax6 = plt.subplot(3, 3, 6)
    if not df_success.empty:
        for test_type in test_types:
            test_data = df_success[df_success['test_type'] == test_type]['response_time']
            if not test_data.empty:
                ax6.hist(test_data, alpha=0.5, label=test_type, bins=20, 
                        color=color_map[test_type], density=True)
        ax6.set_title('Response Time Distribution')
        ax6.set_xlabel('Response Time (seconds)')
        ax6.set_ylabel('Density')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No Response Time Data', ha='center', va='center', transform=ax6.transAxes)
    
    # 7. Throughput Over Time (bottom left)
    ax7 = plt.subplot(3, 3, 7)
    if not df_results.empty:
        # Calculate requests per minute
        df_results['minute'] = df_results['timestamp'].dt.floor('min')
        throughput = df_results.groupby('minute').size()
        ax7.plot(throughput.index, throughput.values, 'g-', marker='o', linewidth=2, markersize=4)
        ax7.set_title('Throughput (Requests per Minute)')
        ax7.set_ylabel('Requests/min')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
    
    # 8. Status Code Distribution (bottom center)
    ax8 = plt.subplot(3, 3, 8)
    status_counts = df_results['status_code'].value_counts().sort_index()
    bars = ax8.bar(range(len(status_counts)), status_counts.values)
    ax8.set_title('HTTP Status Code Distribution')
    ax8.set_ylabel('Count')
    ax8.set_xticks(range(len(status_counts)))
    ax8.set_xticklabels(status_counts.index)
    
    # Color bars based on status code type
    for i, (status_code, bar) in enumerate(zip(status_counts.index, bars)):
        if status_code == 200:
            bar.set_color('green')
        elif status_code == 429:
            bar.set_color('orange')
        elif 400 <= status_code < 500:
            bar.set_color('red')
        elif 500 <= status_code < 600:
            bar.set_color('darkred')
        else:
            bar.set_color('gray')
    
    # 9. Performance Summary Table (bottom right)
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')
    
    # Create summary statistics
    summary_data = []
    for test_type in test_types:
        test_df = df_results[df_results['test_type'] == test_type]
        success_df = test_df[test_df['response_time'] > 0]
        
        if not success_df.empty:
            avg_time = safe_mean(success_df['response_time'])
            p95_time = safe_quantile(success_df['response_time'], 0.95)
        else:
            avg_time = p95_time = 0
        
        total_requests = len(test_df)
        _, _, success_rate = calculate_success_rate(test_df, test_type)
        
        summary_data.append([test_type[:15], total_requests, f"{success_rate:.1f}%", 
                           f"{avg_time:.3f}s", f"{p95_time:.3f}s"])
    
    table = ax9.table(cellText=summary_data,
                     colLabels=['Test Type', 'Total', 'Success%', 'Avg Time', 'P95 Time'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax9.set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Enhanced graphs saved to: {output_path}")
    return True

# ====== Enhanced Summary Analysis ======
def enhanced_summary_analysis(csv_path=CSV_PATH, summary_csv=ENHANCED_SUMMARY_CSV):
    """Generate comprehensive summary statistics"""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("❌ CSV file not found. Skipping summary.")
        return None
    
    if df.empty:
        print("⚠️ No data to summarize.")
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_success = df[df['response_time'] > 0].copy()
    
    summary = []
    
    for test_type in df['test_type'].unique():
        test_df = df[df['test_type'] == test_type]
        success_df = df_success[df_success['test_type'] == test_type]
        
        # Basic metrics
        total_requests = len(test_df)
        successful_requests = len(success_df)
        
        # Context-aware success rate using the fixed function
        success_count, _, success_rate = calculate_success_rate(test_df, test_type)
        
        # Determine success definition for display
        if test_type == "Failure Injection":
            success_definition = "Failure Detection Rate"
        else:
            success_definition = "Success Rate"
        
        # Response time statistics - using safe functions
        if not success_df.empty:
            response_times = success_df['response_time']
            avg_response = safe_mean(response_times)
            median_response = safe_median(response_times)
            p90_response = safe_quantile(response_times, 0.90)
            p95_response = safe_quantile(response_times, 0.95)
            p99_response = safe_quantile(response_times, 0.99)
            min_response = response_times.min()
            max_response = response_times.max()
            std_response = response_times.std() if len(response_times) > 1 else 0
        else:
            avg_response = median_response = p90_response = p95_response = p99_response = 0
            min_response = max_response = std_response = 0
        
        # Error analysis
        error_breakdown = test_df['error_type'].value_counts().to_dict()
        
        # Throughput calculation
        if not test_df.empty:
            duration = (test_df['timestamp'].max() - test_df['timestamp'].min()).total_seconds()
            throughput = total_requests / duration if duration > 0 else 0
        else:
            duration = throughput = 0
        
        # Data transfer
        total_bytes = test_df['bytes_received'].sum()
        avg_bytes = test_df['bytes_received'].mean() if total_requests > 0 else 0
        
        summary.append({
            'Test Type': test_type,
            'Total Requests': total_requests,
            'Successful Requests': successful_requests,
            'Success Count': success_count,
            f'{success_definition} (%)': success_rate,
            'Avg Response Time (s)': avg_response,
            'Median Response Time (s)': median_response,
            'P90 Response Time (s)': p90_response,
            'P95 Response Time (s)': p95_response,
            'P99 Response Time (s)': p99_response,
            'Min Response Time (s)': min_response,
            'Max Response Time (s)': max_response,
            'Std Dev Response Time': std_response,
            'Test Duration (s)': duration,
            'Throughput (req/s)': throughput,
            'Total Bytes Received': total_bytes,
            'Avg Bytes per Request': avg_bytes,
            'Error Breakdown': str(error_breakdown)
        })
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.round(3)
    summary_df.to_csv(summary_csv, index=False)
    
    # Display key metrics
    print("\n📊 Enhanced Summary by Test Type:\n")
    display_cols = ['Test Type', 'Total Requests', 'Success Count', 
                   'Avg Response Time (s)', 'P95 Response Time (s)', 'Throughput (req/s)']
    
    # Handle column name variations
    available_cols = [col for col in display_cols if col in summary_df.columns]
    if available_cols:
        print(summary_df[available_cols].to_string(index=False))
    
    print(f"\n📄 Enhanced summary saved to: {summary_csv}")
    return summary_df

# ====== Performance Analysis ======
def analyze_performance_trends(df_results):
    """Analyze performance trends and provide insights"""
    insights = []
    
    for test_type in df_results['test_type'].unique():
        test_df = df_results[df_results['test_type'] == test_type]
        success_df = test_df[test_df['response_time'] > 0]
        
        if success_df.empty:
            insights.append(f"{test_type}: No successful responses recorded")
            continue
        
        # Response time analysis
        response_times = success_df['response_time']
        mean_time = safe_mean(response_times)
        if mean_time > 0:
            cv = response_times.std() / mean_time
            
            if cv > 1.0:
                insights.append(f"{test_type}: High response time variability (CV={cv:.2f})")
            elif cv < 0.2:
                insights.append(f"{test_type}: Consistent response times (CV={cv:.2f})")
        
        # Performance degradation check
        if len(success_df) > 10:
            first_half = success_df.iloc[:len(success_df)//2]['response_time']
            second_half = success_df.iloc[len(success_df)//2:]['response_time']
            
            first_mean = safe_mean(first_half)
            second_mean = safe_mean(second_half)
            
            if first_mean > 0 and second_mean > first_mean * 1.5:
                insights.append(f"{test_type}: Performance degradation detected "
                              f"({first_mean:.3f}s -> {second_mean:.3f}s)")
            elif first_mean > 0 and second_mean < first_mean * 0.8:
                insights.append(f"{test_type}: Performance improvement detected "
                              f"({first_mean:.3f}s -> {second_mean:.3f}s)")
        
        # Context-aware error analysis
        success_count, total_count, success_rate = calculate_success_rate(test_df, test_type)
        
        if test_type == "Failure Injection":
            if success_rate > 80:
                insights.append(f"{test_type}: Excellent failure detection ({success_rate:.1f}%)")
            elif success_rate < 50:
                insights.append(f"{test_type}: Poor failure detection ({success_rate:.1f}%)")
        elif test_type == "Rate Limit Test":
            rate_limited_count = len(test_df[test_df['status_code'] == 429])
            if rate_limited_count > 0:
                insights.append(f"{test_type}: Rate limiting detected ({rate_limited_count} requests limited)")
            else:
                insights.append(f"{test_type}: No rate limiting detected - server accepted all requests")
        else:
            # Regular success rate analysis for other tests
            if success_rate > 95:
                insights.append(f"{test_type}: Excellent reliability ({success_rate:.1f}%)")
            elif success_rate < 90:
                insights.append(f"{test_type}: Reliability concerns ({success_rate:.1f}%)")
    
    return insights

# ====== Enhanced PDF Report Generation ======
def generate_enhanced_report(db_path=DB_PATH, csv_path=CSV_PATH, pdf_path=ENHANCED_PDF_PATH):
    """Generate comprehensive PDF report with analysis"""
    # Generate graphs and summary
    graph_generated = generate_enhanced_graphs(db_path, "enhanced_graphs.png")
    summary_df = enhanced_summary_analysis(csv_path)
    
    if summary_df is None or summary_df.empty:
        print("⚠️ Not enough data to generate a complete report.")
        # Still try to create a basic report
        try:
            df_results = pd.read_csv(csv_path)
        except:
            print("❌ Cannot load test results. Skipping report generation.")
            return
    else:
        # Load test results for analysis
        try:
            df_results = pd.read_csv(csv_path)
            df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])
        except:
            print("⚠️ Error loading test results for analysis.")
            return
    
    # Generate insights
    insights = analyze_performance_trends(df_results) if not df_results.empty else ["No test data available for analysis"]
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 16)
            self.cell(0, 15, 'Performance Testing Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')
    
    pdf = PDF()
    
    # Page 1: Executive Summary
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Executive Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    # Add tested URLs
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(0, 8, "Tested URLs:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", '', 10)
    
    # Get unique URLs from test results
    tested_urls = df_results['url'].unique() if not df_results.empty else ["No URLs tested"]
    for url in tested_urls:
        # Truncate very long URLs to fit on page
        display_url = url if len(str(url)) <= 70 else str(url)[:67] + "..."
        pdf.cell(0, 6, f"- {display_url}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(0, 8, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    # Overall statistics
    if not df_results.empty:
        total_requests = len(df_results)
        total_successful = len(df_results[df_results['response_time'] > 0])
        overall_success_rate = (total_successful / total_requests) * 100 if total_requests > 0 else 0
        
        if total_successful > 0:
            avg_response_time = safe_mean(df_results[df_results['response_time'] > 0]['response_time'])
            p95_response_time = safe_quantile(df_results[df_results['response_time'] > 0]['response_time'], 0.95)
        else:
            avg_response_time = p95_response_time = 0
        
        summary_text = f"""Test Overview:
- Total Requests: {total_requests:,}
- Overall Success Rate: {overall_success_rate:.1f}%
- Average Response Time: {avg_response_time:.3f} seconds
- 95th Percentile Response Time: {p95_response_time:.3f} seconds
- Test Types: {', '.join(df_results['test_type'].unique())}

System Performance:
- Baseline CPU: {baseline_metrics['cpu']:.1f}%
- Baseline RAM: {baseline_metrics['ram']:.1f}%
"""
    else:
        summary_text = """Test Overview:
- No test data available
- Tests may have failed to execute properly
- Check network connectivity and target URL accessibility
"""
    
    pdf.multi_cell(0, 6, summary_text)
    
    # Key insights
    pdf.ln(5)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Key Insights:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", '', 10)
    if insights:
        for insight in insights[:8]:  # Limit to top 8 insights
            # Use multi_cell with proper width management
            try:
                pdf.multi_cell(180, 5, f"- {insight}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            except:
                # Fallback: manually break long lines
                max_chars_per_line = 75
                if len(insight) <= max_chars_per_line:
                    pdf.cell(0, 5, f"- {insight}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                else:
                    # Split into multiple lines
                    words = insight.split(' ')
                    current_line = ""
                    first_line = True
                    
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        
                        if len(test_line) <= max_chars_per_line:
                            current_line = test_line
                        else:
                            # Output current line
                            if current_line:
                                prefix = "- " if first_line else "  "
                                pdf.cell(0, 5, f"{prefix}{current_line}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                                first_line = False
                                current_line = word
                            else:
                                # Single word is too long, just output it
                                prefix = "- " if first_line else "  "
                                pdf.cell(0, 5, f"{prefix}{word}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                                first_line = False
                                current_line = ""
                    
                    # Output remaining text
                    if current_line:
                        prefix = "- " if first_line else "  "
                        pdf.cell(0, 5, f"{prefix}{current_line}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.multi_cell(180, 5, "- No insights available due to lack of test data", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Page 2: Visualizations (only if graphs were generated)
    if graph_generated:
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(0, 10, "Performance Visualizations", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        try:
            pdf.image("enhanced_graphs.png", x=5, y=30, w=200)
        except:
            pdf.set_font("Helvetica", '', 12)
            pdf.cell(0, 10, "Visualization generation failed - insufficient data", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Page 3: Detailed Results (only if we have summary data)
    if summary_df is not None and not summary_df.empty:
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(0, 10, "Detailed Test Results", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)
        
        # Results table
        pdf.set_font("Helvetica", 'B', 9)
        headers = ['Test Type', 'Requests', 'Success Rate', 'Avg Time', 'P95 Time', 'Throughput']
        col_widths = [45, 20, 25, 25, 25, 25]
        
        # Print headers
        for i, header in enumerate(headers):
            if i == len(headers) - 1:
                pdf.cell(col_widths[i], 8, header, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            else:
                pdf.cell(col_widths[i], 8, header, 1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')
        
        # Print data
        pdf.set_font("Helvetica", '', 8)
        for _, row in summary_df.iterrows():
            # Determine success rate column name
            success_col = None
            for col in row.index:
                if 'Rate (%)' in col or 'Detection Rate' in col:
                    success_col = col
                    break
            
            success_rate = row[success_col] if success_col else 0
            
            data_row = [
                str(row['Test Type'])[:20],
                str(int(row['Total Requests'])),
                f"{success_rate:.1f}%",
                f"{row['Avg Response Time (s)']:.3f}",
                f"{row['P95 Response Time (s)']:.3f}",
                f"{row['Throughput (req/s)']:.2f}"
            ]
            
            for i, cell_data in enumerate(data_row):
                if i == len(data_row) - 1:
                    pdf.cell(col_widths[i], 8, cell_data, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
                else:
                    pdf.cell(col_widths[i], 8, cell_data, 1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')
    
    # Page 4: Analysis and Recommendations
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Analysis and Recommendations", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    if not df_results.empty:
        # Performance analysis by test type
        for test_type in df_results['test_type'].unique():
            test_df = df_results[df_results['test_type'] == test_type]
            success_df = test_df[test_df['response_time'] > 0]
            
            pdf.set_font("Helvetica", 'B', 11)
            pdf.cell(0, 8, f"{test_type}:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            pdf.set_font("Helvetica", '', 10)
            
            recommendations = []
            success_count, total_count, success_rate = calculate_success_rate(test_df, test_type)
            
            if test_type.startswith("Load Test"):
                if not success_df.empty:
                    p95_time = safe_quantile(success_df['response_time'], 0.95)
                    if p95_time > 2.0:
                        recommendations.append("Consider performance optimization - P95 response time exceeds 2 seconds")
                    elif p95_time < 0.5:
                        recommendations.append("Excellent response times under load")
                    
                    if success_rate < 95:
                        recommendations.append(f"Reliability concern - only {success_rate:.1f}% requests succeeded")
                else:
                    recommendations.append("Load test failed - no successful responses recorded")
            
            elif test_type == "Rate Limit Test":
                rate_limited = len(test_df[test_df['status_code'] == 429])
                if rate_limited == 0:
                    recommendations.append("No rate limiting detected - consider implementing rate limits for DDoS protection")
                else:
                    recommendations.append(f"Rate limiting working correctly ({rate_limited} requests limited)")
            
            elif test_type == "Failure Injection":
                if success_rate > 80:
                    recommendations.append(f"Excellent error handling - {success_rate:.1f}% of failures properly detected")
                elif success_rate < 50:
                    recommendations.append(f"Poor error handling - only {success_rate:.1f}% of failures detected")
                else:
                    recommendations.append(f"Adequate error handling - {success_rate:.1f}% of failures detected")
            
            elif test_type == "Stability Test":
                if not success_df.empty and len(success_df) > 10:
                    response_times = success_df['response_time']
                    mean_time = safe_mean(response_times)
                    if mean_time > 0:
                        cv = response_times.std() / mean_time
                        if cv > 0.5:
                            recommendations.append("Response time variability indicates stability issues")
                        else:
                            recommendations.append("Stable performance over extended period")
                    
                    if success_rate < 95:
                        recommendations.append(f"Stability concern - only {success_rate:.1f}% requests succeeded over time")
                else:
                    recommendations.append("Stability test failed - insufficient successful responses")
            
            if not recommendations:
                recommendations.append("No specific analysis available - insufficient data")
            
            for rec in recommendations:
                try:
                    pdf.multi_cell(0, 5, f"  - {rec}")
                except:
                    pdf.cell(0, 5, f"  - {rec[:60]}...", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            pdf.ln(3)
    else:
        pdf.set_font("Helvetica", '', 12)
        pdf.multi_cell(0, 6, """No test results available for analysis.

Possible causes:
- Network connectivity issues
- Target server not responding
- Firewall blocking requests
- Invalid URL format

Recommendations:
- Verify target URL is accessible in a web browser
- Check network connection
- Try testing with a known working URL (e.g., http://httpbin.org/get)
- Review error logs for specific connection issues""")
    
    # Save PDF
    try:
        pdf.output(pdf_path)
        print(f"📊 Enhanced PDF report saved to: {pdf_path}")
    except Exception as e:
        print(f"⚠️ Error saving PDF report: {e}")

# ====== Main CLI ======
def main():
    print("🚀 Performance Testing Tool v2.1 - Fixed Edition")
    print("=" * 50)
    
    # Initialize
    init_db()
    collect_baseline_metrics()
    
    # Get targets
    user_input = input("\nEnter target URLs (comma-separated):\n")
    targets = []
    for url in user_input.split(","):
        url = url.strip()
        if url:
            # Normalize URL (add https:// if missing)
            normalized_url = normalize_url(url)
            
            # Validate URL
            if validate_url(normalized_url):
                targets.append({"name": url, "url": normalized_url})
                print(f"✅ Added target: {normalized_url}")
            else:
                print(f"⚠️ Invalid URL format: {url} (skipping)")
    
    if not targets:
        print("❌ No valid URLs entered.")
        return
    
    # Get test selection
    print("\nSelect tests to run:")
    print("1 - Load Test (multiple concurrent user levels)")
    print("2 - Failure Injection Test (test error handling)")
    print("3 - Rate Limit Test (burst testing)")
    print("4 - Stability Test (long-running)")
    print("5 - All Tests")
    print("6 - Custom Configuration")
    
    choice = input("Enter choice (e.g., 1,3,4 or 5 for all): ").split(",")
    choice = [c.strip() for c in choice]
    
    # Custom configuration
    if "6" in choice:
        print("\n⚙️ Custom Configuration:")
        try:
            load_users = input(f"Load test concurrent users ({TEST_CONFIG['load_test']['concurrent_users']}): ")
            if load_users:
                TEST_CONFIG['load_test']['concurrent_users'] = [int(x) for x in load_users.split(",")]
            
            stability_duration = input(f"Stability test duration in seconds ({TEST_CONFIG['stability_test']['duration']}): ")
            if stability_duration:
                TEST_CONFIG['stability_test']['duration'] = int(stability_duration)
        except ValueError:
            print("⚠️ Invalid configuration, using defaults")
    
    # Start system monitoring
    global monitoring_active
    monitoring_active = True
    monitor_thread = threading.Thread(target=monitor_system_resources, daemon=True)
    monitor_thread.start()
    
    try:
        # Run tests
        for target in targets:
            print(f"\n📍 Running tests on {target['url']}")
            print("-" * 40)
            
            if "5" in choice or "1" in choice:
                load_test(target)
                time.sleep(3)  # Cool down between tests
            
            if "5" in choice or "2" in choice:
                failure_injection_test(target)
                time.sleep(3)
            
            if "5" in choice or "3" in choice:
                rate_limit_test(target)
                time.sleep(3)
            
            if "5" in choice or "4" in choice:
                stability_test(target)
                time.sleep(3)
        
        print("\n🔄 Processing results...")
        
        # Stop monitoring and save metrics
        monitoring_active = False
        time.sleep(1)  # Allow final metrics to be collected
        save_system_metrics()
        
        # Export and validate data
        export_to_csv()
        validate_data()
        
        # Generate reports
        print("\n📊 Generating reports...")
        generate_enhanced_report()
        
        print(f"\n🎉 Testing Complete! Results available in:")
        print(f"📁 {DB_PATH} - SQLite database")
        print(f"📁 {CSV_PATH} - Test results")
        print(f"📁 {SYSTEM_METRICS_CSV} - System metrics")
        print(f"📁 {ENHANCED_PDF_PATH} - Comprehensive report")
        if os.path.exists(ENHANCED_SUMMARY_CSV):
            print(f"📁 {ENHANCED_SUMMARY_CSV} - Detailed summary")
        if os.path.exists("enhanced_graphs.png"):
            print(f"📁 enhanced_graphs.png - Performance visualizations")
        
        print("\n📋 Quick Summary:")
        print("- Fixed success rate calculations for all test types")
        print("- Improved timeout handling to show actual response times")
        print("- Enhanced failure injection test logic")
        print("- Consistent rate limit detection and reporting")
        print("- Better error analysis and recommendations")
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        monitoring_active = False
        save_system_metrics()

if __name__ == "__main__":
    main()