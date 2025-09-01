from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import simpy
import threading
import time
from functools import lru_cache, wraps

app = Flask(__name__)

# Performance caching decorator
def cache_for(seconds=60):
    def decorator(func):
        func._cache = {}
        func._cache_time = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()
            
            if key in func._cache and (now - func._cache_time[key]) < seconds:
                return func._cache[key]
            
            result = func(*args, **kwargs)
            func._cache[key] = result
            func._cache_time[key] = now
            
            # Clean old cache entries
            if len(func._cache) > 100:
                oldest_key = min(func._cache_time.keys(), key=lambda k: func._cache_time[k])
                del func._cache[oldest_key]
                del func._cache_time[oldest_key]
            
            return result
        return wrapper
    return decorator

# Global variables
allocation_state = {
    'model_trained': False,
    'current_allocations': [],
    'performance_metrics': {},
    'firebase_enabled': True,
    'total_tasks_processed': 0,
    'system_uptime': datetime.now()
}

class OptimizedIntelligentServerAllocator:
    def __init__(self):
        self.rf_model = None
        self.scaler = StandardScaler()
        self.servers_df = None
        self.allocation_history_df = None
        self.feature_columns = [
            'cpu_available_pct', 'ram_available_gb', 'disk_available_tb', 
            'current_load_pct', 'estimated_burst_time_sec', 'cpu_requirement_pct',
            'ram_requirement_gb', 'disk_requirement_gb', 'task_priority_score'
        ]
        self.server_cache = {}
        self.cache_timeout = 5  # seconds
        self.load_data()
        self.train_model()
    
    def load_data(self):
        """Load all required datasets with optimization"""
        try:
            self.servers_df = pd.read_csv('server_infrastructure.csv')
            self.server_status_df = pd.read_csv('server_realtime_status.csv')
            self.tasks_df = pd.read_csv('task_requests.csv')
            self.allocation_history_df = pd.read_csv('allocation_history.csv')
            self.environmental_df = pd.read_csv('environmental_data.csv')
            print("✅ All datasets loaded successfully")
        except FileNotFoundError as e:
            print(f"❌ Dataset not found: {e}")
            self.create_optimized_data()
    
    def create_optimized_data(self):
        """Create optimized dummy data"""
        print("🔄 Creating optimized dummy data...")
        
        # Reduced server configurations for better performance
        server_configs = [
            {'tier': 'High-End', 'cpu_cores': 64, 'ram_gb': 512, 'disk_tb': 10, 'base_power_watts': 800},
            {'tier': 'High-End', 'cpu_cores': 48, 'ram_gb': 256, 'disk_tb': 8, 'base_power_watts': 650},
            {'tier': 'High-End', 'cpu_cores': 32, 'ram_gb': 128, 'disk_tb': 6, 'base_power_watts': 500},
            {'tier': 'Mid-Range', 'cpu_cores': 24, 'ram_gb': 64, 'disk_tb': 4, 'base_power_watts': 400},
            {'tier': 'Mid-Range', 'cpu_cores': 16, 'ram_gb': 32, 'disk_tb': 2, 'base_power_watts': 300},
            {'tier': 'Mid-Range', 'cpu_cores': 16, 'ram_gb': 64, 'disk_tb': 4, 'base_power_watts': 350},
            {'tier': 'Mid-Range', 'cpu_cores': 20, 'ram_gb': 48, 'disk_tb': 3, 'base_power_watts': 380},
            {'tier': 'Mid-Range', 'cpu_cores': 12, 'ram_gb': 24, 'disk_tb': 2, 'base_power_watts': 250},
            {'tier': 'Standard', 'cpu_cores': 8, 'ram_gb': 16, 'disk_tb': 1, 'base_power_watts': 200},
            {'tier': 'Standard', 'cpu_cores': 8, 'ram_gb': 32, 'disk_tb': 2, 'base_power_watts': 220},
            {'tier': 'Standard', 'cpu_cores': 12, 'ram_gb': 16, 'disk_tb': 1.5, 'base_power_watts': 230},
            {'tier': 'Standard', 'cpu_cores': 4, 'ram_gb': 8, 'disk_tb': 0.5, 'base_power_watts': 120},
            {'tier': 'Standard', 'cpu_cores': 6, 'ram_gb': 12, 'disk_tb': 1, 'base_power_watts': 150},
            {'tier': 'Standard', 'cpu_cores': 4, 'ram_gb': 16, 'disk_tb': 1, 'base_power_watts': 140},
            {'tier': 'Standard', 'cpu_cores': 8, 'ram_gb': 8, 'disk_tb': 0.5, 'base_power_watts': 160}
        ]
        
        server_data = []
        for i, config in enumerate(server_configs):
            server_data.append({
                'server_id': f'SRV-{i+1:02d}',
                'server_tier': config['tier'],
                'cpu_cores': config['cpu_cores'],
                'ram_gb': config['ram_gb'],
                'disk_tb': config['disk_tb'],
                'base_power_watts': config['base_power_watts'],
                'cooling_efficiency': round(random.uniform(0.65, 0.95), 2),
                'location': f'Rack-{(i//5)+1}',
                'age_months': random.randint(6, 48),
                'status': 'active'
            })
        
        self.servers_df = pd.DataFrame(server_data)
        
        # Optimized allocation history - reduced size for better performance
        history_data = []
        for i in range(1000):  # Reduced from 2000
            cpu_available = random.uniform(10, 90)
            load_pct = max(5, min(95, 100 - cpu_available + random.uniform(-10, 10)))
            
            ram_available = random.uniform(1, 128)
            disk_available = random.uniform(0.1, 8)
            
            cpu_req = random.uniform(1, 70)
            ram_req = random.uniform(0.1, 32)
            disk_req = random.uniform(0.01, 10)
            burst_time = random.uniform(1, 1800)
            priority_score = random.randint(1, 4)
            
            resource_efficiency = min(
                cpu_available / max(cpu_req, 1),
                (ram_available * 1024) / max(ram_req * 1024, 1),
                (disk_available * 1024) / max(disk_req, 1)
            )
            load_efficiency = (100 - load_pct) / 100
            priority_bonus = priority_score * 2
            
            performance_score = min(100, 
                (resource_efficiency * 30 + load_efficiency * 50 + priority_bonus + 10) * 
                random.uniform(0.8, 1.2)
            )
            
            history_data.append({
                'cpu_available_pct': round(cpu_available, 2),
                'ram_available_gb': round(ram_available, 2),
                'disk_available_tb': round(disk_available, 3),
                'current_load_pct': round(load_pct, 2),
                'estimated_burst_time_sec': round(burst_time, 2),
                'cpu_requirement_pct': round(cpu_req, 2),
                'ram_requirement_gb': round(ram_req, 2),
                'disk_requirement_gb': round(disk_req, 2),
                'task_priority_score': priority_score,
                'performance_score': max(0, min(100, performance_score))
            })
        
        self.allocation_history_df = pd.DataFrame(history_data)
    
    def train_model(self):
        """Optimized model training"""
        try:
            print("🤖 Training optimized Random Forest model...")
            
            X = self.allocation_history_df[self.feature_columns].fillna(0)
            y = self.allocation_history_df['performance_score'].fillna(75)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Optimized Random Forest parameters for faster training
            self.rf_model = RandomForestRegressor(
                n_estimators=50,  # Reduced from 150
                max_depth=10,     # Reduced from 15
                min_samples_split=10,  # Increased for speed
                min_samples_leaf=5,    # Increased for speed
                random_state=42,
                n_jobs=-1
            )
            
            self.rf_model.fit(X_train_scaled, y_train)
            
            y_pred = self.rf_model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            print(f"✅ Optimized model trained successfully!")
            print(f"📊 R² Score: {r2:.3f}")
            print(f"📊 MSE: {mse:.3f}")
            
            global allocation_state
            allocation_state['model_trained'] = True
            allocation_state['performance_metrics'] = {
                'r2_score': r2,
                'mse': mse,
                'feature_importance': dict(zip(self.feature_columns, self.rf_model.feature_importances_)),
                'training_timestamp': datetime.now().isoformat(),
                'training_samples': len(X_train)
            }
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            allocation_state['model_trained'] = False
    
    @cache_for(5)  # Cache server status for 5 seconds
    def get_current_server_status_optimized(self, limit_servers=10):
        """Highly optimized server status with caching"""
        current_time = datetime.now()
        hour = current_time.hour
        
        # Simplified load calculation
        if 9 <= hour <= 17:
            base_load_multiplier = 1.3
        elif 18 <= hour <= 22:
            base_load_multiplier = 1.1
        else:
            base_load_multiplier = 0.7
        
        current_status = {}
        
        # Limit servers for better performance
        servers_to_process = self.servers_df.head(limit_servers)
        
        for _, server in servers_to_process.iterrows():
            server_id = server['server_id']
            
            # Use vectorized operations where possible
            current_cpu_usage = min(0.85, random.uniform(0.15, 0.6) * base_load_multiplier)
            cpu_available_pct = round((1 - current_cpu_usage) * 100, 1)
            
            ram_usage_pct = min(0.8, current_cpu_usage * 0.7 + random.uniform(-0.1, 0.1))
            ram_available_gb = round(server['ram_gb'] * (1 - ram_usage_pct), 1)
            
            disk_usage_pct = random.uniform(0.2, 0.6)
            disk_available_tb = round(server['disk_tb'] * (1 - disk_usage_pct), 2)
            
            current_load_pct = round(current_cpu_usage * 100, 1)
            temperature_c = round(25 + (current_cpu_usage * 25) + random.uniform(-3, 3), 1)
            active_threads = int(server['cpu_cores'] * current_cpu_usage)
            power_consumption = round(server['base_power_watts'] * (0.3 + current_cpu_usage * 0.7), 1)
            
            current_status[server_id] = {
                'cpu_available_pct': cpu_available_pct,
                'ram_available_gb': ram_available_gb,
                'disk_available_tb': disk_available_tb,
                'current_load_pct': current_load_pct,
                'temperature_c': temperature_c,
                'active_threads': active_threads,
                'server_tier': server.get('server_tier', 'Standard'),
                'power_consumption_watts': power_consumption,
                'timestamp': current_time.isoformat()
            }
        
        return current_status
    
    def predict_allocation_score(self, task_requirements, server_status):
        """Optimized prediction with fallback"""
        if not self.rf_model:
            return random.uniform(60, 90)
        
        try:
            features = np.array([[
                server_status['cpu_available_pct'],
                server_status['ram_available_gb'],
                server_status['disk_available_tb'],
                server_status['current_load_pct'],
                task_requirements['estimated_burst_time_sec'],
                task_requirements['cpu_requirement_pct'],
                task_requirements['ram_requirement_gb'],
                task_requirements['disk_requirement_gb'],
                task_requirements['task_priority_score']
            ]])
            
            features_scaled = self.scaler.transform(features)
            score = self.rf_model.predict(features_scaled)[0]
            return max(0, min(100, score))
            
        except Exception as e:
            return random.uniform(60, 90)
    
    def allocate_task_to_server_optimized(self, task_requirements):
        """Optimized task allocation"""
        current_server_status = self.get_current_server_status_optimized()
        best_server = None
        best_score = 0
        allocation_scores = {}
        
        for server_id, status in current_server_status.items():
            # Quick resource check
            can_handle = (
                status['cpu_available_pct'] >= task_requirements['cpu_requirement_pct'] and
                status['ram_available_gb'] >= task_requirements['ram_requirement_gb'] and
                status['disk_available_tb'] >= task_requirements['disk_requirement_gb'] / 1024
            )
            
            if can_handle:
                score = self.predict_allocation_score(task_requirements, status)
                allocation_scores[server_id] = score
                
                if score > best_score:
                    best_score = score
                    best_server = server_id
            else:
                allocation_scores[server_id] = 0
        
        # Calculate simplified additional metrics
        energy_efficiency_score = 0
        heat_impact_score = 0
        
        if best_server:
            server_status = current_server_status[best_server]
            energy_efficiency_score = max(0, 100 - server_status['power_consumption_watts'] / 10)
            heat_impact_score = max(0, 100 - server_status['temperature_c'])
        
        return {
            'recommended_server': best_server,
            'confidence_score': best_score,
            'all_server_scores': allocation_scores,
            'server_status': current_server_status,
            'allocation_timestamp': datetime.now().isoformat(),
            'energy_efficiency_score': round(energy_efficiency_score, 1),
            'heat_impact_score': round(heat_impact_score, 1)
        }
    
    def simulate_task_processing_optimized(self, task_id, server_id, processing_time):
        """Optimized task processing simulation"""
        start_time = datetime.now()
        
        # Simplified calculations
        server_info = self.servers_df[self.servers_df['server_id'] == server_id].iloc[0]
        base_power = server_info['base_power_watts']
        
        energy_consumed_wh = (base_power * processing_time) / 3600
        heat_generated_j = energy_consumed_wh * 3600 * 0.7
        carbon_intensity = random.uniform(300, 500)
        carbon_footprint_g = (energy_consumed_wh / 1000) * carbon_intensity
        
        result = {
            'task_id': task_id,
            'server_id': server_id,
            'processing_time_sec': processing_time,
            'energy_consumed_wh': round(energy_consumed_wh, 2),
            'heat_generated_j': round(heat_generated_j, 2),
            'carbon_footprint_g': round(carbon_footprint_g, 2),
            'status': 'Completed',
            'start_time': start_time.isoformat(),
            'completion_time': datetime.now().isoformat(),
            'server_tier': server_info['server_tier']
        }
        
        global allocation_state
        allocation_state['total_tasks_processed'] += 1
        
        return result

# Initialize optimized allocator
allocator = OptimizedIntelligentServerAllocator()

# Optimized SimPy simulation
class OptimizedTaskProcessingSimulation:
    def __init__(self, env, servers):
        self.env = env
        self.servers = {server_id: simpy.Resource(env, 1) for server_id in servers}
        self.processed_tasks = []
        self.total_energy = 0
        self.total_processing_time = 0
        self.total_carbon_footprint = 0
    
    def process_task(self, task_id, server_id, processing_time, energy_requirement):
        """Optimized task processing"""
        start_time = self.env.now
        
        with self.servers[server_id].request() as request:
            yield request
            yield self.env.timeout(processing_time)
            
            actual_time = self.env.now - start_time
            carbon_footprint = energy_requirement * random.uniform(0.3, 0.5)
            
            self.processed_tasks.append({
                'task_id': task_id,
                'server_id': server_id,
                'processing_time': actual_time,
                'energy_consumed': energy_requirement,
                'carbon_footprint': carbon_footprint
            })
            
            self.total_energy += energy_requirement
            self.total_processing_time += actual_time
            self.total_carbon_footprint += carbon_footprint

def run_optimized_simulation(tasks, duration=100):
    """Optimized allocation simulation"""
    env = simpy.Environment()
    server_ids = [f'SRV-{i+1:02d}' for i in range(min(10, len(tasks)))]  # Limit servers
    sim = OptimizedTaskProcessingSimulation(env, server_ids)
    
    # Limit tasks for better performance
    limited_tasks = tasks[:min(30, len(tasks))]
    
    for i, task in enumerate(limited_tasks):
        processing_time = task.get('estimated_burst_time_sec', 10)
        energy_req = random.uniform(10, 200)
        
        allocation_result = allocator.allocate_task_to_server_optimized(task)
        server_id = allocation_result['recommended_server'] or server_ids[0]
        
        env.process(sim.process_task(f'TASK-{i+1}', server_id, processing_time, energy_req))
    
    env.run(until=duration)
    
    return {
        'processed_tasks': len(sim.processed_tasks),
        'total_energy_consumed': round(sim.total_energy, 2),
        'total_processing_time': round(sim.total_processing_time, 2),
        'total_carbon_footprint': round(sim.total_carbon_footprint, 2),
        'average_energy_per_task': round(sim.total_energy / max(len(sim.processed_tasks), 1), 2),
        'average_processing_time': round(sim.total_processing_time / max(len(sim.processed_tasks), 1), 2),
        'task_details': sim.processed_tasks
    }

# Optimized Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard-data')
@cache_for(8)  # Cache for 8 seconds
def get_dashboard_data_optimized():
    """Highly optimized dashboard data"""
    try:
        current_status = allocator.get_current_server_status_optimized(limit_servers=10)
        
        # Efficient calculations using numpy
        load_values = np.array([status.get('current_load_pct', 0) for status in current_status.values()])
        temp_values = np.array([status.get('temperature_c', 25) for status in current_status.values()])
        power_values = np.array([status.get('power_consumption_watts', 0) for status in current_status.values()])
        thread_values = np.array([status.get('active_threads', 0) for status in current_status.values()])
        
        # Fast aggregations
        total_servers = len(allocator.servers_df)
        active_servers = np.sum(load_values > 5)
        avg_cpu_available = round(100 - np.mean(load_values), 1)
        avg_temperature = round(np.mean(temp_values), 1)
        total_active_threads = int(np.sum(thread_values))
        total_power_consumption = round(np.sum(power_values) / 1000, 2)
        
        # Simplified environmental metrics
        current_carbon_intensity = random.randint(250, 450)
        renewable_percentage = random.randint(30, 70)
        current_pue = round(random.uniform(1.2, 1.6), 2)
        
        uptime_hours = (datetime.now() - allocation_state['system_uptime']).total_seconds() / 3600
        
        dashboard_data = {
            'server_metrics': {
                'total_servers': total_servers,
                'active_servers': int(active_servers),
                'avg_cpu_available_pct': avg_cpu_available,
                'avg_temperature_c': avg_temperature,
                'total_active_threads': total_active_threads,
                'total_power_consumption_kw': total_power_consumption
            },
            'environmental_metrics': {
                'current_carbon_intensity': current_carbon_intensity,
                'renewable_percentage': renewable_percentage,
                'current_pue': current_pue,
                'estimated_carbon_emissions_kg_per_hour': round(total_power_consumption * (current_carbon_intensity / 1000), 2)
            },
            'system_metrics': {
                'total_tasks_processed': allocation_state['total_tasks_processed'],
                'system_uptime_hours': round(uptime_hours, 2),
                'model_trained': allocation_state['model_trained'],
                'firebase_enabled': allocation_state['firebase_enabled']
            },
            'server_status': current_status,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'data': dashboard_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/server-comparison')
@cache_for(6)  # Cache for 6 seconds
def get_server_comparison_optimized():
    """Optimized server comparison"""
    try:
        sample_task = {
            'cpu_requirement_pct': 25,
            'ram_requirement_gb': 4,
            'disk_requirement_gb': 1,
            'estimated_burst_time_sec': 120,
            'task_priority_score': 2
        }
        
        allocation_result = allocator.allocate_task_to_server_optimized(sample_task)
        
        server_comparison = []
        for _, server in allocator.servers_df.head(10).iterrows():  # Limit to 10 servers
            server_id = server['server_id']
            status = allocation_result['server_status'].get(server_id, {})
            score = allocation_result['all_server_scores'].get(server_id, 0)
            
            server_comparison.append({
                'server_id': server_id,
                'server_tier': server.get('server_tier', 'Standard'),
                'cpu_cores': server['cpu_cores'],
                'ram_gb': server['ram_gb'],
                'disk_tb': server['disk_tb'],
                'cpu_available_pct': status.get('cpu_available_pct', 0),
                'ram_available_gb': status.get('ram_available_gb', 0),
                'current_load_pct': status.get('current_load_pct', 0),
                'temperature_c': status.get('temperature_c', 0),
                'power_consumption_watts': status.get('power_consumption_watts', 0),
                'allocation_score': round(score, 2),
                'performance_trend': 'stable',  # Simplified
                'recommendation': 'Recommended' if server_id == allocation_result['recommended_server'] else 'Available' if score > 0 else 'Overloaded'
            })
        
        server_comparison.sort(key=lambda x: x['allocation_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'servers': server_comparison,
            'sample_task': sample_task,
            'allocation_summary': {
                'recommended_server': allocation_result['recommended_server'],
                'confidence_score': allocation_result['confidence_score'],
                'energy_efficiency_score': allocation_result['energy_efficiency_score'],
                'heat_impact_score': allocation_result['heat_impact_score']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/allocate-task', methods=['POST'])
def allocate_task_optimized():
    """Optimized task allocation"""
    try:
        task_data = request.get_json()
        
        required_fields = ['cpu_requirement_pct', 'ram_requirement_gb', 'disk_requirement_gb', 'estimated_burst_time_sec']
        for field in required_fields:
            if field not in task_data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'})
        
        priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        task_data['task_priority_score'] = priority_map.get(task_data.get('priority', 'Medium'), 2)
        
        task_id = f"TASK-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"
        
        allocation_result = allocator.allocate_task_to_server_optimized(task_data)
        
        processing_result = None
        if allocation_result['recommended_server']:
            processing_result = allocator.simulate_task_processing_optimized(
                task_id, 
                allocation_result['recommended_server'],
                task_data['estimated_burst_time_sec']
            )
            allocation_result['processing_result'] = processing_result
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'allocation_result': allocation_result,
            'task_data': task_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/run-simulation', methods=['POST'])
def run_simulation_optimized():
    """Optimized simulation"""
    try:
        request_data = request.get_json()
        num_tasks = min(int(request_data.get('num_tasks', 30)), 50)  # Limit tasks
        duration = min(int(request_data.get('duration', 100)), 200)  # Limit duration
        
        task_types = ['Web-Request', 'Data-Analysis', 'ML-Training', 'File-Processing']
        sample_tasks = []
        
        # Optimized task generation
        task_specs = {
            'Web-Request': {'cpu': (1, 10), 'ram': (0.1, 2), 'burst': (0.1, 5)},
            'Data-Analysis': {'cpu': (15, 40), 'ram': (2, 16), 'burst': (30, 300)},
            'ML-Training': {'cpu': (30, 70), 'ram': (8, 64), 'burst': (300, 1800)},
            'File-Processing': {'cpu': (10, 25), 'ram': (1, 8), 'burst': (10, 120)}
        }
        
        for i in range(num_tasks):
            task_type = random.choice(task_types)
            spec = task_specs[task_type]
            
            sample_tasks.append({
                'task_type': task_type,
                'cpu_requirement_pct': random.uniform(*spec['cpu']),
                'ram_requirement_gb': random.uniform(*spec['ram']),
                'disk_requirement_gb': random.uniform(0.1, 5),
                'estimated_burst_time_sec': random.uniform(*spec['burst']),
                'task_priority_score': random.randint(1, 4)
            })
        
        simulation_results = run_optimized_simulation(sample_tasks, duration)
        
        return jsonify({
            'success': True,
            'results': simulation_results,
            'simulation_config': {
                'num_tasks': num_tasks,
                'duration': duration,
                'task_types_distribution': {task_type: sum(1 for t in sample_tasks if t['task_type'] == task_type) for task_type in task_types}
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model-performance')
@cache_for(30)  # Cache for 30 seconds
def get_model_performance_optimized():
    """Optimized model performance"""
    try:
        if not allocation_state['model_trained']:
            return jsonify({'success': False, 'error': 'Model not trained yet'})
        
        performance_data = allocation_state['performance_metrics'].copy()
        
        if 'feature_importance' in performance_data:
            importance_items = list(performance_data['feature_importance'].items())
            importance_items.sort(key=lambda x: x[1], reverse=True)
            performance_data['feature_importance_ranked'] = importance_items[:8]  # Limit to top 8
        
        return jsonify({
            'success': True,
            'performance': performance_data,
            'model_info': {
                'algorithm': 'Optimized Random Forest Regressor',
                'n_estimators': 50,
                'max_depth': 10,
                'training_samples': len(allocator.allocation_history_df),
                'feature_count': len(allocator.feature_columns)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/firebase/servers', methods=['GET'])
def get_servers_for_firebase():
    """Optimized server data for Firebase"""
    try:
        servers_data = allocator.servers_df.head(10).to_dict('records')  # Limit servers
        return jsonify({
            'success': True,
            'servers': servers_data,
            'count': len(servers_data),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/firebase/training-data', methods=['GET'])
def get_training_data_for_firebase():
    """Optimized training data for Firebase"""
    try:
        limit = min(int(request.args.get('limit', 100)), 500)  # Limit data
        training_data = allocator.allocation_history_df.head(limit).to_dict('records')
        return jsonify({
            'success': True,
            'training_data': training_data,
            'total_available': len(allocator.allocation_history_df),
            'returned_count': len(training_data),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/system-status')
@cache_for(10)  # Cache for 10 seconds
def get_system_status():
    """Optimized system status"""
    try:
        uptime_delta = datetime.now() - allocation_state['system_uptime']
        
        system_status = {
            'status': 'healthy',
            'uptime_seconds': uptime_delta.total_seconds(),
            'total_tasks_processed': allocation_state['total_tasks_processed'],
            'model_trained': allocation_state['model_trained'],
            'firebase_enabled': allocation_state['firebase_enabled'],
            'active_servers': len([s for s in allocator.get_current_server_status_optimized().values() if s['current_load_pct'] > 5]),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'system_status': system_status
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🌱 NeuroSimGreen Optimized Server Starting...")
    print(f"📊 Servers loaded: {len(allocator.servers_df)}")
    print(f"🤖 Model trained: {allocation_state['model_trained']}")
    print("🚀 Ready for high-performance intelligent server allocation!")
    app.run(debug=True, host='0.0.0.0', port=5000)
