
## 1. Updated Flask Application (app.py)

 
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

app = Flask(__name__)

# Global variables
allocation_state = {
    'model_trained': False,
    'current_allocations': [],
    'performance_metrics': {}
}

class IntelligentServerAllocator:
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
        self.load_data()
        self.train_model()
    
    def load_data(self):
        """Load all required datasets"""
        try:
            self.servers_df = pd.read_csv('server_infrastructure.csv')
            self.server_status_df = pd.read_csv('server_realtime_status.csv')
            self.tasks_df = pd.read_csv('task_requests.csv')
            self.allocation_history_df = pd.read_csv('allocation_history.csv')
            self.environmental_df = pd.read_csv('environmental_data.csv')
            print("✅ All datasets loaded successfully")
        except FileNotFoundError as e:
            print(f"❌ Dataset not found: {e}")
            self.create_minimal_data()
    
    def create_minimal_data(self):
        """Create minimal dummy data if CSV files are missing"""
        print("🔄 Creating minimal dummy data...")
        
        # Create 15 servers
        server_data = []
        for i in range(15):
            server_data.append({
                'server_id': f'SRV-{i+1:02d}',
                'cpu_cores': random.choice([4, 8, 16, 24, 32, 48, 64]),
                'ram_gb': random.choice([8, 16, 32, 64, 128, 256, 512]),
                'disk_tb': random.choice([0.5, 1, 2, 4, 6, 8, 10]),
                'base_power_watts': random.randint(120, 800),
                'cooling_efficiency': round(random.uniform(0.65, 0.95), 2)
            })
        
        self.servers_df = pd.DataFrame(server_data)
        
        # Create allocation history for training
        history_data = []
        for i in range(1000):
            history_data.append({
                'cpu_available_pct': random.uniform(10, 90),
                'ram_available_gb': random.uniform(1, 64),
                'disk_available_tb': random.uniform(0.1, 5),
                'current_load_pct': random.uniform(10, 85),
                'estimated_burst_time_sec': random.uniform(1, 1800),
                'cpu_requirement_pct': random.uniform(1, 50),
                'ram_requirement_gb': random.uniform(0.1, 16),
                'disk_requirement_gb': random.uniform(0.01, 10),
                'task_priority_score': random.randint(1, 4),
                'performance_score': random.uniform(60, 100)
            })
        
        self.allocation_history_df = pd.DataFrame(history_data)
    
    def train_model(self):
        """Train Random Forest model for server allocation"""
        try:
            print("🤖 Training Random Forest model...")
            
            # Prepare training data
            X = self.allocation_history_df[self.feature_columns].fillna(0)
            y = self.allocation_history_df['performance_score'].fillna(75)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.rf_model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            print(f"✅ Model trained successfully!")
            print(f"📊 R² Score: {r2:.3f}")
            print(f"📊 MSE: {mse:.3f}")
            
            global allocation_state
            allocation_state['model_trained'] = True
            allocation_state['performance_metrics'] = {
                'r2_score': r2,
                'mse': mse,
                'feature_importance': dict(zip(self.feature_columns, self.rf_model.feature_importances_))
            }
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            allocation_state['model_trained'] = False
    
    def get_current_server_status(self):
        """Get current status of all servers"""
        current_status = {}
        
        for _, server in self.servers_df.iterrows():
            # Simulate current server status
            current_status[server['server_id']] = {
                'cpu_available_pct': round(random.uniform(20, 80), 2),
                'ram_available_gb': round(random.uniform(1, server['ram_gb'] * 0.7), 2),
                'disk_available_tb': round(random.uniform(0.1, server['disk_tb'] * 0.6), 3),
                'current_load_pct': round(random.uniform(15, 75), 2),
                'temperature_c': round(25 + random.uniform(5, 35), 1),
                'active_threads': random.randint(0, server['cpu_cores'] * 2)
            }
        
        return current_status
    
    def predict_allocation_score(self, task_requirements, server_status):
        """Predict allocation performance score for a task-server combination"""
        if not self.rf_model:
            return random.uniform(60, 90)  # Fallback random score
        
        try:
            # Prepare feature vector
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
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict performance score
            score = self.rf_model.predict(features_scaled)[0]
            return max(0, min(100, score))  # Ensure score is between 0-100
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return random.uniform(60, 90)
    
    def allocate_task_to_server(self, task_requirements):
        """Find the optimal server for a given task using Random Forest"""
        current_server_status = self.get_current_server_status()
        best_server = None
        best_score = 0
        allocation_scores = {}
        
        # Evaluate each server
        for server_id, status in current_server_status.items():
            # Check if server can handle the task
            if (status['cpu_available_pct'] >= task_requirements['cpu_requirement_pct'] and
                status['ram_available_gb'] >= task_requirements['ram_requirement_gb'] and
                status['disk_available_tb'] >= task_requirements['disk_requirement_gb'] / 1024):  # Convert GB to TB
                
                # Predict allocation score
                score = self.predict_allocation_score(task_requirements, status)
                allocation_scores[server_id] = score
                
                if score > best_score:
                    best_score = score
                    best_server = server_id
            else:
                allocation_scores[server_id] = 0  # Server cannot handle task
        
        return {
            'recommended_server': best_server,
            'confidence_score': best_score,
            'all_server_scores': allocation_scores,
            'server_status': current_server_status
        }
    
    def simulate_task_processing(self, task_id, server_id, processing_time):
        """Simulate task processing and update server status"""
        # This would update server resource utilization in real implementation
        result = {
            'task_id': task_id,
            'server_id': server_id,
            'processing_time_sec': processing_time,
            'energy_consumed_wh': random.uniform(50, 200),
            'heat_generated_j': random.uniform(1000, 5000),
            'status': 'Completed',
            'completion_time': datetime.now()
        }
        return result

# Initialize the allocator
allocator = IntelligentServerAllocator()

# SimPy simulation for task processing
class TaskProcessingSimulation:
    def __init__(self, env, servers):
        self.env = env
        self.servers = {server_id: simpy.Resource(env, 1) for server_id in servers}
        self.processed_tasks = []
        self.total_energy = 0
        self.total_processing_time = 0
    
    def process_task(self, task_id, server_id, processing_time, energy_requirement):
        """Simulate task processing on a specific server"""
        start_time = self.env.now
        
        with self.servers[server_id].request() as request:
            yield request
            
            # Simulate processing
            yield self.env.timeout(processing_time)
            
            # Record results
            actual_time = self.env.now - start_time
            self.processed_tasks.append({
                'task_id': task_id,
                'server_id': server_id,
                'processing_time': actual_time,
                'energy_consumed': energy_requirement
            })
            
            self.total_energy += energy_requirement
            self.total_processing_time += actual_time

def run_allocation_simulation(tasks, duration=100):
    """Run SimPy simulation of task allocation"""
    env = simpy.Environment()
    server_ids = [f'SRV-{i+1:02d}' for i in range(15)]
    sim = TaskProcessingSimulation(env, server_ids)
    
    # Process tasks
    for i, task in enumerate(tasks[:50]):  # Limit to 50 tasks for simulation
        processing_time = task.get('estimated_burst_time_sec', 10)
        energy_req = random.uniform(10, 100)
        
        # Get allocation recommendation
        allocation_result = allocator.allocate_task_to_server(task)
        server_id = allocation_result['recommended_server'] or 'SRV-01'
        
        env.process(sim.process_task(f'TASK-{i+1}', server_id, processing_time, energy_req))
    
    env.run(until=duration)
    
    return {
        'processed_tasks': len(sim.processed_tasks),
        'total_energy_consumed': sim.total_energy,
        'total_processing_time': sim.total_processing_time,
        'average_energy_per_task': sim.total_energy / max(len(sim.processed_tasks), 1),
        'task_details': sim.processed_tasks
    }

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard-data')
def get_dashboard_data():
    """Get real-time dashboard data"""
    try:
        current_status = allocator.get_current_server_status()
        
        # Calculate aggregate metrics
        total_servers = len(allocator.servers_df)
        active_servers = sum(1 for status in current_status.values() if status['current_load_pct'] > 10)
        avg_cpu_available = np.mean([status['cpu_available_pct'] for status in current_status.values()])
        avg_temperature = np.mean([status['temperature_c'] for status in current_status.values()])
        total_active_threads = sum(status['active_threads'] for status in current_status.values())
        
        # Get recent environmental data
        current_carbon_intensity = random.uniform(250, 450)  # g CO2/kWh
        renewable_percentage = random.uniform(30, 70)
        current_pue = random.uniform(1.2, 1.6)
        
        dashboard_data = {
            'server_metrics': {
                'total_servers': total_servers,
                'active_servers': active_servers,
                'avg_cpu_available_pct': round(avg_cpu_available, 1),
                'avg_temperature_c': round(avg_temperature, 1),
                'total_active_threads': total_active_threads
            },
            'environmental_metrics': {
                'current_carbon_intensity': round(current_carbon_intensity, 1),
                'renewable_percentage': round(renewable_percentage, 1),
                'current_pue': round(current_pue, 2)
            },
            'ml_model_status': {
                'model_trained': allocation_state['model_trained'],
                'performance_metrics': allocation_state.get('performance_metrics', {})
            },
            'server_status': current_status
        }
        
        return jsonify({
            'success': True,
            'data': dashboard_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/allocate-task', methods=['POST'])
def allocate_task():
    """Allocate a new task to the optimal server"""
    try:
        task_data = request.get_json()
        
        # Validate required fields
        required_fields = ['cpu_requirement_pct', 'ram_requirement_gb', 'disk_requirement_gb', 'estimated_burst_time_sec']
        for field in required_fields:
            if field not in task_data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'})
        
        # Add priority score
        priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        task_data['task_priority_score'] = priority_map.get(task_data.get('priority', 'Medium'), 2)
        
        # Get allocation recommendation
        allocation_result = allocator.allocate_task_to_server(task_data)
        
        # Generate task ID
        task_id = f"TASK-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"
        
        # Simulate task processing
        if allocation_result['recommended_server']:
            processing_result = allocator.simulate_task_processing(
                task_id, 
                allocation_result['recommended_server'],
                task_data['estimated_burst_time_sec']
            )
            allocation_result['processing_result'] = processing_result
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'allocation_result': allocation_result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/server-comparison')
def get_server_comparison():
    """Get detailed comparison of all servers for current conditions"""
    try:
        # Sample task for comparison
        sample_task = {
            'cpu_requirement_pct': 25,
            'ram_requirement_gb': 4,
            'disk_requirement_gb': 1,
            'estimated_burst_time_sec': 120,
            'task_priority_score': 2
        }
        
        allocation_result = allocator.allocate_task_to_server(sample_task)
        
        # Add server specifications
        server_comparison = []
        for _, server in allocator.servers_df.iterrows():
            server_id = server['server_id']
            status = allocation_result['server_status'].get(server_id, {})
            score = allocation_result['all_server_scores'].get(server_id, 0)
            
            server_comparison.append({
                'server_id': server_id,
                'cpu_cores': server['cpu_cores'],
                'ram_gb': server['ram_gb'],
                'disk_tb': server['disk_tb'],
                'cpu_available_pct': status.get('cpu_available_pct', 0),
                'ram_available_gb': status.get('ram_available_gb', 0),
                'current_load_pct': status.get('current_load_pct', 0),
                'temperature_c': status.get('temperature_c', 0),
                'allocation_score': round(score, 2),
                'recommendation': 'Recommended' if server_id == allocation_result['recommended_server'] else 'Available' if score > 0 else 'Overloaded'
            })
        
        # Sort by allocation score
        server_comparison.sort(key=lambda x: x['allocation_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'servers': server_comparison,
            'sample_task': sample_task
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/run-simulation', methods=['POST'])
def run_simulation():
    """Run allocation simulation"""
    try:
        request_data = request.get_json()
        num_tasks = int(request_data.get('num_tasks', 50))
        duration = int(request_data.get('duration', 100))
        
        # Generate sample tasks
        sample_tasks = []
        task_types = ['Web-Request', 'Data-Analysis', 'ML-Training', 'File-Processing']
        
        for i in range(num_tasks):
            task_type = random.choice(task_types)
            
            if task_type == 'Web-Request':
                cpu_req = random.uniform(1, 10)
                ram_req = random.uniform(0.1, 2)
                burst_time = random.uniform(0.1, 5)
            elif task_type == 'Data-Analysis':
                cpu_req = random.uniform(15, 40)
                ram_req = random.uniform(2, 16)
                burst_time = random.uniform(30, 300)
            elif task_type == 'ML-Training':
                cpu_req = random.uniform(30, 70)
                ram_req = random.uniform(8, 64)
                burst_time = random.uniform(300, 1800)
            else:  # File-Processing
                cpu_req = random.uniform(10, 25)
                ram_req = random.uniform(1, 8)
                burst_time = random.uniform(10, 120)
            
            sample_tasks.append({
                'task_type': task_type,
                'cpu_requirement_pct': cpu_req,
                'ram_requirement_gb': ram_req,
                'disk_requirement_gb': random.uniform(0.1, 5),
                'estimated_burst_time_sec': burst_time,
                'task_priority_score': random.randint(1, 4)
            })
        
        # Run simulation
        simulation_results = run_allocation_simulation(sample_tasks, duration)
        
        return jsonify({
            'success': True,
            'results': simulation_results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model-performance')
def get_model_performance():
    """Get Random Forest model performance metrics"""
    try:
        if not allocation_state['model_trained']:
            return jsonify({'success': False, 'error': 'Model not trained yet'})
        
        performance_data = allocation_state['performance_metrics'].copy()
        
        # Add feature importance ranking
        if 'feature_importance' in performance_data:
            importance_items = list(performance_data['feature_importance'].items())
            importance_items.sort(key=lambda x: x[1], reverse=True)
            performance_data['feature_importance_ranked'] = importance_items
        
        return jsonify({
            'success': True,
            'performance': performance_data,
            'model_info': {
                'algorithm': 'Random Forest Regressor',
                'n_estimators': 100,
                'max_depth': 10,
                'training_samples': len(allocator.allocation_history_df) if allocator.allocation_history_df is not None else 0
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
