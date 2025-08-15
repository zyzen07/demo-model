# Create comprehensive dummy CSV data for the NeuroSimGreen project
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# 1. Data Center Infrastructure Data
def create_datacenter_infrastructure():
    data = []
    server_types = ['CPU-Intensive', 'GPU-Intensive', 'Memory-Intensive', 'Storage-Intensive']
    locations = ['US-East', 'US-West', 'EU-Central', 'Asia-Pacific']
    
    for i in range(100):  # 100 servers
        data.append({
            'server_id': f'SRV-{i+1:03d}',
            'server_type': random.choice(server_types),
            'location': random.choice(locations),
            'cpu_cores': random.choice([8, 16, 32, 64]),
            'memory_gb': random.choice([32, 64, 128, 256]),
            'storage_tb': random.choice([1, 2, 4, 8]),
            'power_rating_watts': random.randint(200, 800),
            'efficiency_rating': round(random.uniform(0.7, 0.95), 2),
            'cooling_requirement': round(random.uniform(50, 200), 2),
            'age_years': random.randint(1, 8)
        })
    
    return pd.DataFrame(data)

# 2. Energy and Carbon Data
def create_energy_carbon_data():
    data = []
    locations = ['US-East', 'US-West', 'EU-Central', 'Asia-Pacific']
    
    # Generate hourly data for 30 days
    start_date = datetime(2024, 1, 1)
    for day in range(30):
        for hour in range(24):
            timestamp = start_date + timedelta(days=day, hours=hour)
            for location in locations:
                # Carbon intensity varies by location and time
                base_carbon = {'US-East': 400, 'US-West': 300, 'EU-Central': 250, 'Asia-Pacific': 500}
                carbon_intensity = base_carbon[location] + random.randint(-50, 50)
                
                # Renewable energy percentage varies
                renewable_pct = random.uniform(0.2, 0.8)
                
                data.append({
                    'timestamp': timestamp,
                    'location': location,
                    'carbon_intensity_gco2_kwh': carbon_intensity,
                    'renewable_energy_pct': round(renewable_pct, 2),
                    'grid_demand_mw': random.randint(500, 2000),
                    'electricity_price_per_kwh': round(random.uniform(0.08, 0.15), 3)
                })
    
    return pd.DataFrame(data)

# 3. Workload Data
def create_workload_data():
    data = []
    workload_types = ['ML-Training', 'Web-Service', 'Data-Processing', 'AI-Inference']
    priorities = ['Low', 'Medium', 'High', 'Critical']
    
    for i in range(500):  # 500 workloads
        workload_type = random.choice(workload_types)
        
        # Different workload types have different characteristics
        if workload_type == 'ML-Training':
            duration_hours = random.uniform(2, 24)
            cpu_usage = random.uniform(0.7, 0.95)
            memory_usage = random.uniform(0.6, 0.9)
            energy_efficiency = random.uniform(0.6, 0.8)
        elif workload_type == 'Web-Service':
            duration_hours = random.uniform(0.1, 2)
            cpu_usage = random.uniform(0.2, 0.5)
            memory_usage = random.uniform(0.3, 0.6)
            energy_efficiency = random.uniform(0.8, 0.95)
        elif workload_type == 'Data-Processing':
            duration_hours = random.uniform(1, 8)
            cpu_usage = random.uniform(0.5, 0.8)
            memory_usage = random.uniform(0.7, 0.95)
            energy_efficiency = random.uniform(0.7, 0.85)
        else:  # AI-Inference
            duration_hours = random.uniform(0.05, 0.5)
            cpu_usage = random.uniform(0.4, 0.7)
            memory_usage = random.uniform(0.3, 0.6)
            energy_efficiency = random.uniform(0.75, 0.9)
        
        data.append({
            'workload_id': f'WL-{i+1:04d}',
            'workload_type': workload_type,
            'priority': random.choice(priorities),
            'estimated_duration_hours': round(duration_hours, 2),
            'cpu_requirement': round(cpu_usage, 2),
            'memory_requirement_gb': random.randint(4, 64),
            'estimated_energy_kwh': round(random.uniform(0.5, 10), 2),
            'carbon_tolerance': random.choice(['Flexible', 'Moderate', 'Strict']),
            'deadline_hours': round(duration_hours * random.uniform(1.2, 3.0), 2),
            'energy_efficiency_score': round(energy_efficiency, 2)
        })
    
    return pd.DataFrame(data)

# 4. Historical Performance Data
def create_performance_data():
    data = []
    
    # Generate performance data for past 30 days
    start_date = datetime(2024, 1, 1)
    for day in range(30):
        for hour in range(24):
            timestamp = start_date + timedelta(days=day, hours=hour)
            
            data.append({
                'timestamp': timestamp,
                'total_power_consumption_kw': round(random.uniform(800, 1200), 2),
                'total_carbon_emissions_kg': round(random.uniform(300, 600), 2),
                'cpu_utilization_pct': round(random.uniform(40, 85), 1),
                'memory_utilization_pct': round(random.uniform(45, 80), 1),
                'cooling_efficiency_cop': round(random.uniform(2.5, 4.0), 2),
                'pue_power_usage_effectiveness': round(random.uniform(1.2, 1.8), 2),
                'renewable_energy_used_pct': round(random.uniform(20, 80), 1),
                'workloads_completed': random.randint(50, 150),
                'avg_response_time_ms': round(random.uniform(100, 500), 1)
            })
    
    return pd.DataFrame(data)

# 5. Sustainability Metrics Data
def create_sustainability_metrics():
    data = []
    metrics = ['Carbon_Footprint', 'Energy_Efficiency', 'Renewable_Usage', 
               'Resource_Utilization', 'Cooling_Efficiency', 'Waste_Heat_Recovery']
    
    start_date = datetime(2024, 1, 1)
    for day in range(30):
        date = start_date + timedelta(days=day)
        for metric in metrics:
            if metric == 'Carbon_Footprint':
                value = random.uniform(200, 800)  # kg CO2
                target = 400
                unit = 'kg_CO2'
            elif metric == 'Energy_Efficiency':
                value = random.uniform(0.6, 0.9)  # efficiency ratio
                target = 0.85
                unit = 'ratio'
            elif metric == 'Renewable_Usage':
                value = random.uniform(0.2, 0.8)  # percentage
                target = 0.7
                unit = 'percentage'
            elif metric == 'Resource_Utilization':
                value = random.uniform(0.5, 0.85)  # percentage
                target = 0.8
                unit = 'percentage'
            elif metric == 'Cooling_Efficiency':
                value = random.uniform(2.5, 4.0)  # COP
                target = 3.5
                unit = 'COP'
            else:  # Waste_Heat_Recovery
                value = random.uniform(0.1, 0.4)  # percentage
                target = 0.3
                unit = 'percentage'
            
            data.append({
                'date': date,
                'metric_name': metric,
                'value': round(value, 3),
                'target_value': target,
                'unit': unit,
                'performance_score': round(min(value/target, 1.0) * 100, 1)
            })
    
    return pd.DataFrame(data)

# Generate and save all datasets
if __name__ == "__main__":
    print("Creating NeuroSimGreen dummy datasets...")

    # Generate the datasets
    datacenter_df = create_datacenter_infrastructure()
    energy_df = create_energy_carbon_data()
    workload_df = create_workload_data()
    performance_df = create_performance_data()
    sustainability_df = create_sustainability_metrics()

    # Save to CSV files
    datacenter_df.to_csv('datacenter_infrastructure.csv', index=False)
    energy_df.to_csv('energy_carbon_data.csv', index=False)
    workload_df.to_csv('workload_data.csv', index=False)
    performance_df.to_csv('historical_performance.csv', index=False)
    sustainability_df.to_csv('sustainability_metrics.csv', index=False)

    print("Dataset creation completed!")
    print(f"Data Center Infrastructure: {len(datacenter_df)} records")
    print(f"Energy & Carbon Data: {len(energy_df)} records")
    print(f"Workload Data: {len(workload_df)} records")
    print(f"Historical Performance: {len(performance_df)} records")
    print(f"Sustainability Metrics: {len(sustainability_df)} records")
