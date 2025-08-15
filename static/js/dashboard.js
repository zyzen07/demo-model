// NeuroSimGreen Intelligent Server Allocation Dashboard
class IntelligentAllocationDashboard {
    constructor() {
        this.charts = {};
        this.serverData = [];
        this.initializeEventListeners();
        this.loadDashboardData();
        this.loadModelPerformance();
        this.startRealTimeUpdates();
    }

    initializeEventListeners() {
        // Task allocation form
        document.getElementById('taskForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitTaskAllocation();
        });

        // Task type change event to update default values
        document.getElementById('taskType').addEventListener('change', (e) => {
            this.updateTaskDefaults(e.target.value);
        });

        // Server refresh button
        document.getElementById('refreshServers').addEventListener('click', () => {
            this.loadServerComparison();
        });

        // Simulation button
        document.getElementById('runSimulation').addEventListener('click', () => {
            this.runAllocationSimulation();
        });
    }

    updateTaskDefaults(taskType) {
        const defaults = {
            'Web-Request': { cpu: 5, ram: 1, disk: 0.1, burst: 2 },
            'Data-Analysis': { cpu: 30, ram: 8, disk: 2, burst: 180 },
            'ML-Training': { cpu: 50, ram: 32, disk: 10, burst: 900 },
            'File-Processing': { cpu: 20, ram: 4, disk: 5, burst: 60 },
            'Database-Query': { cpu: 15, ram: 2, disk: 0.5, burst: 15 },
            'Video-Processing': { cpu: 60, ram: 16, disk: 25, burst: 600 }
        };

        const values = defaults[taskType] || defaults['Web-Request'];
        document.getElementById('cpuRequirement').value = values.cpu;
        document.getElementById('ramRequirement').value = values.ram;
        document.getElementById('diskRequirement').value = values.disk;
        document.getElementById('burstTime').value = values.burst;
    }

    async loadDashboardData() {
        try {
            const response = await fetch('/api/dashboard-data');
            const data = await response.json();

            if (data.success) {
                this.updateDashboardMetrics(data.data);
                this.createCharts(data.data);
            }
        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }

    updateDashboardMetrics(data) {
        const serverMetrics = data.server_metrics;
        const envMetrics = data.environmental_metrics;

        document.getElementById('total-servers').textContent = serverMetrics.total_servers;
        document.getElementById('active-servers').textContent = serverMetrics.active_servers;
        document.getElementById('avg-cpu-available').textContent = `${serverMetrics.avg_cpu_available_pct}%`;
        document.getElementById('avg-temperature').textContent = `${serverMetrics.avg_temperature_c}°C`;
        document.getElementById('carbon-intensity').textContent = `${envMetrics.current_carbon_intensity}g`;
        document.getElementById('active-threads').textContent = serverMetrics.total_active_threads;
    }

    createCharts(data) {
        // Server Load Distribution Chart
        const loadCtx = document.getElementById('serverLoadChart').getContext('2d');
        const serverIds = Object.keys(data.server_status);
        const loadData = serverIds.map(id => data.server_status[id].current_load_pct);

        this.charts.serverLoadChart = new Chart(loadCtx, {
            type: 'bar',
            data: {
                labels: serverIds,
                datasets: [{
                    label: 'Current Load (%)',
                    data: loadData,
                    backgroundColor: loadData.map(load => 
                        load < 30 ? '#28a745' : 
                        load < 70 ? '#ffc107' : '#dc3545'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Load Percentage'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Current Server Load Distribution'
                    }
                }
            }
        });

        // Temperature Chart
        const tempCtx = document.getElementById('temperatureChart').getContext('2d');
        const tempData = serverIds.map(id => data.server_status[id].temperature_c);

        this.charts.temperatureChart = new Chart(tempCtx, {
            type: 'line',
            data: {
                labels: serverIds,
                datasets: [{
                    label: 'Temperature (°C)',
                    data: tempData,
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Temperature (°C)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Server Temperature Monitoring'
                    }
                }
            }
        });
    }

    async submitTaskAllocation() {
        this.showLoadingModal();
        
        try {
            const taskData = {
                task_type: document.getElementById('taskType').value,
                cpu_requirement_pct: parseFloat(document.getElementById('cpuRequirement').value),
                ram_requirement_gb: parseFloat(document.getElementById('ramRequirement').value),
                disk_requirement_gb: parseFloat(document.getElementById('diskRequirement').value),
                estimated_burst_time_sec: parseFloat(document.getElementById('burstTime').value),
                priority: document.getElementById('priority').value
            };

            const response = await fetch('/api/allocate-task', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(taskData)
            });

            const result = await response.json();
            
            if (result.success) {
                this.displayAllocationResult(result);
                this.showSuccess('Task allocated successfully!');
                this.loadServerComparison(); // Refresh server status
            } else {
                this.showError('Task allocation failed: ' + result.error);
            }
        } catch (error) {
            console.error('Error allocating task:', error);
            this.showError('Failed to allocate task');
        } finally {
            this.hideLoadingModal();
        }
    }

    displayAllocationResult(result) {
        const allocationDiv = document.getElementById('allocationResult');
        const allocation = result.allocation_result;
        
        let html = `
            <div class="alert alert-info">
                <h6><i class="fas fa-info-circle"></i> Task ID: ${result.task_id}</h6>
            </div>
        `;
        
        if (allocation.recommended_server) {
            html += `
                <div class="alert alert-success">
                    <h5><i class="fas fa-check-circle"></i> Recommended Server: ${allocation.recommended_server}</h5>
                    <p><strong>Confidence Score:</strong> ${allocation.confidence_score.toFixed(2)}/100</p>
                </div>
            `;
            
            if (allocation.processing_result) {
                html += `
                    <div class="mt-3">
                        <h6>Processing Results:</h6>
                        <ul class="list-unstyled">
                            <li><strong>Processing Time:</strong> ${allocation.processing_result.processing_time_sec.toFixed(2)} seconds</li>
                            <li><strong>Energy Consumed:</strong> ${allocation.processing_result.energy_consumed_wh.toFixed(2)} Wh</li>
                            <li><strong>Heat Generated:</strong> ${allocation.processing_result.heat_generated_j.toFixed(0)} J</li>
                            <li><strong>Status:</strong> <span class="badge bg-success">${allocation.processing_result.status}</span></li>
                        </ul>
                    </div>
                `;
            }
        } else {
            html += `
                <div class="alert alert-warning">
                    <h5><i class="fas fa-exclamation-triangle"></i> No Suitable Server Found</h5>
                    <p>All servers are currently overloaded or unable to handle this task.</p>
                </div>
            `;
        }
        
        // Show top 3 servers with scores
        const topServers = Object.entries(allocation.all_server_scores)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 3);
        
        if (topServers.length > 0) {
            html += `
                <div class="mt-3">
                    <h6>Top Server Candidates:</h6>
                    <div class="row">
            `;
            
            topServers.forEach(([serverId, score], index) => {
                const badgeClass = index === 0 ? 'bg-success' : score > 50 ? 'bg-primary' : 'bg-secondary';
                html += `
                    <div class="col-md-4">
                        <div class="card mb-2">
                            <div class="card-body text-center">
                                <h6>${serverId}</h6>
                                <span class="badge ${badgeClass}">${score.toFixed(1)}</span>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        }
        
        allocationDiv.innerHTML = html;
    }

    async loadServerComparison() {
        try {
            const response = await fetch('/api/server-comparison');
            const data = await response.json();

            if (data.success) {
                this.displayServerComparison(data.servers);
            }
        } catch (error) {
            console.error('Error loading server comparison:', error);
        }
    }

    displayServerComparison(servers) {
        const tableDiv = document.getElementById('serverComparisonTable');
        
        let html = `
            <div class="table-responsive">
                <table class="table table-striped table-hover">
                    <thead class="table-dark">
                        <tr>
                            <th>Server ID</th>
                            <th>Specs (CPU/RAM/Disk)</th>
                            <th>Available Resources</th>
                            <th>Current Load</th>
                            <th>Temperature</th>
                            <th>AI Score</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        servers.forEach(server => {
            const statusClass = server.recommendation === 'Recommended' ? 'table-success' : 
                               server.recommendation === 'Available' ? '' : 'table-warning';
            
            const scoreColor = server.allocation_score > 70 ? 'text-success' : 
                              server.allocation_score > 40 ? 'text-warning' : 'text-danger';
            
            html += `
                <tr class="${statusClass}">
                    <td><strong>${server.server_id}</strong></td>
                    <td>${server.cpu_cores}C / ${server.ram_gb}GB / ${server.disk_tb}TB</td>
                    <td>
                        CPU: ${server.cpu_available_pct}%<br>
                        RAM: ${server.ram_available_gb.toFixed(1)}GB
                    </td>
                    <td>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar ${server.current_load_pct > 80 ? 'bg-danger' : server.current_load_pct > 50 ? 'bg-warning' : 'bg-success'}" 
                                 style="width: ${server.current_load_pct}%">
                                ${server.current_load_pct.toFixed(1)}%
                            </div>
                        </div>
                    </td>
                    <td>${server.temperature_c.toFixed(1)}°C</td>
                    <td class="${scoreColor}"><strong>${server.allocation_score.toFixed(1)}</strong></td>
                    <td><span class="badge ${server.recommendation === 'Recommended' ? 'bg-success' : server.recommendation === 'Available' ? 'bg-primary' : 'bg-warning'}">${server.recommendation}</span></td>
                </tr>
            `;
        });
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
        
        tableDiv.innerHTML = html;
    }

    async loadModelPerformance() {
        try {
            const response = await fetch('/api/model-performance');
            const data = await response.json();

            if (data.success) {
                this.displayModelPerformance(data.performance, data.model_info);
                this.createFeatureImportanceChart(data.performance.feature_importance_ranked);
            }
        } catch (error) {
            console.error('Error loading model performance:', error);
        }
    }

    displayModelPerformance(performance, modelInfo) {
        const performanceDiv = document.getElementById('modelPerformance');
        
        const html = `
            <div class="row">
                <div class="col-md-6">
                    <div class="card bg-primary text-white mb-3">
                        <div class="card-body text-center">
                            <h5>R² Score</h5>
                            <h3>${performance.r2_score.toFixed(4)}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card bg-info text-white mb-3">
                        <div class="card-body text-center">
                            <h5>MSE</h5>
                            <h3>${performance.mse.toFixed(2)}</h3>
                        </div>
                    </div>
                </div>
            </div>
            <div class="mt-3">
                <h6>Model Information:</h6>
                <ul class="list-unstyled">
                    <li><strong>Algorithm:</strong> ${modelInfo.algorithm}</li>
                    <li><strong>Estimators:</strong> ${modelInfo.n_estimators}</li>
                    <li><strong>Max Depth:</strong> ${modelInfo.max_depth}</li>
                    <li><strong>Training Samples:</strong> ${modelInfo.training_samples}</li>
                </ul>
            </div>
        `;
        
        performanceDiv.innerHTML = html;
    }

    createFeatureImportanceChart(featureImportance) {
        const ctx = document.getElementById('featureImportanceChart').getContext('2d');
        
        if (this.charts.featureImportanceChart) {
            this.charts.featureImportanceChart.destroy();
        }
        
        const features = featureImportance.map(item => item[0].replace(/_/g, ' '));
        const importance = featureImportance.map(item => item[1]);
        
        this.charts.featureImportanceChart = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: features,
                datasets: [{
                    label: 'Feature Importance',
                    data: importance,
                    backgroundColor: '#28a745',
                    borderColor: '#1e7e34',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Importance Score'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Random Forest Feature Importance'
                    }
                }
            }
        });
    }

    async runAllocationSimulation() {
        this.showLoadingModal();
        
        try {
            const simulationData = {
                num_tasks: parseInt(document.getElementById('numTasks').value),
                duration: parseInt(document.getElementById('simulationDuration').value)
            };

            const response = await fetch('/api/run-simulation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(simulationData)
            });

            const result = await response.json();
            
            if (result.success) {
                this.displaySimulationResults(result.results);
                this.showSuccess('Simulation completed successfully!');
            } else {
                this.showError('Simulation failed: ' + result.error);
            }
        } catch (error) {
            console.error('Error running simulation:', error);
            this.showError('Failed to run simulation');
        } finally {
            this.hideLoadingModal();
        }
    }

    displaySimulationResults(results) {
        const resultsDiv = document.getElementById('simulationResults');
        
        const html = `
            <div class="row">
                <div class="col-md-3">
                    <div class="card bg-success text-white">
                        <div class="card-body text-center">
                            <h5>Tasks Processed</h5>
                            <h3>${results.processed_tasks}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-warning text-white">
                        <div class="card-body text-center">
                            <h5>Total Energy (Wh)</h5>
                            <h3>${results.total_energy_consumed.toFixed(2)}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-info text-white">
                        <div class="card-body text-center">
                            <h5>Total Time</h5>
                            <h3>${results.total_processing_time.toFixed(1)}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-primary text-white">
                        <div class="card-body text-center">
                            <h5>Avg Energy/Task</h5>
                            <h3>${results.average_energy_per_task.toFixed(2)} Wh</h3>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
    }

    startRealTimeUpdates() {
        // Update dashboard every 30 seconds
        setInterval(() => {
            this.loadDashboardData();
        }, 30000);

        // Update server comparison every 15 seconds
        setInterval(() => {
            this.loadServerComparison();
        }, 15000);
    }

    showLoadingModal() {
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
    }

    hideLoadingModal() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) {
            modal.hide();
        }
    }

    showSuccess(message) {
        console.log('Success:', message);
        // You can implement toast notifications here
    }

    showError(message) {
        console.error('Error:', message);
        alert('Error: ' + message);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new IntelligentAllocationDashboard();
});
