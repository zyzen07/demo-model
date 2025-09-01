// High-Performance NeuroSimGreen Dashboard with Complete Functionality
class OptimizedNeuroSimGreenDashboard {
    constructor() {
        this.charts = {};
        this.serverData = [];
        this.firebaseReady = false;
        this.updateInterval = null;
        this.cache = new Map();
        this.loadingStates = new Set();
        this.requestController = new AbortController();
        this.performanceObserver = null;
        this.isDestroyed = false;
        
        this.initializeApp();
    }

    async initializeApp() {
        console.time('Dashboard Initialization');
        
        // Show immediate feedback
        this.showLoadingSkeletons();
        this.updateSystemStatus('Initializing high-performance dashboard...');
        
        // Initialize performance monitoring
        this.initializePerformanceMonitoring();
        
        // Wait for Firebase
        await this.waitForFirebase();
        
        // Initialize components
        this.initializeEventListeners();
        this.initializeTooltips();
        
        // Load data with priorities
        await this.loadCriticalData();
        this.loadNonCriticalDataAsync();
        
        // Setup real-time updates
        this.setupOptimizedRealTimeUpdates();
        
        console.timeEnd('Dashboard Initialization');
        this.showToast('High-performance dashboard ready!', 'success');
    }

    initializePerformanceMonitoring() {
        if ('PerformanceObserver' in window) {
            this.performanceObserver = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                entries.forEach((entry) => {
                    if (entry.duration > 1000) {
                        console.warn(`Slow operation: ${entry.name} - ${entry.duration.toFixed(2)}ms`);
                    }
                });
            });
            this.performanceObserver.observe({entryTypes: ['measure', 'navigation']});
        }
    }

    showLoadingSkeletons() {
        // Server comparison skeleton
        const serverTableDiv = document.getElementById('serverComparisonTable');
        if (serverTableDiv) {
            serverTableDiv.innerHTML = `
                <div class="skeleton-loader">
                    ${Array(8).fill(0).map(() => `
                        <div class="d-flex justify-content-between align-items-center mb-2 p-2 border rounded">
                            <div class="skeleton skeleton-text" style="width: 80px;"></div>
                            <div class="skeleton skeleton-text" style="width: 60px;"></div>
                            <div class="skeleton skeleton-text" style="width: 100px;"></div>
                            <div class="skeleton skeleton-text" style="width: 40px;"></div>
                            <div class="skeleton skeleton-text" style="width: 60px;"></div>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // Model performance skeleton
        const modelDiv = document.getElementById('modelPerformance');
        if (modelDiv) {
            modelDiv.innerHTML = `
                <div class="skeleton-loader">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="skeleton skeleton-card mb-2" style="height: 80px;"></div>
                        </div>
                        <div class="col-md-6">
                            <div class="skeleton skeleton-card mb-2" style="height: 80px;"></div>
                        </div>
                    </div>
                    <div class="mt-2">
                        <div class="skeleton skeleton-text" style="width: 70%;"></div>
                        <div class="skeleton skeleton-text" style="width: 50%;"></div>
                    </div>
                </div>
            `;
        }

        // Simulation results skeleton
        const simDiv = document.getElementById('simulationResults');
        if (simDiv) {
            simDiv.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i> 
                    <strong>Ready:</strong> Configure parameters and run simulation.
                </div>
            `;
        }
    }

    async waitForFirebase() {
        let attempts = 0;
        while (!window.neuroFirebase && attempts < 10) {
            await new Promise(resolve => setTimeout(resolve, 500));
            attempts++;
        }

        if (window.neuroFirebase) {
            this.firebase = window.neuroFirebase;
            this.firebaseReady = true;
            this.updateFirebaseStatus(true);
            console.log('🔥 Firebase integration ready!');
        } else {
            console.warn('⚠️ Firebase not available - continuing without cloud integration');
            this.updateFirebaseStatus(false);
        }
    }

    // Optimized caching system
    getFromCache(key, maxAge = 5000) {
        const cached = this.cache.get(key);
        if (cached && (Date.now() - cached.timestamp) < maxAge) {
            return cached.data;
        }
        return null;
    }

    setCache(key, data) {
        this.cache.set(key, {
            data: data,
            timestamp: Date.now()
        });
        
        // Cleanup old cache entries
        if (this.cache.size > 50) {
            const oldestKey = Array.from(this.cache.keys())[0];
            this.cache.delete(oldestKey);
        }
    }

    // Optimized API requests
    async makeOptimizedRequest(url, options = {}) {
        try {
            const response = await fetch(url, {
                ...options,
                signal: this.requestController.signal
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Request aborted');
                return null;
            }
            throw error;
        }
    }

    async loadCriticalData() {
        console.time('Critical Data Loading');
        
        try {
            await this.loadDashboardDataOptimized();
        } catch (error) {
            console.error('Error loading critical data:', error);
            this.showToast('Error loading critical data', 'error');
        }
        
        console.timeEnd('Critical Data Loading');
    }

    loadNonCriticalDataAsync() {
        // Load with progressive delays to prevent blocking
        const loadTasks = [
            { fn: () => this.loadServerComparisonOptimized(), delay: 100 },
            { fn: () => this.loadModelPerformanceOptimized(), delay: 200 }
        ];

        for (const task of loadTasks) {
            setTimeout(async () => {
                try {
                    await task.fn();
                } catch (error) {
                    console.error('Error in non-critical loading:', error);
                }
            }, task.delay);
        }
    }

    initializeEventListeners() {
        // Task allocation form
        const taskForm = document.getElementById('taskForm');
        if (taskForm) {
            taskForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.submitTaskAllocation();
            });
        }

        // Real-time input updates
        this.setupInputUpdates();

        // Task type change
        const taskType = document.getElementById('taskType');
        if (taskType) {
            taskType.addEventListener('change', (e) => {
                this.updateTaskDefaults(e.target.value);
            });
        }

        // Server management
        const refreshServers = document.getElementById('refreshServers');
        if (refreshServers) {
            refreshServers.addEventListener('click', () => {
                this.loadServerComparisonOptimized();
            });
        }

        // Server filtering
        document.querySelectorAll('input[name="serverFilter"]').forEach(radio => {
            radio.addEventListener('change', () => {
                this.filterServers();
            });
        });

        // Simulation
        const runSimulation = document.getElementById('runSimulation');
        if (runSimulation) {
            runSimulation.addEventListener('click', () => {
                this.runOptimizedSimulation();
            });
        }

        // Firebase integration
        this.setupFirebaseEventListeners();
    }

    setupInputUpdates() {
        const inputs = [
            { id: 'cpuRequirement', display: 'cpuReqDisplay', suffix: '%' },
            { id: 'ramRequirement', display: 'ramReqDisplay', suffix: ' GB' },
            { id: 'diskRequirement', display: 'diskReqDisplay', suffix: ' GB' },
            { id: 'burstTime', display: 'burstTimeDisplay', formatter: this.formatTime }
        ];

        inputs.forEach(({ id, display, suffix, formatter }) => {
            const element = document.getElementById(id);
            const displayElement = document.getElementById(display);
            
            if (element && displayElement) {
                element.addEventListener('input', (e) => {
                    const value = parseFloat(e.target.value);
                    if (formatter) {
                        displayElement.textContent = formatter(value);
                    } else {
                        displayElement.textContent = value + suffix;
                    }
                });
            }
        });
    }

    formatTime(seconds) {
        if (seconds < 60) {
            return seconds + ' seconds';
        } else if (seconds < 3600) {
            return Math.round(seconds / 60 * 10) / 10 + ' minutes';
        } else {
            return Math.round(seconds / 3600 * 10) / 10 + ' hours';
        }
    }

    setupFirebaseEventListeners() {
        const firebaseButtons = [
            { id: 'saveServersToFirebase', handler: () => this.saveServersToFirebase() },
            { id: 'saveTrainingDataToFirebase', handler: () => this.saveTrainingDataToFirebase() },
            { id: 'saveAllToFirebase', handler: () => this.saveAllDataToFirebase() },
            { id: 'loadServersFromFirebase', handler: () => this.loadServersFromFirebase() },
            { id: 'loadAllocationsFromFirebase', handler: () => this.loadAllocationsFromFirebase() },
            { id: 'loadAllFromFirebase', handler: () => this.loadAllFromFirebase() }
        ];

        firebaseButtons.forEach(({ id, handler }) => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('click', handler);
            }
        });
    }

    initializeTooltips() {
        if (typeof bootstrap !== 'undefined') {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
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
        
        const updates = [
            { id: 'cpuRequirement', value: values.cpu },
            { id: 'ramRequirement', value: values.ram },
            { id: 'diskRequirement', value: values.disk },
            { id: 'burstTime', value: values.burst }
        ];

        updates.forEach(({ id, value }) => {
            const element = document.getElementById(id);
            if (element) {
                element.value = value;
                element.dispatchEvent(new Event('input'));
            }
        });
    }

    async loadDashboardDataOptimized() {
        const cacheKey = 'dashboard-data';
        const cachedData = this.getFromCache(cacheKey, 8000);
        
        if (cachedData) {
            this.updateDashboardMetrics(cachedData);
            this.createOptimizedCharts(cachedData);
            return;
        }

        try {
            const data = await this.makeOptimizedRequest('/api/dashboard-data');
            
            if (data && data.success) {
                this.setCache(cacheKey, data.data);
                this.updateDashboardMetrics(data.data);
                this.createOptimizedCharts(data.data);
            }
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showErrorState('dashboard');
        }
    }

    updateDashboardMetrics(data) {
        const serverMetrics = data.server_metrics || {};
        const envMetrics = data.environmental_metrics || {};
        const systemMetrics = data.system_metrics || {};

        // Batch DOM updates for better performance
        const updates = [
            ['total-servers', serverMetrics.total_servers || 15],
            ['active-servers', serverMetrics.active_servers || 0],
            ['avg-cpu-available', `${serverMetrics.avg_cpu_available_pct || 0}%`],
            ['avg-temperature', `${serverMetrics.avg_temperature_c || 0}°C`],
            ['carbon-intensity', envMetrics.current_carbon_intensity || 0],
            ['active-threads', serverMetrics.total_active_threads || 0],
            ['total-power', `${serverMetrics.total_power_consumption_kw || 0} kW`],
            ['total-tasks-processed', systemMetrics.total_tasks_processed || 0],
            ['system-uptime', `${systemMetrics.system_uptime_hours || 0} hrs`],
            ['carbon-emissions', `${envMetrics.estimated_carbon_emissions_kg_per_hour || 0} kg/hr`]
        ];

        // Use requestAnimationFrame for smooth updates
        requestAnimationFrame(() => {
            updates.forEach(([id, value]) => {
                const element = document.getElementById(id);
                if (element) element.textContent = value;
            });
        });
    }

    createOptimizedCharts(data) {
        if (!data.server_status) return;
        
        const serverIds = Object.keys(data.server_status).slice(0, 10);
        
        // Use requestIdleCallback for non-critical chart creation
        if ('requestIdleCallback' in window) {
            requestIdleCallback(() => {
                this.createServerLoadChartOptimized(data, serverIds);
                this.createTemperatureChartOptimized(data, serverIds);
            });
            
            requestIdleCallback(() => {
                this.createPowerConsumptionChartOptimized(data, serverIds);
                this.createServerTierChartOptimized(data);
            });
        } else {
            setTimeout(() => {
                this.createServerLoadChartOptimized(data, serverIds);
                this.createTemperatureChartOptimized(data, serverIds);
            }, 50);
            
            setTimeout(() => {
                this.createPowerConsumptionChartOptimized(data, serverIds);
                this.createServerTierChartOptimized(data);
            }, 100);
        }
    }

    createServerLoadChartOptimized(data, serverIds) {
        const ctx = document.getElementById('serverLoadChart')?.getContext('2d');
        if (!ctx) return;

        const loadData = serverIds.map(id => data.server_status[id]?.current_load_pct || 0);

        if (this.charts.serverLoadChart) {
            // Update existing chart instead of destroying
            this.charts.serverLoadChart.data.labels = serverIds;
            this.charts.serverLoadChart.data.datasets[0].data = loadData;
            this.charts.serverLoadChart.data.datasets.backgroundColor = loadData.map(load => 
                load < 30 ? '#28a745' : load < 70 ? '#ffc107' : '#dc3545'
            );
            this.charts.serverLoadChart.update('none');
            return;
        }

        this.charts.serverLoadChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: serverIds,
                datasets: [{
                    label: 'Load (%)',
                    data: loadData,
                    backgroundColor: loadData.map(load => 
                        load < 30 ? '#28a745' : load < 70 ? '#ffc107' : '#dc3545'
                    ),
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                interaction: { intersect: false },
                scales: {
                    y: { 
                        beginAtZero: true, 
                        max: 100,
                        ticks: { maxTicksLimit: 5 }
                    }
                },
                plugins: {
                    title: { display: true, text: 'Server Load Distribution (Top 10)' },
                    legend: { display: false }
                }
            }
        });
    }

    createTemperatureChartOptimized(data, serverIds) {
        const ctx = document.getElementById('temperatureChart')?.getContext('2d');
        if (!ctx) return;

        const tempData = serverIds.map(id => data.server_status[id]?.temperature_c || 25);

        if (this.charts.temperatureChart) {
            this.charts.temperatureChart.data.labels = serverIds;
            this.charts.temperatureChart.data.datasets[0].data = tempData;
            this.charts.temperatureChart.update('none');
            return;
        }

        this.charts.temperatureChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: serverIds,
                datasets: [{
                    label: 'Temperature (°C)',
                    data: tempData,
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                interaction: { intersect: false },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 20,
                        ticks: { maxTicksLimit: 6 }
                    }
                },
                plugins: {
                    title: { display: true, text: 'Temperature Monitoring' }
                }
            }
        });
    }

    createPowerConsumptionChartOptimized(data, serverIds) {
        const ctx = document.getElementById('powerChart')?.getContext('2d');
        if (!ctx) return;

        const powerData = serverIds.map(id => data.server_status[id]?.power_consumption_watts || 0);

        if (this.charts.powerChart) {
            this.charts.powerChart.data.labels = serverIds;
            this.charts.powerChart.data.datasets[0].data = powerData;
            this.charts.powerChart.update('none');
            return;
        }

        this.charts.powerChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: serverIds,
                datasets: [{
                    label: 'Power (W)',
                    data: powerData,
                    backgroundColor: '#17a2b8',
                    borderColor: '#117a8b',
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                scales: {
                    y: { 
                        beginAtZero: true,
                        ticks: { maxTicksLimit: 5 }
                    }
                },
                plugins: {
                    title: { display: true, text: 'Power Consumption' },
                    legend: { display: false }
                }
            }
        });
    }

    createServerTierChartOptimized(data) {
        const ctx = document.getElementById('serverTierChart')?.getContext('2d');
        if (!ctx) return;
        
        const tierCounts = {};
        Object.values(data.server_status).forEach(status => {
            const tier = status.server_tier || 'Standard';
            tierCounts[tier] = (tierCounts[tier] || 0) + 1;
        });

        if (this.charts.serverTierChart) {
            this.charts.serverTierChart.data.labels = Object.keys(tierCounts);
            this.charts.serverTierChart.data.datasets[0].data = Object.values(tierCounts);
            this.charts.serverTierChart.update('none');
            return;
        }

        this.charts.serverTierChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(tierCounts),
                datasets: [{
                    data: Object.values(tierCounts),
                    backgroundColor: ['#28a745', '#ffc107', '#dc3545', '#6c757d']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: {
                    title: { display: true, text: 'Server Tier Distribution' },
                    legend: { position: 'bottom' }
                }
            }
        });
    }

    async loadServerComparisonOptimized() {
        if (this.loadingStates.has('servers')) return;
        this.loadingStates.add('servers');

        const cacheKey = 'server-comparison';
        const cachedData = this.getFromCache(cacheKey, 6000);
        
        if (cachedData) {
            this.displayServerComparisonOptimized(cachedData.servers);
            this.loadingStates.delete('servers');
            return;
        }

        try {
            const data = await this.makeOptimizedRequest('/api/server-comparison');
            
            if (data && data.success) {
                this.setCache(cacheKey, data);
                this.displayServerComparisonOptimized(data.servers);
            }
        } catch (error) {
            console.error('Error loading server comparison:', error);
            this.showErrorState('servers');
        } finally {
            this.loadingStates.delete('servers');
        }
    }

    displayServerComparisonOptimized(servers) {
        const tableDiv = document.getElementById('serverComparisonTable');
        if (!tableDiv) return;

        // Use DocumentFragment for better performance
        const fragment = document.createDocumentFragment();
        const tableContainer = document.createElement('div');
        tableContainer.className = 'table-responsive';
        
        const table = document.createElement('table');
        table.className = 'table table-striped table-hover table-sm';
        
        table.innerHTML = `
            <thead class="table-dark">
                <tr>
                    <th>Server</th>
                    <th>Tier</th>
                    <th>Resources</th>
                    <th>Load</th>
                    <th>Temp</th>
                    <th>AI Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody></tbody>
        `;
        
        const tbody = table.querySelector('tbody');
        
        servers.slice(0, 10).forEach(server => {
            const row = document.createElement('tr');
            const statusClass = server.recommendation === 'Recommended' ? 'table-success' : 
                               server.recommendation === 'Available' ? '' : 'table-warning';
            row.className = statusClass;
            
            const scoreColor = server.allocation_score > 70 ? 'text-success' : 
                              server.allocation_score > 40 ? 'text-warning' : 'text-danger';
            
            row.innerHTML = `
                <td><strong>${server.server_id}</strong></td>
                <td><span class="badge bg-secondary">${server.server_tier || 'Standard'}</span></td>
                <td>
                    <small>
                        ${server.cpu_cores}C/${server.ram_gb}GB<br>
                        CPU: ${server.cpu_available_pct.toFixed(1)}%
                    </small>
                </td>
                <td>
                    <div class="progress" style="height: 15px; width: 60px;">
                        <div class="progress-bar ${server.current_load_pct > 80 ? 'bg-danger' : server.current_load_pct > 50 ? 'bg-warning' : 'bg-success'}" 
                             style="width: ${server.current_load_pct}%"></div>
                    </div>
                    <small>${server.current_load_pct.toFixed(1)}%</small>
                </td>
                <td>${server.temperature_c.toFixed(1)}°C</td>
                <td class="${scoreColor}"><strong>${server.allocation_score.toFixed(1)}</strong></td>
                <td><span class="badge ${server.recommendation === 'Recommended' ? 'bg-success' : server.recommendation === 'Available' ? 'bg-primary' : 'bg-warning'}">${server.recommendation}</span></td>
            `;
            
            tbody.appendChild(row);
        });
        
        tableContainer.appendChild(table);
        fragment.appendChild(tableContainer);
        
        requestAnimationFrame(() => {
            tableDiv.innerHTML = '';
            tableDiv.appendChild(fragment);
        });
    }

    filterServers() {
        const activeFilter = document.querySelector('input[name="serverFilter"]:checked')?.id;
        const rows = document.querySelectorAll('#serverComparisonTable tbody tr');
        
        rows.forEach(row => {
            const tierBadge = row.querySelector('.badge');
            const tier = tierBadge?.textContent || 'Standard';
            
            let show = true;
            if (activeFilter === 'filterHighEnd') {
                show = tier === 'High-End';
            } else if (activeFilter === 'filterMidRange') {
                show = tier === 'Mid-Range';
            } else if (activeFilter === 'filterStandard') {
                show = tier === 'Standard';
            }
            
            row.style.display = show ? '' : 'none';
        });
    }

    async loadModelPerformanceOptimized() {
        if (this.loadingStates.has('model')) return;
        this.loadingStates.add('model');

        const cacheKey = 'model-performance';
        const cachedData = this.getFromCache(cacheKey, 30000);
        
        if (cachedData) {
            this.displayModelPerformanceOptimized(cachedData.performance, cachedData.model_info);
            this.createFeatureImportanceChartOptimized(cachedData.performance.feature_importance_ranked);
            this.loadingStates.delete('model');
            return;
        }

        try {
            const data = await this.makeOptimizedRequest('/api/model-performance');
            
            if (data && data.success) {
                this.setCache(cacheKey, data);
                this.displayModelPerformanceOptimized(data.performance, data.model_info);
                this.createFeatureImportanceChartOptimized(data.performance.feature_importance_ranked);
            }
        } catch (error) {
            console.error('Error loading model performance:', error);
            this.showErrorState('model');
        } finally {
            this.loadingStates.delete('model');
        }
    }

    displayModelPerformanceOptimized(performance, modelInfo) {
        const performanceDiv = document.getElementById('modelPerformance');
        if (!performanceDiv) return;
        
        const html = `
            <div class="row">
                <div class="col-md-6">
                    <div class="card bg-primary text-white mb-2">
                        <div class="card-body text-center p-2">
                            <h6>R² Score</h6>
                            <h4>${performance.r2_score.toFixed(4)}</h4>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card bg-info text-white mb-2">
                        <div class="card-body text-center p-2">
                            <h6>MSE</h6>
                            <h4>${performance.mse.toFixed(2)}</h4>
                        </div>
                    </div>
                </div>
            </div>
            <div class="mt-2">
                <small>
                    <strong>Algorithm:</strong> ${modelInfo.algorithm}<br>
                    <strong>Training Samples:</strong> ${modelInfo.training_samples.toLocaleString()}<br>
                    <strong>Features:</strong> ${modelInfo.feature_count}
                </small>
            </div>
        `;
        
        performanceDiv.innerHTML = html;
    }

    createFeatureImportanceChartOptimized(featureImportance) {
        const ctx = document.getElementById('featureImportanceChart')?.getContext('2d');
        if (!ctx || !featureImportance) return;
        
        if (this.charts.featureImportanceChart) {
            this.charts.featureImportanceChart.destroy();
        }
        
        const topFeatures = featureImportance.slice(0, 8);
        const features = topFeatures.map(item => item[0].replace(/_/g, ' ').substring(0, 15));
        const importance = topFeatures.map(item => item[1]);
        
        this.charts.featureImportanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features,
                datasets: [{
                    label: 'Importance',
                    data: importance,
                    backgroundColor: '#28a745',
                    borderColor: '#1e7e34',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                animation: { duration: 800 },
                scales: {
                    x: {
                        beginAtZero: true,
                        title: { display: true, text: 'Importance' }
                    }
                },
                plugins: {
                    title: { display: true, text: 'Top Feature Importance' }
                }
            }
        });
    }

    async submitTaskAllocation() {
        this.showLoadingModal('Allocating Task', 'AI is analyzing optimal server allocation...');
        this.updateLoadingProgress(20);
        
        try {
            const taskData = {
                task_type: document.getElementById('taskType')?.value || 'Web-Request',
                cpu_requirement_pct: parseFloat(document.getElementById('cpuRequirement')?.value || 25),
                ram_requirement_gb: parseFloat(document.getElementById('ramRequirement')?.value || 4),
                disk_requirement_gb: parseFloat(document.getElementById('diskRequirement')?.value || 1),
                estimated_burst_time_sec: parseFloat(document.getElementById('burstTime')?.value || 120),
                priority: document.getElementById('priority')?.value || 'Medium'
            };

            this.updateLoadingProgress(50);

            const result = await this.makeOptimizedRequest('/api/allocate-task', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(taskData)
            });

            this.updateLoadingProgress(70);

            if (result && result.success) {
                this.displayEnhancedAllocationResult(result);
                
                // Save to Firebase if available
                if (this.firebaseReady) {
                    await this.firebase.addData('neurosim_tasks', {
                        task_id: result.task_id,
                        task_data: taskData,
                        allocation_result: result.allocation_result,
                        timestamp: new Date().toISOString()
                    });
                }
                
                this.updateLoadingProgress(100);
                this.showToast('Task allocated successfully with AI optimization!', 'success');
                
                // Refresh server status
                setTimeout(() => this.loadServerComparisonOptimized(), 1000);
            } else {
                throw new Error(result?.error || 'Unknown error');
            }
        } catch (error) {
            console.error('Error allocating task:', error);
            this.showToast('Task allocation failed: ' + error.message, 'error');
        } finally {
            this.hideLoadingModal();
        }
    }

    displayEnhancedAllocationResult(result) {
        const allocationDiv = document.getElementById('allocationResult');
        if (!allocationDiv) return;

        const allocation = result.allocation_result;
        
        let html = `
            <div class="alert alert-info mb-3">
                <div class="d-flex align-items-center">
                    <i class="fas fa-info-circle fa-2x me-3"></i>
                    <div>
                        <h6 class="mb-1">Task ID: ${result.task_id}</h6>
                        <small>Generated at ${new Date(result.timestamp).toLocaleString()}</small>
                    </div>
                </div>
            </div>
        `;
        
        if (allocation.recommended_server) {
            html += `
                <div class="alert alert-success mb-3">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-robot fa-2x me-3"></i>
                        <div class="flex-grow-1">
                            <h5 class="mb-2">🎯 AI Recommendation: ${allocation.recommended_server}</h5>
                            <div class="row">
                                <div class="col-md-6">
                                    <strong>Confidence Score:</strong> 
                                    <span class="badge bg-primary fs-6">${allocation.confidence_score.toFixed(1)}/100</span>
                                </div>
                                <div class="col-md-6">
                                    <strong>Energy Efficiency:</strong> 
                                    <span class="badge bg-success fs-6">${allocation.energy_efficiency_score?.toFixed(1) || 0}/100</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            if (allocation.processing_result) {
                const processing = allocation.processing_result;
                html += `
                    <div class="card mb-3">
                        <div class="card-header">
                            <h6><i class="fas fa-cogs"></i> Processing Results</h6>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <small class="text-muted">Processing Time</small>
                                    <div class="h5 text-primary">${processing.processing_time_sec.toFixed(2)}s</div>
                                </div>
                                <div class="col-md-6">
                                    <small class="text-muted">Energy Consumed</small>
                                    <div class="h5 text-warning">${processing.energy_consumed_wh.toFixed(2)} Wh</div>
                                </div>
                                <div class="col-md-6">
                                    <small class="text-muted">Heat Generated</small>
                                    <div class="h5 text-danger">${processing.heat_generated_j.toFixed(0)} J</div>
                                </div>
                                <div class="col-md-6">
                                    <small class="text-muted">Carbon Footprint</small>
                                    <div class="h5 text-info">${processing.carbon_footprint_g?.toFixed(1) || 0} g CO₂</div>
                                </div>
                            </div>
                            <div class="mt-2">
                                <span class="badge bg-success">
                                    <i class="fas fa-check"></i> ${processing.status}
                                </span>
                                <span class="badge bg-info">
                                    <i class="fas fa-server"></i> ${processing.server_tier} Server
                                </span>
                            </div>
                        </div>
                    </div>
                `;
            }
        } else {
            html += `
                <div class="alert alert-warning">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-exclamation-triangle fa-2x me-3"></i>
                        <div>
                            <h5>⚠️ No Suitable Server Found</h5>
                            <p class="mb-0">All servers are currently overloaded or lack sufficient resources for this task.</p>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Show top server candidates
        const topServers = Object.entries(allocation.all_server_scores || {})
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5);
        
        if (topServers.length > 0) {
            html += `
                <div class="mt-3">
                    <h6><i class="fas fa-chart-bar"></i> Server Allocation Scores</h6>
                    <div class="row">
            `;
            
            topServers.forEach(([serverId, score], index) => {
                const isRecommended = serverId === allocation.recommended_server;
                const cardClass = isRecommended ? 'border-success bg-light' : '';
                const badgeClass = score > 70 ? 'bg-success' : score > 40 ? 'bg-warning' : 'bg-danger';
                const serverStatus = allocation.server_status?.[serverId] || {};
                
                html += `
                    <div class="col-md-6 mb-2">
                        <div class="card ${cardClass}">
                            <div class="card-body p-2">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <h6 class="mb-1">${serverId} ${isRecommended ? '⭐' : ''}</h6>
                                        <small class="text-muted">
                                            Load: ${serverStatus.current_load_pct?.toFixed(1) || 0}% | 
                                            Temp: ${serverStatus.temperature_c || 0}°C
                                        </small>
                                    </div>
                                    <div class="text-end">
                                        <span class="badge ${badgeClass}">${score.toFixed(1)}</span>
                                        <div class="progress mt-1" style="width: 60px; height: 6px;">
                                            <div class="progress-bar ${badgeClass}" 
                                                 style="width: ${score}%"></div>
                                        </div>
                                    </div>
                                </div>
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

    async runOptimizedSimulation() {
        this.showLoadingModal('Running Simulation', 'Executing high-performance AI simulation...');
        
        try {
            const numTasks = Math.min(parseInt(document.getElementById('numTasks')?.value || 30), 50);
            const duration = Math.min(parseInt(document.getElementById('simulationDuration')?.value || 100), 200);
            const simulationType = document.getElementById('simulationType')?.value || 'balanced';

            this.updateLoadingProgress(30);

            const result = await this.makeOptimizedRequest('/api/run-simulation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    num_tasks: numTasks,
                    duration: duration,
                    simulation_type: simulationType
                })
            });

            this.updateLoadingProgress(80);

            if (result && result.success) {
                this.displaySimulationResults(result.results);
                this.showToast('Simulation completed successfully!', 'success');
            } else {
                throw new Error(result?.error || 'Simulation failed');
            }

            this.updateLoadingProgress(100);
        } catch (error) {
            console.error('Error running simulation:', error);
            this.showToast('Simulation failed: ' + error.message, 'error');
        } finally {
            this.hideLoadingModal();
        }
    }

    displaySimulationResults(results) {
        const resultsDiv = document.getElementById('simulationResults');
        if (!resultsDiv) return;
        
        const html = `
            <div class="row">
                <div class="col-md-3">
                    <div class="card bg-success text-white">
                        <div class="card-body text-center p-2">
                            <h6>Tasks Processed</h6>
                            <h4>${results.processed_tasks}</h4>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-warning text-white">
                        <div class="card-body text-center p-2">
                            <h6>Total Energy</h6>
                            <h4>${results.total_energy_consumed.toFixed(2)} Wh</h4>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-info text-white">
                        <div class="card-body text-center p-2">
                            <h6>Total Time</h6>
                            <h4>${results.total_processing_time.toFixed(1)}</h4>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-primary text-white">
                        <div class="card-body text-center p-2">
                            <h6>Avg Energy/Task</h6>
                            <h4>${results.average_energy_per_task.toFixed(2)} Wh</h4>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-md-12">
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle"></i>
                        <strong>Simulation Complete:</strong> 
                        Processed ${results.processed_tasks} tasks with average energy consumption of ${results.average_energy_per_task.toFixed(2)} Wh per task.
                    </div>
                </div>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
    }

    // Firebase Integration Methods
    async saveServersToFirebase() {
        if (!this.firebaseReady) {
            this.showToast('Firebase not available', 'error');
            return;
        }

        this.showLoadingModal('Saving to Firebase', 'Uploading server data to cloud...');
        
        try {
            const data = await this.makeOptimizedRequest('/api/firebase/servers');
            
            if (data && data.success) {
                for (const server of data.servers) {
                    await this.firebase.saveData('neurosim_servers', server.server_id, server);
                }
                this.showToast(`Successfully saved ${data.servers.length} servers to Firebase`, 'success');
            }
        } catch (error) {
            console.error('Error saving servers to Firebase:', error);
            this.showToast('Failed to save servers to Firebase', 'error');
        } finally {
            this.hideLoadingModal();
        }
    }

    async saveTrainingDataToFirebase() {
        if (!this.firebaseReady) {
            this.showToast('Firebase not available', 'error');
            return;
        }

        this.showLoadingModal('Saving Training Data', 'Uploading ML training data...');
        
        try {
            const data = await this.makeOptimizedRequest('/api/firebase/training-data?limit=100');
            
            if (data && data.success) {
                let saved = 0;
                for (const record of data.training_data) {
                    await this.firebase.addData('neurosim_training', record);
                    saved++;
                    this.updateLoadingProgress((saved / data.training_data.length) * 100);
                }
                this.showToast(`Successfully saved ${saved} training records to Firebase`, 'success');
            }
        } catch (error) {
            console.error('Error saving training data to Firebase:', error);
            this.showToast('Failed to save training data to Firebase', 'error');
        } finally {
            this.hideLoadingModal();
        }
    }

    async saveAllDataToFirebase() {
        if (!this.firebaseReady) {
            this.showToast('Firebase not available', 'error');
            return;
        }

        this.showLoadingModal('Saving All Data', 'Performing complete data backup to Firebase...');
        
        try {
            await this.saveServersToFirebase();
            await this.saveTrainingDataToFirebase();
            this.showToast('All data successfully backed up to Firebase', 'success');
        } catch (error) {
            console.error('Error saving all data to Firebase:', error);
            this.showToast('Failed to complete full backup', 'error');
        } finally {
            this.hideLoadingModal();
        }
    }

    async loadServersFromFirebase() {
        if (!this.firebaseReady) {
            this.showToast('Firebase not available', 'error');
            return;
        }

        this.showLoadingModal('Loading from Firebase', 'Downloading server data...');
        
        try {
            const result = await this.firebase.getAllData('neurosim_servers');
            if (result.success) {
                console.log(`Loaded ${result.data.length} servers from Firebase`);
                this.showToast(`Successfully loaded ${result.data.length} servers from Firebase`, 'success');
            }
        } catch (error) {
            console.error('Error loading servers from Firebase:', error);
            this.showToast('Failed to load servers from Firebase', 'error');
        } finally {
            this.hideLoadingModal();
        }
    }

    async loadAllocationsFromFirebase() {
        if (!this.firebaseReady) {
            this.showToast('Firebase not available', 'error');
            return;
        }

        this.showLoadingModal('Loading Allocations', 'Downloading allocation history...');
        
        try {
            const result = await this.firebase.getAllData('neurosim_allocations');
            if (result.success) {
                console.log(`Loaded ${result.data.length} allocations from Firebase`);
                this.showToast(`Successfully loaded ${result.data.length} allocations from Firebase`, 'success');
            }
        } catch (error) {
            console.error('Error loading allocations from Firebase:', error);
            this.showToast('Failed to load allocations from Firebase', 'error');
        } finally {
            this.hideLoadingModal();
        }
    }

    async loadAllFromFirebase() {
        if (!this.firebaseReady) {
            this.showToast('Firebase not available', 'error');
            return;
        }

        this.showLoadingModal('Syncing All Data', 'Downloading all data from Firebase...');
        
        try {
            await this.loadServersFromFirebase();
            await this.loadAllocationsFromFirebase();
            this.showToast('All data synchronized from Firebase', 'success');
        } catch (error) {
            console.error('Error syncing all data from Firebase:', error);
            this.showToast('Failed to sync all data', 'error');
        } finally {
            this.hideLoadingModal();
        }
    }

    updateSystemStatus(message = null) {
        if (message) {
            const statusElement = document.getElementById('systemStatusText');
            if (statusElement) {
                statusElement.textContent = message;
            }
            return;
        }

        fetch('/api/system-status')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const status = data.system_status;
                    const uptime = Math.floor(status.uptime_seconds / 3600);
                    const statusElement = document.getElementById('systemStatusText');
                    if (statusElement) {
                        statusElement.innerHTML = 
                            `System healthy | Uptime: ${uptime}h | Tasks: ${status.total_tasks_processed} | Model: ${status.model_trained ? 'Ready' : 'Training'}`;
                    }
                }
            })
            .catch(error => console.error('Error updating system status:', error));
    }

    setupOptimizedRealTimeUpdates() {
        // Clear any existing intervals
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        // Update dashboard every 20 seconds (reduced frequency)
        this.updateInterval = setInterval(() => {
            if (!this.isDestroyed) {
                this.loadDashboardDataOptimized();
                this.updateSystemStatus();
            }
        }, 20000);

        // Update server comparison every 15 seconds
        setInterval(() => {
            if (!this.isDestroyed) {
                this.loadServerComparisonOptimized();
            }
        }, 15000);
    }

    updateFirebaseStatus(connected) {
        const statusDiv = document.getElementById('firebaseStatus');
        if (!statusDiv) return;

        if (connected) {
            statusDiv.innerHTML = `
                <div class="d-flex align-items-center text-success">
                    <i class="fas fa-check-circle me-2"></i>
                    <span><strong>Connected to Firebase Firestore</strong></span>
                </div>
                <small class="text-muted">Project: cloud-storage-c0971</small>
            `;
        } else {
            statusDiv.innerHTML = `
                <div class="d-flex align-items-center text-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <span><strong>Firebase not available</strong> - Running in local mode</span>
                </div>
            `;
        }
    }

    showErrorState(section) {
        const errorHtml = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>Loading Error:</strong> Unable to load ${section} data. 
                <button class="btn btn-sm btn-outline-warning ms-2" onclick="location.reload()">
                    <i class="fas fa-sync"></i> Retry
                </button>
            </div>
        `;
        
        switch(section) {
            case 'servers':
                const serverDiv = document.getElementById('serverComparisonTable');
                if (serverDiv) serverDiv.innerHTML = errorHtml;
                break;
            case 'model':
                const modelDiv = document.getElementById('modelPerformance');
                if (modelDiv) modelDiv.innerHTML = errorHtml;
                break;
            case 'dashboard':
                console.error('Dashboard data loading failed');
                break;
        }
    }

    // Enhanced Loading and Toast Methods
    showLoadingModal(title = 'Processing', message = 'Please wait...') {
        const titleEl = document.getElementById('loadingTitle');
        const messageEl = document.getElementById('loadingMessage');
        const progressEl = document.getElementById('loadingProgress');
        
        if (titleEl) titleEl.textContent = title;
        if (messageEl) messageEl.textContent = message;
        if (progressEl) progressEl.style.width = '0%';
        
        const modalEl = document.getElementById('loadingModal');
        if (modalEl && typeof bootstrap !== 'undefined') {
            const modal = new bootstrap.Modal(modalEl);
            modal.show();
        }
    }

    updateLoadingProgress(percentage) {
        const progressEl = document.getElementById('loadingProgress');
        if (progressEl) {
            progressEl.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
        }
    }

    hideLoadingModal() {
        const modalEl = document.getElementById('loadingModal');
        if (modalEl && typeof bootstrap !== 'undefined') {
            const modal = bootstrap.Modal.getInstance(modalEl);
            if (modal) {
                modal.hide();
            }
        }
    }

    showToast(message, type = 'success') {
        const toast = document.getElementById('notificationToast');
        const icon = document.getElementById('toastIcon');
        const title = document.getElementById('toastTitle');
        const messageEl = document.getElementById('toastMessage');

        if (!toast || !icon || !title || !messageEl) return;

        if (type === 'success') {
            icon.className = 'fas fa-check-circle text-success me-2';
            title.textContent = 'Success';
        } else {
            icon.className = 'fas fa-exclamation-circle text-danger me-2';
            title.textContent = 'Error';
        }

        messageEl.textContent = message;
        
        if (typeof bootstrap !== 'undefined') {
            const bsToast = new bootstrap.Toast(toast);
            bsToast.show();
        }
    }

    // Cleanup method to prevent memory leaks
    destroy() {
        this.isDestroyed = true;
        
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        if (this.requestController) {
            this.requestController.abort();
        }
        
        if (this.performanceObserver) {
            this.performanceObserver.disconnect();
        }
        
        // Destroy all charts
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        
        this.cache.clear();
        this.loadingStates.clear();
    }
}

// Initialize optimized dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new OptimizedNeuroSimGreenDashboard();
    
    // Handle page unload
    window.addEventListener('beforeunload', () => {
        if (window.dashboard) {
            window.dashboard.destroy();
        }
    });
});

// Export for potential module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OptimizedNeuroSimGreenDashboard;
}
