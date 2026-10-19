/*
 * TrafficPulse — Frontend Logic
 * Smart Traffic Congestion Monitoring & Prediction System
 * B.Tech 2nd-Year CSE Mini-Project
 *
 * Responsibilities:
 * 1. Form input validation and asynchronous JSON submission to Flask.
 * 2. Visual rendering of Scikit-learn model prediction & confidence bars.
 * 3. Interactive Scenario & Corridor Presets loading.
 * 4. Chart.js metric toggle (Average Vehicle Volume vs. Average Travel Speed).
 * 5. Smooth navigation scrolling and active link tracking across all sections.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. DOM Element References ---
    const form = document.getElementById('predictionForm');
    const roadSelect = document.getElementById('road_name');
    const daySelect = document.getElementById('day_of_week');
    const hourInput = document.getElementById('hour_of_day');
    const vehicleInput = document.getElementById('vehicle_count');
    const speedInput = document.getElementById('avg_speed_kmh');
    const predictBtn = document.getElementById('predictBtn');
    const resetBtn = document.getElementById('resetBtn');
    const errorBox = document.getElementById('errorMessage');

    const placeholderResult = document.getElementById('placeholderResult');
    const predictionResult = document.getElementById('predictionResult');
    const resultBadge = document.getElementById('resultBadge');
    const resultHeading = document.getElementById('resultHeading');
    const resultAdvisory = document.getElementById('resultAdvisory');
    
    const probLow = document.getElementById('probLow');
    const probModerate = document.getElementById('probModerate');
    const probHigh = document.getElementById('probHigh');
    const probBarLow = document.getElementById('probBarLow');
    const probBarModerate = document.getElementById('probBarModerate');
    const probBarHigh = document.getElementById('probBarHigh');

    const navLinks = document.querySelectorAll('.nav-link');
    const presetButtons = document.querySelectorAll('.btn-preset, .chip-btn');

    // Chart toggle elements
    const btnToggleVehicles = document.getElementById('btnToggleVehicles');
    const btnToggleSpeed = document.getElementById('btnToggleSpeed');
    const chartTitle = document.getElementById('chartTitle');
    let trafficChartInstance = null;
    let cachedChartData = null;

    // --- 2. Helper Functions: Error Management ---
    function showError(message) {
        errorBox.textContent = message;
        errorBox.style.display = 'block';
    }

    function clearError() {
        errorBox.textContent = '';
        errorBox.style.display = 'none';
    }

    // --- 3. Scenario & Corridor Presets Handler ---
    function loadPresetValues(road, day, hour, vehicles, speed) {
        clearError();
        if (road) roadSelect.value = road;
        if (day) daySelect.value = day;
        if (hour !== undefined) hourInput.value = hour;
        if (vehicles !== undefined) vehicleInput.value = vehicles;
        if (speed !== undefined) speedInput.value = speed;

        // Smoothly scroll to the prediction section
        const predictSection = document.getElementById('predict');
        if (predictSection) {
            predictSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        // Add a brief subtle pulse to inputs to highlight loaded data
        [roadSelect, daySelect, hourInput, vehicleInput, speedInput].forEach(el => {
            el.style.transition = 'box-shadow 0.2s ease';
            el.style.boxShadow = '0 0 0 3px rgba(132, 204, 22, 0.4)';
            setTimeout(() => {
                el.style.boxShadow = '';
            }, 600);
        });
    }

    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const road = btn.dataset.road;
            const day = btn.dataset.day;
            const hour = btn.dataset.hour;
            const vehicles = btn.dataset.vehicles;
            const speed = btn.dataset.speed;
            loadPresetValues(road, day, hour, vehicles, speed);
        });
    });

    // --- 4. Navigation Smooth Scrolling & Active State ---
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const targetId = link.getAttribute('href');
            if (targetId && targetId.startsWith('#')) {
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    e.preventDefault();
                    navLinks.forEach(l => l.classList.remove('active'));
                    link.classList.add('active');
                    targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        });
    });

    // Update active nav link on scroll
    window.addEventListener('scroll', () => {
        const sections = document.querySelectorAll('section[id]');
        const scrollPosition = window.scrollY + 120;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');

            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    });

    // --- 5. Form Submission & ML Prediction Request ---
    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent full page reload
        clearError();

        // Extract input values
        const road = roadSelect.value.trim();
        const day = daySelect.value.trim();
        const hour = hourInput.value.trim();
        const vehicles = vehicleInput.value.trim();
        const speed = speedInput.value.trim();

        // --- Client-side validation ---
        if (!road) {
            showError('Please select a road corridor in Mohali.');
            roadSelect.focus();
            return;
        }

        if (!day) {
            showError('Please select a day of the week.');
            daySelect.focus();
            return;
        }

        if (hour === '' || isNaN(hour)) {
            showError('Please enter a valid hour of the day (0 to 23).');
            hourInput.focus();
            return;
        }

        const parsedHour = parseInt(hour, 10);
        if (parsedHour < 0 || parsedHour > 23) {
            showError('Hour must be between 0 (midnight) and 23 (11 PM).');
            hourInput.focus();
            return;
        }

        if (vehicles === '' || isNaN(vehicles)) {
            showError('Please enter the number of observed vehicles.');
            vehicleInput.focus();
            return;
        }

        const parsedVehicles = parseInt(vehicles, 10);
        if (parsedVehicles <= 0 || parsedVehicles > 4000) {
            showError('Vehicle count must be a realistic number between 1 and 4000.');
            vehicleInput.focus();
            return;
        }

        if (speed === '' || isNaN(speed)) {
            showError('Please enter an average vehicle speed in km/h.');
            speedInput.focus();
            return;
        }

        const parsedSpeed = parseFloat(speed);
        if (parsedSpeed <= 0 || parsedSpeed > 150) {
            showError('Average speed must be between 1 and 150 km/h.');
            speedInput.focus();
            return;
        }

        // Prepare JSON payload
        const payload = {
            road_name: road,
            day_of_week: day,
            hour_of_day: parsedHour,
            vehicle_count: parsedVehicles,
            avg_speed_kmh: parsedSpeed
        };

        // UI Loading State
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="btn-text">Running Model...</span>';

        try {
            // POST request to Flask endpoint
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                showError(data.error || 'Prediction request failed. Please check your inputs.');
                return;
            }

            // Display the predicted result
            renderPredictionResult(data);

        } catch (err) {
            console.error('Fetch error:', err);
            showError('Unable to connect to Flask server. Please ensure python app.py is running.');
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = '<span class="btn-text">Predict Congestion</span>';
        }
    });

    // --- 6. Render Prediction Result in UI ---
    function renderPredictionResult(data) {
        // Reset previous state classes
        predictionResult.classList.remove('result-low', 'result-moderate', 'result-high');

        // Apply semantic class from backend (result-low, result-moderate, result-high)
        predictionResult.classList.add(data.badge_class);

        // Populate texts
        resultBadge.textContent = data.congestion_label.toUpperCase();
        resultHeading.textContent = `Predicted Status: ${data.congestion_label}`;
        resultAdvisory.textContent = data.advisory;

        // Populate model probabilities and fill bars
        if (data.confidence) {
            const lowPct = data.confidence.low;
            const modPct = data.confidence.moderate;
            const highPct = data.confidence.high;

            probLow.textContent = `${lowPct}%`;
            probModerate.textContent = `${modPct}%`;
            probHigh.textContent = `${highPct}%`;

            if (probBarLow) probBarLow.style.width = `${lowPct}%`;
            if (probBarModerate) probBarModerate.style.width = `${modPct}%`;
            if (probBarHigh) probBarHigh.style.width = `${highPct}%`;
        }

        // Switch visible panels
        placeholderResult.style.display = 'none';
        predictionResult.style.display = 'block';

        // Gently scroll to result on mobile if needed
        if (window.innerWidth < 860) {
            predictionResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    // --- 7. Form Reset Handler ---
    resetBtn.addEventListener('click', () => {
        clearError();
        placeholderResult.style.display = 'block';
        predictionResult.style.display = 'none';
        if (probBarLow) probBarLow.style.width = '0%';
        if (probBarModerate) probBarModerate.style.width = '0%';
        if (probBarHigh) probBarHigh.style.width = '0%';
    });

    // --- 8. Chart.js Bar Chart with Vehicles / Speed Toggle ---
    async function initTrafficChart() {
        const canvas = document.getElementById('trafficBarChart');
        if (!canvas) return;

        try {
            const response = await fetch('/chart-data');
            const result = await response.json();

            if (!result.success) {
                console.warn('Chart data unavailable:', result.error);
                return;
            }

            cachedChartData = result;
            const ctx = canvas.getContext('2d');

            trafficChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: result.labels,
                    datasets: [{
                        label: 'Average Vehicle Volume (veh/hr)',
                        data: result.avg_vehicles || result.data,
                        backgroundColor: [
                            'rgba(34, 197, 94, 0.85)',   // Green for Low
                            'rgba(245, 158, 11, 0.85)',  // Amber for Moderate
                            'rgba(239, 68, 68, 0.85)'    // Red for High
                        ],
                        borderColor: [
                            '#16a34a',
                            '#d97706',
                            '#dc2626'
                        ],
                        borderWidth: 1.5,
                        borderRadius: 4,
                        barPercentage: 0.60
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#0f1d33',
                            titleColor: '#ffffff',
                            bodyColor: '#e2e8f0',
                            padding: 10,
                            callbacks: {
                                label: function(context) {
                                    const isSpeed = btnToggleSpeed.classList.contains('active');
                                    const unit = isSpeed ? ' km/h' : ' veh/hr';
                                    return ` Average: ${context.parsed.y}${unit}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: '#e2e8f0' },
                            ticks: {
                                color: '#475569',
                                font: {
                                    family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                                    size: 11
                                }
                            },
                            title: {
                                display: true,
                                text: 'Average Vehicles Recorded (per hour)',
                                color: '#0f1d33',
                                font: { size: 12, weight: '600' }
                            }
                        },
                        x: {
                            grid: { display: false },
                            ticks: {
                                color: '#0f1d33',
                                font: {
                                    family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                                    size: 12,
                                    weight: '600'
                                }
                            }
                        }
                    }
                }
            });

            // Set up Toggle Listeners
            if (btnToggleVehicles && btnToggleSpeed) {
                btnToggleVehicles.addEventListener('click', () => {
                    if (!cachedChartData || !trafficChartInstance) return;
                    btnToggleVehicles.classList.add('active');
                    btnToggleSpeed.classList.remove('active');
                    chartTitle.textContent = 'Average Vehicle Volume by Congestion Category';

                    trafficChartInstance.data.datasets[0].label = 'Average Vehicle Volume (veh/hr)';
                    trafficChartInstance.data.datasets[0].data = cachedChartData.avg_vehicles;
                    trafficChartInstance.options.scales.y.title.text = 'Average Vehicles Recorded (per hour)';
                    trafficChartInstance.update();
                });

                btnToggleSpeed.addEventListener('click', () => {
                    if (!cachedChartData || !trafficChartInstance) return;
                    btnToggleSpeed.classList.add('active');
                    btnToggleVehicles.classList.remove('active');
                    chartTitle.textContent = 'Average Travel Speed by Congestion Category';

                    trafficChartInstance.data.datasets[0].label = 'Average Travel Speed (km/h)';
                    trafficChartInstance.data.datasets[0].data = cachedChartData.avg_speeds;
                    trafficChartInstance.options.scales.y.title.text = 'Average Speed (km/h)';
                    trafficChartInstance.update();
                });
            }

        } catch (error) {
            console.error('Failed to render chart:', error);
        }
    }

    // Load chart on DOM ready
    initTrafficChart();
});
