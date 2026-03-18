document.addEventListener('DOMContentLoaded', () => {
    fetchDashboardData();
    setInterval(fetchDashboardData, 300000); // Update every 5 mins
});

let forecastChart = null;

async function fetchDashboardData() {
    try {
        const response = await fetch('/api/dashboard_data');
        const data = await response.json();
        
        if (data.error) {
            console.error(data.error);
            return;
        }
        
        updateCurrentStatus(data.current);
        renderChart(data.historical, data.forecast);
        generateInsights(data.current, data.forecast);
        
    } catch (error) {
        console.error('Error fetching dashboard data:', error);
    }
}

function updateCurrentStatus(current) {
    document.getElementById('current-time').innerText = `Last Updated: ${current.timestamp}`;
    document.getElementById('current-aqi').innerText = current.aqi;
    document.getElementById('metric-temp').innerText = current.temperature;
    document.getElementById('metric-humidity').innerText = current.humidity;
    
    // Update AQI style
    const ring = document.querySelector('.aqi-ring');
    const statusText = document.getElementById('aqi-status');
    
    let color = '#2ea44f'; // Good
    let status = 'Good';
    
    if (current.aqi > 100) {
        color = '#cf222e'; // Unhealthy
        status = 'Unhealthy';
    } else if (current.aqi > 50) {
        color = '#d29922'; // Moderate
        status = 'Moderate';
    }
    
    ring.style.borderColor = color;
    ring.style.boxShadow = `0 0 30px ${color}33`; // Add transparency 33 = 20%
    statusText.innerText = status;
    statusText.style.color = color;
}

function renderChart(historical, forecast) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    // Combine timestamps and data
    const labels = [...historical.timestamps, ...forecast.timestamps];
    
    // Data structures for Chart.js
    const historicalSeries = historical.aqi.map((val, index) => ({
        x: historical.timestamps[index],
        y: val
    }));
    
    const forecastSeries = forecast.aqi.map((val, index) => ({
        x: forecast.timestamps[index],
        y: val
    }));
    
    // Create datasets
    const datasets = [
        {
            label: 'Historical AQI',
            data: historicalSeries,
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88, 166, 255, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            fill: true
        },
        {
            label: 'Predicted AQI (Forecast)',
            data: forecastSeries,
            borderColor: '#2ea44f',
            borderDash: [5, 5], // Dashed line for forecast
            backgroundColor: 'rgba(46, 164, 79, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            fill: true
        }
    ];

    if (forecastChart) {
        forecastChart.data.labels = labels;
        forecastChart.data.datasets = datasets;
        forecastChart.update();
    } else {
        forecastChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'hour',
                            displayFormats: {
                                hour: 'MMM d, HH:mm'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time',
                            color: '#8b949e'
                        },
                        grid: {
                            color: 'rgba(255,255,255,0.03)'
                        },
                        ticks: {
                            color: '#8b949e'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'AQI Value',
                            color: '#8b949e'
                        },
                        grid: {
                            color: 'rgba(255,255,255,0.03)'
                        },
                        ticks: {
                            color: '#8b949e'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#f0f6fc'
                        }
                    },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(22, 27, 34, 0.9)',
                        titleColor: '#f0f6fc',
                        bodyColor: '#e1e7ed',
                        borderColor: '#30363d',
                        borderWidth: 1
                    }
                }
            }
        });
    }
}

function generateInsights(current, forecast) {
    const predictionText = document.getElementById('insight-prediction');
    const recommendationText = document.getElementById('insight-recommendation');
    
    // Quick Trend Analyzer
    const avgForecast = forecast.aqi.reduce((a, b) => a + b, 0) / forecast.aqi.length;
    const currentAqi = current.aqi;
    
    if (avgForecast > currentAqi + 10) {
        predictionText.innerText = "The AI model predicts a rising trend in AQI over the next 24 hours. Pollution levels are expecting to increase.";
        recommendationText.innerText = "Consider keeping windows closed tomorrow morning and monitoring local air channels.";
    } else if (avgForecast < currentAqi - 10) {
        predictionText.innerText = "Air quality is projected to improve significantly over the next day.";
        recommendationText.innerText = "Good time for outdoor activities tomorrow! The air is clearing up.";
    } else {
        predictionText.innerText = "The model expects air quality to remain stable at current levels for the upcoming day.";
        recommendationText.innerText = "Standard precautions apply. No unusual spikes forecast.";
    }
}
