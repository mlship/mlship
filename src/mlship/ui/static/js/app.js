// Get base URL from template
const BASE_URL = window.BASE_URL || '';

// Loading Component
function LoadingSpinner() {
    return (
        <div className="flex items-center justify-center p-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        </div>
    );
}

// Main App Component
function App() {
    const [modelInfo, setModelInfo] = React.useState(window.MODEL_INFO || null);
    const [metrics, setMetrics] = React.useState(null);
    const [metricsHistory, setMetricsHistory] = React.useState([]);
    const [predictions, setPredictions] = React.useState(() => {
        // Initialize predictions from localStorage
        try {
            const savedPredictions = localStorage.getItem('mlship_predictions');
            return savedPredictions ? JSON.parse(savedPredictions).map(pred => ({
                ...pred,
                timestamp: new Date(pred.timestamp) // Convert timestamp back to Date object
            })) : [];
        } catch (e) {
            console.error('Error loading predictions from localStorage:', e);
            return [];
        }
    });
    const [error, setError] = React.useState(null);
    const [isLoading, setIsLoading] = React.useState(false);
    const [lastUpdateTime, setLastUpdateTime] = React.useState(Date.now());

    // Save predictions to localStorage whenever they change
    React.useEffect(() => {
        try {
            localStorage.setItem('mlship_predictions', JSON.stringify(predictions));
        } catch (e) {
            console.error('Error saving predictions to localStorage:', e);
        }
    }, [predictions]);

    React.useEffect(() => {
        if (!modelInfo) {
            setIsLoading(true);
            // Fetch model info if not provided in template
            fetch(`${BASE_URL}/api/model-info`)
                .then(res => res.json())
                .then(data => {
                    setModelInfo(data);
                    setIsLoading(false);
                })
                .catch(err => {
                    setError(err.message);
                    setIsLoading(false);
                });
        }

        // Setup WebSocket for metrics and predictions
        const wsMetrics = new WebSocket(`ws://${window.location.host}${BASE_URL}/ws/metrics`);
        const wsPredictions = new WebSocket(`ws://${window.location.host}${BASE_URL}/ws/predictions`);

        wsMetrics.onmessage = (event) => {
            const now = Date.now();
            // Only update every 2 seconds
            if (now - lastUpdateTime >= 2000) {
                const newMetrics = JSON.parse(event.data);
                setMetrics(newMetrics);
                setMetricsHistory(prev => {
                    const newHistory = [...prev, {
                        time: new Date(),
                        ...newMetrics
                    }].slice(-30); // Keep last 30 data points
                    return newHistory;
                });
                setLastUpdateTime(now);
            }
        };

        wsPredictions.onmessage = (event) => {
            const prediction = JSON.parse(event.data);
            setPredictions(prev => [{
                inputs: prediction.inputs[0],
                prediction: prediction.predictions[0],
                timestamp: new Date()
            }, ...prev]);
        };

        wsMetrics.onerror = (error) => {
            console.error('Metrics WebSocket error:', error);
            setError('Failed to connect to metrics stream');
        };

        wsPredictions.onerror = (error) => {
            console.error('Predictions WebSocket error:', error);
        };

        return () => {
            wsMetrics.close();
            wsPredictions.close();
        };
    }, [lastUpdateTime]);

    if (isLoading) {
        return (
            <div className="container mx-auto px-4 py-8">
                <header className="mb-8">
                    <h1 className="text-4xl font-bold text-gray-800">MLship Dashboard</h1>
                </header>
                <LoadingSpinner />
            </div>
        );
    }

    if (error) {
        return (
            <div className="container mx-auto px-4 py-8">
                <header className="mb-8">
                    <h1 className="text-4xl font-bold text-gray-800">MLship Dashboard</h1>
                </header>
                <div className="bg-red-100 text-red-700 p-4 rounded">
                    Error: {error}
                </div>
            </div>
        );
    }

    return (
        <div className="container mx-auto px-4 py-8">
            <header className="mb-8">
                <h1 className="text-4xl font-bold text-gray-800">MLship Dashboard</h1>
            </header>

            <div className="grid grid-cols-1 gap-6">
                {/* Top Row: Prediction Interface */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <PredictionForm 
                        modelInfo={modelInfo} 
                        onPredict={(newPrediction) => {
                            setPredictions(prev => [newPrediction, ...prev]);
                        }}
                        onError={setError}
                    />
                    <PredictionResults 
                        predictions={predictions}
                        onClear={() => {
                            setPredictions([]);
                            localStorage.removeItem('mlship_predictions');
                        }}
                    />
                </div>

                {/* Middle Row: Real-time Metrics */}
                <div className="grid grid-cols-1 gap-6">
                    <MetricsCard metrics={metrics} metricsHistory={metricsHistory} />
                </div>

                {/* Bottom Row: Model Info */}
                <div className="grid grid-cols-1">
                    <ModelInfoCard modelInfo={modelInfo} />
                </div>
            </div>
        </div>
    );
}

// Model Info Component
function ModelInfoCard({ modelInfo }) {
    if (!modelInfo) return <LoadingSpinner />;

    return (
        <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-2xl font-bold mb-4">Model Information</h2>
            <div className="space-y-2">
                <p><span className="font-semibold">Type:</span> {modelInfo.type}</p>
                <div>
                    <h3 className="font-semibold">Parameters:</h3>
                    <pre className="bg-gray-50 p-2 rounded text-sm overflow-auto">
                        {JSON.stringify(modelInfo.params, null, 2)}
                    </pre>
                </div>
                {modelInfo.features.length > 0 && (
                    <div>
                        <h3 className="font-semibold">Features:</h3>
                        <ul className="list-disc list-inside">
                            {modelInfo.features.map((feature, i) => (
                                <li key={i}>{feature}</li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </div>
    );
}

// Metrics Component
function MetricsCard({ metrics, metricsHistory }) {
    if (!metrics) return <LoadingSpinner />;

    const uptime = Math.floor((Date.now() / 1000 - metrics.start_time) / 60);

    React.useEffect(() => {
        if (metricsHistory.length > 0) {
            const latencyTrace = {
                x: metricsHistory.map(m => m.time),
                y: metricsHistory.map(m => m.avg_latency),
                type: 'scatter',
                name: 'Avg Latency (ms)',
                line: { color: '#4F46E5' }
            };

            const requestsTrace = {
                x: metricsHistory.map(m => m.time),
                y: metricsHistory.map(m => m.requests),
                type: 'scatter',
                name: 'Total Requests',
                yaxis: 'y2',
                line: { color: '#059669' }
            };

            const layout = {
                title: '',
                height: 250,
                margin: { t: 10, r: 50, l: 50, b: 30 },
                xaxis: { title: 'Time' },
                yaxis: { 
                    title: 'Avg Latency (ms)',
                    titlefont: { color: '#4F46E5' },
                    tickfont: { color: '#4F46E5' }
                },
                yaxis2: {
                    title: 'Total Requests',
                    titlefont: { color: '#059669' },
                    tickfont: { color: '#059669' },
                    overlaying: 'y',
                    side: 'right'
                },
                showlegend: true,
                legend: {
                    orientation: 'h',
                    y: -0.2
                }
            };

            const config = {
                displayModeBar: false
            };

            Plotly.newPlot('metrics-chart', [latencyTrace, requestsTrace], layout, config);
        }
    }, [metricsHistory]);

    return (
        <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-2xl font-bold mb-4">Real-time Metrics</h2>
            
            {/* Current Values */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="text-center">
                    <p className="text-gray-600">Uptime</p>
                    <p className="text-2xl font-bold">{uptime}m</p>
                </div>
                <div className="text-center">
                    <p className="text-gray-600">Requests</p>
                    <p className="text-2xl font-bold">{metrics.requests}</p>
                </div>
                <div className="text-center">
                    <p className="text-gray-600">Avg Latency</p>
                    <p className="text-2xl font-bold">{metrics.avg_latency}ms</p>
                </div>
            </div>

            {/* Metrics Chart */}
            <div id="metrics-chart" className="w-full"></div>
        </div>
    );
}

// Prediction Form Component
function PredictionForm({ modelInfo, onPredict, onError }) {
    if (!modelInfo) return null;

    const [inputs, setInputs] = React.useState(
        Array(modelInfo.n_features || 0).fill('')
    );
    const [isSubmitting, setIsSubmitting] = React.useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);
        try {
            const response = await fetch(`${BASE_URL}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    inputs: [inputs.map(Number)]
                })
            });
            const data = await response.json();
            if (data.error) {
                onError(data.error);
            } else {
                onPredict({
                    inputs: inputs.map(Number),
                    prediction: data.predictions[0],
                    timestamp: new Date()
                });
                // Clear inputs after successful prediction
                setInputs(Array(modelInfo.n_features || 0).fill(''));
            }
        } catch (err) {
            onError(err.message);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-2xl font-bold mb-4">Make Prediction</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
                {inputs.map((value, i) => (
                    <div key={i}>
                        <label className="block text-sm font-medium text-gray-700">
                            {modelInfo.features[i] || `Feature ${i + 1}`}
                        </label>
                        <input
                            type="number"
                            step="any"
                            value={value}
                            onChange={e => {
                                const newInputs = [...inputs];
                                newInputs[i] = e.target.value;
                                setInputs(newInputs);
                            }}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                            required
                            disabled={isSubmitting}
                        />
                    </div>
                ))}
                <button
                    type="submit"
                    className={`w-full py-2 px-4 rounded ${
                        isSubmitting
                            ? 'bg-indigo-400 cursor-not-allowed'
                            : 'bg-indigo-600 hover:bg-indigo-700'
                    } text-white`}
                    disabled={isSubmitting}
                >
                    {isSubmitting ? 'Predicting...' : 'Predict'}
                </button>
            </form>
        </div>
    );
}

// Prediction Results Component
function PredictionResults({ predictions, onClear }) {
    return (
        <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold">Prediction History</h2>
                {predictions.length > 0 && (
                    <button
                        onClick={onClear}
                        className="text-sm text-red-600 hover:text-red-800"
                    >
                        Clear History
                    </button>
                )}
            </div>
            {predictions.length === 0 ? (
                <p className="text-gray-500">Make a prediction to see results</p>
            ) : (
                <div className="predictions-container">
                    {predictions.map((pred, i) => (
                        <div key={i} className="p-4 bg-gray-50 rounded border border-gray-200 mb-2">
                            <div className="flex justify-between items-start">
                                <div>
                                    <div className="font-medium">Input:</div>
                                    <div className="text-sm text-gray-600">
                                        [{pred.inputs.join(', ')}]
                                    </div>
                                    <div className="font-medium mt-2">Prediction:</div>
                                    <div className="text-lg">{pred.prediction}</div>
                                </div>
                                <div className="text-xs text-gray-500">
                                    {new Date(pred.timestamp).toLocaleTimeString()}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

// Render the App
ReactDOM.render(<App />, document.getElementById('root')); 