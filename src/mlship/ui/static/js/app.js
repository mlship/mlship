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

// Metrics Component
function MetricsCard({ metrics, metricsHistory }) {
    React.useEffect(() => {
        if (metricsHistory && metricsHistory.length > 0) {
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
                displayModeBar: false,
                responsive: true
            };

            Plotly.newPlot('metrics-chart', [latencyTrace, requestsTrace], layout, config);
        }
    }, [metricsHistory]);

    return (
        <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-2xl font-bold mb-4">Real-time Metrics</h2>
            <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-500">Requests</p>
                    <p className="text-2xl font-semibold">{metrics ? metrics.requests : 0}</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-500">Avg Latency (ms)</p>
                    <p className="text-2xl font-semibold">{metrics ? metrics.avg_latency : 0}</p>
                </div>
            </div>
            <div id="metrics-chart" className="w-full"></div>
        </div>
    );
}

// Input Component based on type
function ModelInput({ type, value, onChange, disabled }) {
    switch (type) {
        case 'numeric':
            return (
                <input
                    type="number"
                    step="any"
                    value={value}
                    onChange={onChange}
                    className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                    required
                    disabled={disabled}
                />
            );
        case 'text':
            return (
                <textarea
                    value={value}
                    onChange={onChange}
                    className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                    rows="3"
                    required
                    disabled={disabled}
                />
            );
        case 'image':
            return (
                <div className="mt-1">
                    <input
                        type="file"
                        accept="image/*"
                        onChange={onChange}
                        className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
                        disabled={disabled}
                    />
                    {value && (
                        <div className="mt-2">
                            <img 
                                src={typeof value === 'string' ? value : URL.createObjectURL(value)} 
                                alt="Preview" 
                                className="max-h-48 rounded-md"
                            />
                        </div>
                    )}
                </div>
            );
        default:
            return null;
    }
}

// Output Component based on type
function PredictionOutput({ type, data }) {
    switch (type) {
        case 'numeric':
            return <div className="text-lg">{Number(data).toFixed(4)}</div>;
        case 'label':
            return (
                <div>
                    <div className="text-lg">{data.label}</div>
                    {data.score !== undefined && (
                        <div className="text-sm text-gray-500">
                            Confidence: {(data.score * 100).toFixed(2)}%
                        </div>
                    )}
                </div>
            );
        case 'text':
            return <div className="text-lg whitespace-pre-wrap">{data.generated_text}</div>;
        case 'segmentation':
            return (
                <div>
                    <div className="text-lg">Segments detected: {data.segments.length}</div>
                    <ul className="mt-2 text-sm">
                        {data.segments.map((segment, i) => (
                            <li key={i} className="text-gray-600">
                                {segment.label}: {(segment.score * 100).toFixed(2)}%
                            </li>
                        ))}
                    </ul>
                </div>
            );
        default:
            return <div className="text-lg">{JSON.stringify(data)}</div>;
    }
}

// Prediction Form Component
function PredictionForm({ modelInfo, onPredict, onError }) {
    if (!modelInfo) return null;

    const [inputs, setInputs] = React.useState(
        modelInfo.input_type === 'numeric' 
            ? Array(modelInfo.n_features || 0).fill('')
            : ['']
    );
    const [isSubmitting, setIsSubmitting] = React.useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);
        try {
            const formData = new FormData();
            
            if (modelInfo.input_type === 'image') {
                formData.append('image', inputs[0]);
                const response = await fetch(`${BASE_URL}/api/predict`, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.error) {
                    onError(data.error);
                } else {
                    onPredict({
                        inputs: [inputs[0].name],
                        prediction: data.predictions[0],
                        timestamp: new Date()
                    });
                    setInputs(['']);
                }
            } else {
                const response = await fetch(`${BASE_URL}/api/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        inputs: modelInfo.input_type === 'numeric' 
                            ? [inputs.map(Number)]
                            : inputs
                    })
                });
                const data = await response.json();
                if (data.error) {
                    onError(data.error);
                } else {
                    onPredict({
                        inputs: modelInfo.input_type === 'numeric' 
                            ? inputs.map(Number)
                            : inputs,
                        prediction: data.predictions[0],
                        timestamp: new Date()
                    });
                    setInputs(modelInfo.input_type === 'numeric' 
                        ? Array(modelInfo.n_features || 0).fill('')
                        : ['']
                    );
                }
            }
        } catch (err) {
            onError(err.message);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleInputChange = (index, value) => {
        if (modelInfo.input_type === 'image') {
            setInputs([value.target.files[0]]);
        } else {
            const newInputs = [...inputs];
            newInputs[index] = modelInfo.input_type === 'text' ? value.target.value : value.target.value;
            setInputs(newInputs);
        }
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-2xl font-bold mb-4">Make Prediction</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
                {inputs.map((value, i) => (
                    <div key={i}>
                        <label className="block text-sm font-medium text-gray-700">
                            {modelInfo.features && modelInfo.features[i] || `Input ${i + 1}`}
                        </label>
                        <ModelInput
                            type={modelInfo.input_type}
                            value={value}
                            onChange={(e) => handleInputChange(i, e)}
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
                    {isSubmitting ? <LoadingSpinner /> : "Predict"}
                </button>
            </form>
        </div>
    );
}

// Prediction Results Component
function PredictionResults({ predictions, onClear, modelInfo }) {
    if (!modelInfo) return null;

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
                                        {modelInfo.input_type === 'numeric' 
                                            ? `[${pred.inputs.join(', ')}]`
                                            : pred.inputs[0]
                                        }
                                    </div>
                                    <div className="font-medium mt-2">Prediction:</div>
                                    <PredictionOutput 
                                        type={modelInfo.output_type} 
                                        data={pred.prediction}
                                    />
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

// Main App Component
function App() {
    const [modelInfo, setModelInfo] = React.useState(window.MODEL_INFO || {});
    const [metrics, setMetrics] = React.useState({
        requests: 0,
        avg_latency: 0,
    });
    const [metricsHistory, setMetricsHistory] = React.useState([]);
    const [predictions, setPredictions] = React.useState(() => {
        try {
            const savedPredictions = localStorage.getItem('mlship_predictions');
            const savedModelType = localStorage.getItem('mlship_model_type');
            
            // Only restore predictions if they're from the same model type
            if (savedPredictions && savedModelType === modelInfo.type) {
                return JSON.parse(savedPredictions).map(pred => ({
                    ...pred,
                    timestamp: new Date(pred.timestamp)
                }));
            }
            return [];
        } catch (e) {
            console.error('Error loading predictions:', e);
            return [];
        }
    });
    const [error, setError] = React.useState(null);
    const [isLoading, setIsLoading] = React.useState(false);
    const [lastUpdateTime, setLastUpdateTime] = React.useState(Date.now());

    // Save predictions and model type to localStorage whenever they change
    React.useEffect(() => {
        try {
            localStorage.setItem('mlship_predictions', JSON.stringify(predictions));
            if (modelInfo && modelInfo.type) {
                localStorage.setItem('mlship_model_type', modelInfo.type);
            }
        } catch (e) {
            console.error('Error saving predictions:', e);
        }
    }, [predictions, modelInfo]);

    React.useEffect(() => {
        // Setup WebSocket for metrics
        const ws = new WebSocket(`ws://${window.location.host}/ws/metrics`);
        ws.onmessage = (event) => {
            const now = Date.now();
            if (now - lastUpdateTime >= 1000) {  // Update every second
                const data = JSON.parse(event.data);
                setMetrics(data);
                setMetricsHistory(prev => {
                    const newHistory = [...prev, {
                        time: new Date(),
                        ...data
                    }].slice(-30);  // Keep last 30 data points
                    return newHistory;
                });
                setLastUpdateTime(now);
            }
        };
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            setError('Failed to connect to metrics stream');
        };
        return () => ws.close();
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

    if (!modelInfo || !modelInfo.type) {
        return (
            <div className="container mx-auto px-4 py-8">
                <header className="mb-8">
                    <h1 className="text-4xl font-bold text-gray-800">MLship Dashboard</h1>
                </header>
                <div className="bg-yellow-100 text-yellow-700 p-4 rounded">
                    No model loaded. Please deploy a model first.
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
                        modelInfo={modelInfo}
                    />
                </div>

                {/* Middle Row: Real-time Metrics */}
                <div className="grid grid-cols-1 gap-6">
                    <MetricsCard metrics={metrics} metricsHistory={metricsHistory} />
                </div>

                {/* Bottom Row: Model Info */}
                <div className="grid grid-cols-1">
                    <div className="bg-white p-6 rounded-lg shadow">
                        <h2 className="text-2xl font-bold mb-4">Model Information</h2>
                        <div className="space-y-2">
                            <p><span className="font-semibold">Type:</span> {modelInfo.type}</p>
                            <p><span className="font-semibold">Input Type:</span> {modelInfo.input_type}</p>
                            <p><span className="font-semibold">Output Type:</span> {modelInfo.output_type}</p>
                            {modelInfo.features && modelInfo.features.length > 0 && (
                                <div>
                                    <p className="font-semibold">Features:</p>
                                    <ul className="list-disc list-inside">
                                        {modelInfo.features.map((feature, i) => (
                                            <li key={i}>{feature}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// Render the App
ReactDOM.render(<App />, document.getElementById('root')); 