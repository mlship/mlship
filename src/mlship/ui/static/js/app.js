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

// Prediction Form Component
function PredictionForm({ modelInfo, onPredict, onError }) {
    const [inputs, setInputs] = React.useState([]);
    const [isLoading, setIsLoading] = React.useState(false);
    const [imageFile, setImageFile] = React.useState(null);
    const [imagePreview, setImagePreview] = React.useState(null);

    React.useEffect(() => {
        if (modelInfo) {
            // For numeric or tensor inputs (except image models), create array of empty values matching input shape
            if ((modelInfo.input_type === 'numeric' || modelInfo.input_type === 'tensor' || modelInfo.framework === 'sklearn') 
                && modelInfo.type !== 'PTImageClassifier') {
                const inputSize = modelInfo.features?.length || modelInfo.input_shape?.[1] || 0;
                setInputs(Array(inputSize).fill('').map((_, i) => ({
                    name: modelInfo.features?.[i] || `Input ${i + 1}`,
                    value: ''
                })));
            }
        }
    }, [modelInfo]);

    const isImageModel = modelInfo?.type === 'PTImageClassifier';
    const isTextModel = modelInfo?.input_type === 'text';

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImageFile(file);
            // Create preview
            const reader = new FileReader();
            reader.onloadend = () => {
                setImagePreview(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const preprocessImage = async (file) => {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                // Resize to match model input size (32x32)
                canvas.width = 32;
                canvas.height = 32;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, 32, 32);
                
                // Get pixel data
                const imageData = ctx.getImageData(0, 0, 32, 32);
                const data = imageData.data;
                
                // Convert to RGB channels
                const pixels = [];
                for (let i = 0; i < data.length; i += 4) {
                    pixels.push(data[i] / 255.0);     // R
                    pixels.push(data[i + 1] / 255.0); // G
                    pixels.push(data[i + 2] / 255.0); // B
                }
                
                // Send as a single flattened array - the server will reshape it
                resolve([pixels]);
            };
            img.src = URL.createObjectURL(file);
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        try {
            let inputData;
            
            if (isImageModel) {
                if (!imageFile) {
                    throw new Error('Please select an image');
                }
                inputData = await preprocessImage(imageFile);
            } else if (isTextModel) {
                // For text models, send the text as a single input
                inputData = [[inputs[0].value]];
            } else {
                // Convert numeric inputs to array
                inputData = [inputs.map(i => parseFloat(i.value) || 0)];
            }

            const response = await fetch(`${BASE_URL}/api/model/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ inputs: inputData })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Prediction failed');
            }

            const result = await response.json();
            const timestamp = new Date();
            
            onPredict({
                inputs: isImageModel ? imagePreview : inputs.map(i => i.value),
                prediction: result.prediction,
                timestamp,
                isImage: isImageModel,
                modelPath: modelInfo.model_path
            });

            // Clear inputs after successful prediction
            if (!isImageModel) {
                setInputs(inputs.map(i => ({ ...i, value: '' })));
            }
        } catch (error) {
            onError(error.message);
        } finally {
            setIsLoading(false);
        }
    };

    if (!modelInfo) return <LoadingSpinner />;

    return (
        <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-2xl font-bold mb-4">Make Prediction</h2>
            
            {/* Model Type Info */}
            <div className="mb-4 text-sm text-gray-600">
                <p>Model Type: {modelInfo.type}</p>
                <p>Framework: {modelInfo.framework}</p>
                <p>Input Type: {modelInfo.input_type}</p>
                {modelInfo.input_shape && (
                    <p>Input Shape: {JSON.stringify(modelInfo.input_shape)}</p>
                )}
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
                {isImageModel ? (
                    <div>
                        <label className="block text-sm font-medium text-gray-700">
                            Upload Image (will be resized to 32x32)
                        </label>
                        <input
                            type="file"
                            accept="image/*"
                            onChange={handleImageChange}
                            className="mt-1 block w-full"
                        />
                        {imagePreview && (
                            <div className="mt-2">
                                <img
                                    src={imagePreview}
                                    alt="Preview"
                                    className="max-w-xs rounded border"
                                />
                            </div>
                        )}
                    </div>
                ) : isTextModel ? (
                    <div>
                        <label className="block text-sm font-medium text-gray-700">
                            Enter Text
                        </label>
                        <textarea
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                            rows="4"
                            value={inputs[0]?.value || ''}
                            onChange={(e) => {
                                const newInputs = [...inputs];
                                if (!newInputs[0]) newInputs[0] = { name: 'text', value: '' };
                                newInputs[0].value = e.target.value;
                                setInputs(newInputs);
                            }}
                            placeholder="Enter your text here..."
                            required
                        />
                    </div>
                ) : inputs.length > 0 ? (
                    <div className="space-y-2">
                        {inputs.map((input, index) => (
                            <div key={index}>
                                <label className="block text-sm font-medium text-gray-700">
                                    {input.name}
                                </label>
                                <input
                                    type="number"
                                    step="any"
                                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                                    value={input.value}
                                    onChange={(e) => {
                                        const newInputs = [...inputs];
                                        newInputs[index].value = e.target.value;
                                        setInputs(newInputs);
                                    }}
                                    required
                                />
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="text-red-600">
                        Error: Model does not specify input shape or features
                    </div>
                )}

                <button
                    type="submit"
                    disabled={isLoading || (isImageModel && !imageFile) || (!isImageModel && inputs.length === 0)}
                    className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                >
                    {isLoading ? <LoadingSpinner /> : 'Predict'}
                </button>
            </form>
        </div>
    );
}

// Prediction Results Component
function PredictionResults({ predictions, onClear }) {
    if (!predictions || predictions.length === 0) {
        return (
            <div className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-2xl font-bold mb-4">Predictions</h2>
                <p className="text-gray-500">No predictions yet</p>
            </div>
        );
    }

    return (
        <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold">Predictions</h2>
                <button
                    onClick={onClear}
                    className="px-3 py-1 text-sm text-red-600 hover:text-red-800"
                >
                    Clear History
                </button>
            </div>
            <div className="predictions-container">
                {predictions.map((pred, index) => (
                    <div key={index} className="mb-4 p-4 border rounded relative">
                        {/* Model filename in top right */}
                        <div className="absolute top-2 right-2 text-sm text-gray-500 font-mono">
                            {pred.modelPath ? pred.modelPath.split('/').pop() : 'Unknown Model'}
                        </div>
                        <div className="text-sm text-gray-500 mt-6">
                            {pred.timestamp.toLocaleString()}
                        </div>
                        <div className="mt-2">
                            <strong>Input:</strong>
                            {pred.isImage ? (
                                <div className="mt-1">
                                    <img
                                        src={pred.inputs}
                                        alt="Input"
                                        className="max-w-xs rounded border"
                                    />
                                </div>
                            ) : (
                                <pre className="mt-1 text-sm bg-gray-50 p-2 rounded">
                                    {JSON.stringify(pred.inputs, null, 2)}
                                </pre>
                            )}
                        </div>
                        <div className="mt-2">
                            <strong>Prediction:</strong>
                            <pre className="mt-1 text-sm bg-gray-50 p-2 rounded">
                                {JSON.stringify(pred.prediction, null, 2)}
                            </pre>
                        </div>
                    </div>
                ))}
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
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                    <p><span className="font-semibold">Type:</span> {modelInfo.type}</p>
                    <p><span className="font-semibold">Framework:</span> {modelInfo.framework}</p>
                    <p><span className="font-semibold">Input Type:</span> {modelInfo.input_type}</p>
                    <p><span className="font-semibold">Output Type:</span> {modelInfo.output_type}</p>
                    {modelInfo.input_shape && (
                        <p><span className="font-semibold">Input Shape:</span> {JSON.stringify(modelInfo.input_shape)}</p>
                    )}
                    {modelInfo.output_shape && (
                        <p><span className="font-semibold">Output Shape:</span> {JSON.stringify(modelInfo.output_shape)}</p>
                    )}
                </div>
                <div className="space-y-2">
                    {modelInfo.features && modelInfo.features.length > 0 && (
                        <div>
                            <h3 className="font-semibold">Features:</h3>
                            <ul className="list-disc list-inside">
                                {modelInfo.features.map((feature, i) => (
                                    <li key={i}>{feature}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                    {modelInfo.classes && modelInfo.classes.length > 0 && (
                        <div>
                            <h3 className="font-semibold">Classes:</h3>
                            <ul className="list-disc list-inside">
                                {modelInfo.classes.map((cls, i) => (
                                    <li key={i}>{cls}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                    {modelInfo.preprocessing && (
                        <div>
                            <h3 className="font-semibold">Preprocessing:</h3>
                            <pre className="text-sm bg-gray-50 p-2 rounded">
                                {JSON.stringify(modelInfo.preprocessing, null, 2)}
                            </pre>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

// Metrics Component
function MetricsDisplay({ metrics }) {
    const [metricsHistory, setMetricsHistory] = React.useState([]);
    const [currentUptime, setCurrentUptime] = React.useState(0);
    const [startTime] = React.useState(Date.now() / 1000); // Store initial timestamp in seconds

    // Format uptime into minutes
    const formatUptime = (seconds) => {
        if (seconds === undefined || seconds === null || isNaN(seconds)) return '0 min';
        const minutes = Math.floor(seconds / 60);
        return `${minutes} min`;
    };

    // Update uptime based on start time
    React.useEffect(() => {
        const timer = setInterval(() => {
            const now = Date.now() / 1000;
            const uptime = now - startTime + (metrics?.metrics?.uptime || 0);
            setCurrentUptime(uptime);
        }, 1000);

        return () => clearInterval(timer);
    }, [startTime, metrics?.metrics?.uptime]);

    React.useEffect(() => {
        if (metrics) {
            setMetricsHistory(prev => {
                const newHistory = [...prev];
                // Keep last 100 data points for scrolling
                if (newHistory.length > 100) {
                    newHistory.shift();
                }
                // Add new data point with timestamp
                newHistory.push({
                    time: new Date(),
                    requests: metrics.requests,
                    avg_latency: metrics.avg_latency
                });
                return newHistory;
            });
        }
    }, [metrics]);

    React.useEffect(() => {
        if (metricsHistory.length > 0) {
            const trace = {
                x: metricsHistory.map(m => m.time),
                y: metricsHistory.map(m => m.avg_latency),
                type: 'scatter',
                name: 'Avg Latency (ms)',
                line: { color: '#4F46E5' }
            };

            const layout = {
                title: '',
                height: 250,
                width: Math.max(800, metricsHistory.length * 20), // Make width dynamic based on points
                margin: { t: 10, r: 50, l: 50, b: 30 },
                xaxis: { 
                    title: 'Time',
                    rangeslider: {}, // Add range slider for scrolling
                    type: 'date'
                },
                yaxis: { 
                    title: 'Avg Latency (ms)',
                    titlefont: { color: '#4F46E5' },
                    tickfont: { color: '#4F46E5' }
                },
                showlegend: true,
                legend: {
                    orientation: 'h',
                    y: -0.2
                }
            };

            const config = {
                displayModeBar: false,
                scrollZoom: true, // Enable scroll zoom
                responsive: true
            };

            Plotly.newPlot('metrics-chart', [trace], layout, config);
        }
    }, [metricsHistory]);

    if (!metrics) return null;

    return (
        <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-2">Metrics</h3>
            <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                    <div className="text-sm text-gray-600">Total Requests</div>
                    <div className="text-xl font-bold">{metrics.requests}</div>
                </div>
                <div>
                    <div className="text-sm text-gray-600">Avg Latency</div>
                    <div className="text-xl font-bold">{metrics.avg_latency?.toFixed(2) || '0.00'}ms</div>
                </div>
                <div>
                    <div className="text-sm text-gray-600">Uptime</div>
                    <div className="text-xl font-bold">{formatUptime(currentUptime)}</div>
                </div>
            </div>
            <div className="flex justify-center">
                <div className="overflow-x-auto max-w-full">
                    <div id="metrics-chart" className="w-[800px]"></div>
                </div>
            </div>
        </div>
    );
}

// Main App Component
function App() {
    const [modelInfo, setModelInfo] = React.useState(window.MODEL_INFO || null);
    const [metrics, setMetrics] = React.useState(null);
    const [metricsHistory, setMetricsHistory] = React.useState([]);
    const [predictions, setPredictions] = React.useState(() => {
        try {
            const savedPredictions = localStorage.getItem('mlship_predictions');
            return savedPredictions ? JSON.parse(savedPredictions).map(pred => ({
                ...pred,
                timestamp: new Date(pred.timestamp)
            })) : [];
        } catch (e) {
            console.error('Error loading predictions from localStorage:', e);
            return [];
        }
    });
    const [error, setError] = React.useState(null);
    const [isLoading, setIsLoading] = React.useState(false);
    const [lastUpdateTime, setLastUpdateTime] = React.useState(Date.now());

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
            fetch(`${BASE_URL}/api/model/info`)
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

        const wsMetrics = new WebSocket(`ws://${window.location.host}/ws/metrics`);

        wsMetrics.onmessage = (event) => {
            const now = Date.now();
            if (now - lastUpdateTime >= 2000) {
                const newMetrics = JSON.parse(event.data);
                setMetrics(newMetrics);
                setMetricsHistory(prev => {
                    const newHistory = [...prev, {
                        time: new Date(),
                        ...newMetrics
                    }].slice(-30);
                    return newHistory;
                });
                setLastUpdateTime(now);
            }
        };

        wsMetrics.onerror = (error) => {
            console.error('Metrics WebSocket error:', error);
            setError('Failed to connect to metrics stream');
        };

        return () => {
            wsMetrics.close();
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
                    <MetricsDisplay metrics={metrics} />
                </div>

                {/* Bottom Row: Model Info */}
                <div className="grid grid-cols-1">
                    <ModelInfoCard modelInfo={modelInfo} />
                </div>
            </div>
        </div>
    );
}

// Render the app
ReactDOM.render(<App />, document.getElementById('root')); 