function ModelInfo({ modelInfo }) {
  if (!modelInfo || modelInfo.status !== 'loaded') {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative">
        {modelInfo?.error || 'No model loaded. Please deploy a model first.'}
      </div>
    );
  }

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Model Information</h2>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p><span className="font-medium">Type:</span> {modelInfo.type}</p>
          <p><span className="font-medium">Input Type:</span> {modelInfo.input_type}</p>
          <p><span className="font-medium">Output Type:</span> {modelInfo.output_type}</p>
          <p><span className="font-medium">Features:</span> {modelInfo.features.join(', ')}</p>
        </div>
        <div>
          <p><span className="font-medium">Status:</span> {modelInfo.status}</p>
          <p><span className="font-medium">Request Count:</span> {modelInfo.request_count}</p>
          <p><span className="font-medium">Average Latency:</span> {modelInfo.average_latency.toFixed(2)}ms</p>
        </div>
      </div>
    </div>
  );
}

function PredictionForm({ modelInfo, onPredict }) {
  const [inputs, setInputs] = React.useState({});
  const [error, setError] = React.useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    try {
      const inputArray = modelInfo.features.map(feature => parseFloat(inputs[feature] || 0));
      await onPredict(inputArray);
      // Clear form after successful prediction
      setInputs({});
    } catch (err) {
      setError(err.message);
    }
  };

  if (!modelInfo || !modelInfo.features) return null;

  return (
    <div className="bg-white shadow rounded-lg p-6 mt-6">
      <h2 className="text-xl font-semibold mb-4">Make Prediction</h2>
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4">
          {error}
        </div>
      )}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          {modelInfo.features.map(feature => (
            <div key={feature}>
              <label className="block text-sm font-medium text-gray-700">{feature}</label>
              <input
                type="number"
                step="any"
                value={inputs[feature] || ''}
                onChange={e => setInputs(prev => ({ ...prev, [feature]: e.target.value }))}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                required
              />
            </div>
          ))}
        </div>
        <button
          type="submit"
          className="w-full bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
        >
          Predict
        </button>
      </form>
    </div>
  );
}

function PredictionsList({ predictions }) {
  return (
    <div className="bg-white shadow rounded-lg p-6 mt-6">
      <h2 className="text-xl font-semibold mb-4">Recent Predictions</h2>
      <div className="predictions-container">
        {predictions.map((pred, idx) => (
          <div key={idx} className="border-b border-gray-200 py-4 last:border-b-0">
            <div className="flex justify-between items-start">
              <div>
                <strong className="text-gray-700">Input:</strong>{' '}
                {pred.inputs.map((val, i) => (
                  <span key={i} className="mr-2">
                    {window.MODEL_INFO.features[i]}: {val}
                  </span>
                ))}
              </div>
              <div>
                <strong className="text-gray-700">Prediction:</strong>{' '}
                <span className="text-indigo-600 font-medium">{pred.prediction}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MetricsGraph({ metrics }) {
  const graphRef = React.useRef(null);

  React.useEffect(() => {
    if (!metrics || !metrics.time || metrics.time.length === 0) return;

    const traces = [
      {
        name: 'Requests',
        x: metrics.time,
        y: metrics.requests,
        type: 'scatter',
        yaxis: 'y1'
      },
      {
        name: 'Avg Latency (ms)',
        x: metrics.time,
        y: metrics.latency,
        type: 'scatter',
        yaxis: 'y2'
      }
    ];

    const layout = {
      title: 'Real-time Metrics',
      xaxis: { title: 'Time' },
      yaxis: { 
        title: 'Total Requests',
        side: 'left'
      },
      yaxis2: {
        title: 'Average Latency (ms)',
        overlaying: 'y',
        side: 'right'
      },
      showlegend: true,
      legend: {
        x: 0,
        y: 1
      }
    };

    Plotly.newPlot(graphRef.current, traces, layout);
  }, [metrics]);

  return (
    <div className="bg-white shadow rounded-lg p-6 mt-6">
      <h2 className="text-xl font-semibold mb-4">Metrics</h2>
      <div ref={graphRef} style={{ width: '100%', height: '300px' }} />
    </div>
  );
}

function App() {
  const [modelInfo, setModelInfo] = React.useState(window.MODEL_INFO);
  const [predictions, setPredictions] = React.useState([]);
  const [metrics, setMetrics] = React.useState({
    time: [],
    requests: [],
    latency: []
  });

  React.useEffect(() => {
    // Set up WebSocket connection for metrics
    const ws = new WebSocket(`ws://${window.location.host}/ws/metrics`);
    
    ws.onmessage = (event) => {
      const metricsData = JSON.parse(event.data);
      const now = new Date();
      
      setMetrics(prev => {
        const newMetrics = {
          time: [...prev.time, now],
          requests: [...prev.requests, metricsData.requests],
          latency: [...prev.latency, metricsData.avg_latency]
        };

        // Keep only last 50 points
        if (newMetrics.time.length > 50) {
          newMetrics.time = newMetrics.time.slice(-50);
          newMetrics.requests = newMetrics.requests.slice(-50);
          newMetrics.latency = newMetrics.latency.slice(-50);
        }

        return newMetrics;
      });
    };

    ws.onclose = () => {
      setTimeout(() => {
        window.location.reload();
      }, 1000);
    };

    return () => ws.close();
  }, []);

  const handlePredict = async (inputs) => {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ inputs: [inputs] }),
    });

    const data = await response.json();
    if (data.error) throw new Error(data.error);

    setPredictions(prev => [{
      inputs,
      prediction: data.predictions[0]
    }, ...prev].slice(0, 100));  // Keep last 100 predictions

    // Update model info after prediction
    const infoResponse = await fetch('/api/model-info');
    const newModelInfo = await infoResponse.json();
    setModelInfo(newModelInfo);
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">MLship Dashboard</h1>
      <ModelInfo modelInfo={modelInfo} />
      <PredictionForm modelInfo={modelInfo} onPredict={handlePredict} />
      <PredictionsList predictions={predictions} />
      <MetricsGraph metrics={metrics} />
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root')); 